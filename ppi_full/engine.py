"""训练协议 + 损失栈 + 评测（v2.3）。
本轮修复：N1（_predict/mc 输出 squeeze(-1) → 全下游 1-D，消除 acc 广播错算与 ece/温度拟合崩溃）。"""
import copy, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef)
from scipy.stats import spearmanr
from config import Paths, TrainCfg, save_json, model_cfg_from_dict
from data import (FeatureStore, PairDataset, BucketSampler, make_collate,
                  load_pairs, validate_splits)
from model import PPIModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_dev(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, device) for k, v in x.items()}
    return x

def bern_kl(p, q, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())

def smooth(y, eps):
    return y * (1 - eps) + eps / 2

def ece(probs, y, bins=15):
    conf = np.asarray(probs)                              # 1-D（N1）
    pred = conf >= 0.5
    y = np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    tot, N = 0.0, len(y)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        tot += m.sum() / N * abs(y[m].mean() - conf[m].mean())
    return float(tot)

def compute_metrics(y, probs):
    y = np.asarray(y).astype(int)                         # [N]
    probs = np.asarray(probs)                             # [N]（N1）
    pred = (probs >= 0.5).astype(int)
    return dict(
        auprc=float(average_precision_score(y, probs)),
        auroc=float(roc_auc_score(y, probs)),
        acc=float((pred == y).mean()),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)),
        brier=float(((probs - y) ** 2).mean()),
        ece=ece(probs, y),
    )

def probs_to_logits(p, eps=1e-6):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    return np.log(p / (1 - p))

def fit_temperature(val_logits, val_y, device):
    z = torch.tensor(val_logits, dtype=torch.float32, device=device)   # [N]
    y = torch.tensor(val_y, dtype=torch.float32, device=device)        # [N]
    logT = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())

@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for batch in loader:                                  # 固定顺序（配对统计前提）
        batch = to_dev(batch, device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
            logits = model(batch)["logits"].float()
        ps.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())   # N1：[B,1]→[B]
        ys.append(batch["y"].cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys)         # 均 [N]

def _loaders(paths, store, tcfg, seed):
    coll = make_collate(store)
    tr = load_pairs(os.path.join(paths.data_dir, "train.csv"), tcfg.date_col or None, tcfg.cutoff or None)
    va = load_pairs(os.path.join(paths.data_dir, "val.csv"))
    te = load_pairs(os.path.join(paths.data_dir, "test.csv"))
    validate_splits({"train": tr, "val": va, "test": te})              # P7
    tr_ds, va_ds, te_ds = PairDataset(tr, store), PairDataset(va, store), PairDataset(te, store)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pin = dev == "cuda"
    tr_loader = DataLoader(tr_ds, batch_sampler=BucketSampler(tr_ds, store, tcfg.bs, seed,
                                                              tcfg.train_subset_frac),
                           collate_fn=coll, num_workers=tcfg.num_workers, pin_memory=pin)
    va_loader = DataLoader(va_ds, batch_size=tcfg.bs, shuffle=False, collate_fn=coll,
                           num_workers=tcfg.num_workers, pin_memory=pin)
    te_loader = DataLoader(te_ds, batch_size=tcfg.bs, shuffle=False, collate_fn=coll,
                           num_workers=tcfg.num_workers, pin_memory=pin)
    return tr_loader, va_loader, te_loader, te, dev

def train_one_run(level, seed, mcfg, tcfg, paths, tag=None, adapter_ckpt=None):
    tag = tag or level
    set_seed(seed)
    store = FeatureStore(paths.feat_dir)
    tr_loader, va_loader, te_loader, te_df, device = _loaders(paths, store, tcfg, seed)

    model = PPIModel(mcfg, adapter_ckpt=adapter_ckpt if mcfg.use_ssl else None).to(device)
    print(f"[{tag}/seed{seed}] params={model.report_params()}")
    is_reim = mcfg.kind == "reim"
    rdrop_a = 0.0 if is_reim else tcfg.rdrop_alpha
    ls_eps = 0.0 if is_reim else tcfg.label_smooth

    run_dir = os.path.join(paths.out_dir, tag, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    save_json({"model": json.loads(json.dumps(mcfg, default=lambda o: getattr(o, "__dict__", str(o)))),
               "level": level, "tag": tag, "seed": seed},
              os.path.join(paths.out_dir, tag, f"seed{seed}", "cfg.json"))

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.wd)
    steps = tcfg.max_epochs * len(tr_loader)
    warm = max(1, int(tcfg.warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    best_val, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, tcfg.max_epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in tr_loader:
            batch = to_dev(batch, device)                 # P0-1
            y = batch["y"]
            y_sm = smooth(y, ls_eps)
            mix_pair = None                               # D1：每 step 采样一次，双前向共享
            if (not is_reim) and tcfg.mixup_prob > 0 and random.random() < tcfg.mixup_prob:
                lam = float(np.random.beta(tcfg.mixup_alpha, tcfg.mixup_alpha))
                perm = torch.randperm(y.shape[0])
                mix_pair = (perm.to(device), lam)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=tcfg.bf16 and device == "cuda"):
                out1 = model(batch, y=y_sm, mix_pair=mix_pair)
                t1 = out1["y_mix"] if out1["y_mix"] is not None else y_sm
                loss = F.binary_cross_entropy_with_logits(out1["logits"].float().squeeze(-1), t1)
                if rdrop_a > 0:
                    out2 = model(batch, y=y_sm, mix_pair=mix_pair)
                    t2 = out2["y_mix"] if out2["y_mix"] is not None else y_sm
                    p1 = torch.sigmoid(out1["logits"].float())
                    p2 = torch.sigmoid(out2["logits"].float())
                    kl = 0.5 * (bern_kl(p1, p2) + bern_kl(p2, p1)) if tcfg.rdrop_sym \
                        else bern_kl(p1, p2)
                    loss = 0.5 * (loss + F.binary_cross_entropy_with_logits(out2["logits"].float().squeeze(-1), t2)) \
                        + rdrop_a * kl.mean()             # D2：KL 权重=α
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        vp, vy = _predict(model, va_loader, device)
        val_auprc = average_precision_score(vy.astype(int), vp)
        mark = ""
        if val_auprc > best_val:
            best_val, best_ep, bad = val_auprc, ep, 0
            best_state = copy.deepcopy(model.state_dict())
            mark = " *"
            torch.save({"best_ep": best_ep, "val_auprc": best_val,
                        "state_dict": model.state_dict()},
                       os.path.join(run_dir, "best_ckpt.pt"))   # 纯 I/O
        else:
            bad += 1
        print(f"[{tag}/seed{seed}] ep{ep:02d} loss={tot / max(nb, 1):.4f} "
              f"val_auprc={val_auprc:.4f} ({time.time() - t0:.0f}s){mark}")
        if bad >= tcfg.patience:
            print("early stop")
            break

    model.load_state_dict(best_state)
    tp, ty = _predict(model, te_loader, device)           # 1-D
    vp2, vy2 = _predict(model, va_loader, device)
    T = fit_temperature(probs_to_logits(vp2), vy2, device)
    probs_cal = 1.0 / (1.0 + np.exp(-probs_to_logits(tp) / T))
    m, m_cal = compute_metrics(ty, tp), compute_metrics(ty, probs_cal)

    out = os.path.join(paths.out_dir, tag, f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "ckpt.pt"))
    np.savez(os.path.join(out, "predictions.npz"), probs=tp, probs_cal=probs_cal,
             logits=probs_to_logits(tp), labels=ty, T=T,
             pid_a=te_df.pid_a.values, pid_b=te_df.pid_b.values)
    save_json(dict(tag=tag, level=level, seed=seed, best_epoch=best_ep, val_auprc=best_val,
                   test=m, test_calibrated=m_cal, T=T),
              os.path.join(out, "metrics.json"))
    print(f"[{tag}/seed{seed}] TEST auprc={m['auprc']:.4f} acc={m['acc']:.4f} "
          f"| cal T={T:.3f} auprc={m_cal['auprc']:.4f}")
    return best_val

def mc_dropout_eval(paths, tag, seed, T=10):
    """MC-Dropout：仅激活 Dropout 模块；输出 1-D（N1）。"""
    run_dir = os.path.join(paths.out_dir, tag, f"seed{seed}")
    for name in ("cfg.json", "ckpt.pt", "predictions.npz"):   # D4：缺产物就别往下走
        if not os.path.exists(os.path.join(run_dir, name)):
            raise FileNotFoundError(f"{tag}/seed{seed} 缺 {name}，无法做 MC-Dropout")
    cfg = json.load(open(os.path.join(run_dir, "cfg.json")))
    mcfg = model_cfg_from_dict(cfg["model"])
    set_seed(seed)
    store = FeatureStore(paths.feat_dir)
    tcfg = TrainCfg()                                     # 推理默认 bs，8GB 安全
    _, _, te_loader, _, device = _loaders(paths, store, tcfg, seed)
    model = PPIModel(mcfg).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu"))
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
    outs = []
    with torch.no_grad():                                 # 推理无需建图：否则 .numpy() 直接报错
        for _ in range(T):
            ps = []
            for batch in te_loader:
                batch = to_dev(batch, device)
                with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                    logits = model(batch)["logits"].float()
                ps.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())   # N1
            outs.append(np.concatenate(ps))
    P = np.stack(outs)                                    # [T,N]
    mean, var = P.mean(0), P.var(0)
    y = np.load(os.path.join(run_dir, "predictions.npz"))["labels"]
    err = np.abs(mean - y)
    rho = float(spearmanr(var, err).statistic)
    m = compute_metrics(y, mean)
    np.savez(os.path.join(run_dir, "mc_dropout.npz"), mc_mean=mean, mc_var=var)
    save_json(dict(tag=tag, seed=seed, T=T, test_mc_mean=m, spearman_var_abserr=rho),
              os.path.join(run_dir, "mc_metrics.json"))
    print(f"[{tag}/seed{seed}] MC-Dropout T={T}: auprc={m['auprc']:.4f} "
          f"spearman(var,|err|)={rho:.3f}")
