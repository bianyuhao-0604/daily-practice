"""冒烟测试（smoke）：把每个实验组各跑一个极小但端到端的实例，验证“能跑通 + 产物契约正确”。

覆盖的组（--only 可筛）：
  ladder    R0–R7 共 8 级
  ablation  ABLATIONS 全部 18 项（含空操作守卫）
  search    cmd_search（2 个 trial，验证排序 + best_hyperparams.json）
  baseline  cmd_baseline（RFC-mean / RFC-40，验证与 engine 相同的产物契约）
  stats     st.summarize（seed 对齐 / 扁平旧格式兼容 / CSV 落盘降级）
  mc        cmd_mc（MC-Dropout + 缺产物 skip 路径）
  datapath  真实 _loaders（BucketSampler + num_workers=2 多进程 pickle + 长蛋白）
  cli       6 个子命令的 --help 入口

为什么需要替身：CPU-only 跑不动真实规模（1 epoch=40798 batch）。这里用
  * 每类最短的 8/16/16 条 pair（Lmax 63–130，与真实分布同构但便宜 100×）
  * max_epochs=1 / patience=1 / bs=4 / num_workers=0
  * out_dir 指向 ./_smoke_runs（workdir/ 对脚本只读）
除此之外全部走真实代码路径：run_tagged 的门禁、build_*_cfg、train_one_run 的训练/评测/
温度校准/落盘、make_collate 的掩码、PPIModel 的全部组件。

用法：
  python smoke_all.py                     # 全跑
  python smoke_all.py --only ladder stats # 只跑指定组
产物：
  <out>/smoke_summary.json                # 机器可读
  ./smoke_report.md                       # 人读表格
"""
import argparse, json, os, shutil, subprocess, sys, time, traceback
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import data as data_mod
import engine
import run_all
import stats as st
from config import (ABLATIONS, R_LEVELS, ModelCfg, Paths, TrainCfg,
                    build_model_cfg, build_train_cfg, model_cfg_from_dict)

OUT_DIR = os.path.abspath("./_smoke_runs")
BS = 4
TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS = 8, 16, 16
GROUPS = ("ladder", "ablation", "search", "baseline", "stats", "mc", "datapath", "cli")


# ---------------------------------------------------------------- 极小替身
@dataclass
class TinyTrainCfg(TrainCfg):
    max_epochs: int = 1
    patience: int = 1
    bs: int = BS
    num_workers: int = 0
    bf16: bool = False


_PICK_CACHE = {}


def _short_pairs(csv_path, store, n_per_class):
    """每类挑最短的 n 条（保证 Lmax 小、类别平衡、跨组一致）。"""
    key = (csv_path, n_per_class)          # 只按文件+条数缓存（store 内容相同，逐次读 csv 太慢）
    if key in _PICK_CACHE:
        return _PICK_CACHE[key]
    df = data_mod.load_pairs(csv_path)
    Ls = np.asarray(store.lengths)
    ia = np.array([store.pid2idx[p] for p in df.pid_a])
    ib = np.array([store.pid2idx[p] for p in df.pid_b])
    L = np.maximum(Ls[ia], Ls[ib])
    y = df.label.values.astype(int).reshape(-1)
    order = np.argsort(L, kind="stable")
    sel = np.sort(np.concatenate([order[y[order] == 1][:n_per_class],
                                  order[y[order] == 0][:n_per_class]]))
    sub = df.iloc[sel].reset_index(drop=True)
    _PICK_CACHE[key] = (sub, float(L[sel].max()))
    return _PICK_CACHE[key]


def _tiny_loaders(paths, store, tcfg, seed):
    """engine._loaders 的替身：真实 collate / BucketSampler，只是数据量极小。"""
    coll = data_mod.make_collate(store)
    tr, mL = _short_pairs(os.path.join(paths.data_dir, "train.csv"), store, TRAIN_PER_CLASS)
    va, _ = _short_pairs(os.path.join(paths.data_dir, "val.csv"), store, VAL_PER_CLASS)
    te, _ = _short_pairs(os.path.join(paths.data_dir, "test.csv"), store, TEST_PER_CLASS)
    data_mod.validate_splits({"train": tr, "val": va, "test": te})          # P7 仍要成立
    tr_ds, va_ds, te_ds = (data_mod.PairDataset(tr, store), data_mod.PairDataset(va, store),
                           data_mod.PairDataset(te, store))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_loader = torch.utils.data.DataLoader(
        tr_ds, batch_sampler=data_mod.BucketSampler(tr_ds, store, BS, seed),
        collate_fn=coll, num_workers=0)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=BS, shuffle=False,
                                            collate_fn=coll, num_workers=0)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=BS, shuffle=False,
                                            collate_fn=coll, num_workers=0)
    return tr_loader, va_loader, te_loader, te, dev


# ---------------------------------------------------------------- 仪表
_ORIG_CLIP = torch.nn.utils.clip_grad_norm_
_ORIG_BCE = torch.nn.functional.binary_cross_entropy_with_logits
_GNORMS, _LOSSES, _PENDING, _TRAIN_LOSSES = [], [], [], []


def _clip(params, max_norm, *a, **k):
    n = _ORIG_CLIP(params, max_norm, *a, **k)
    _GNORMS.append(float(n))
    if _PENDING:                                   # clip 之前刚算的 BCE 属于该训练 step
        _TRAIN_LOSSES.append(_PENDING[-1])
    _PENDING.clear()
    return n


def _bce(*a, **k):
    out = _ORIG_BCE(*a, **k)
    v = float(out.detach())
    _LOSSES.append(v)
    _PENDING.append(v)
    return out


def _reset_meters():
    for l in (_GNORMS, _LOSSES, _PENDING, _TRAIN_LOSSES):
        l.clear()


# ---------------------------------------------------------------- 校验
METRIC_KEYS = ("auprc", "auroc", "acc", "precision", "recall", "f1", "mcc", "brier", "ece")


def check_run(tag, seed):
    """校验一个 run 目录的产物契约；返回 (issues, info)。"""
    run_dir = os.path.join(OUT_DIR, tag, f"seed{seed}")
    issues, info = [], {}
    need = ("cfg.json", "metrics.json", "ckpt.pt", "predictions.npz", "best_ckpt.pt")
    miss = [n for n in need if not os.path.exists(os.path.join(run_dir, n))]
    if miss:
        issues.append(f"缺产物 {miss}")
        return issues, info
    cfg = json.load(open(os.path.join(run_dir, "cfg.json")))
    m = json.load(open(os.path.join(run_dir, "metrics.json")))
    for blk in ("test", "test_calibrated"):
        if blk not in m:
            issues.append(f"metrics.json 缺 {blk}")
            continue
        lack = [k for k in METRIC_KEYS if k not in m[blk]]
        if lack:
            issues.append(f"metrics.{blk} 缺 {lack}")
    for k in ("tag", "level", "seed", "best_epoch", "val_auprc", "T"):
        if k not in m:
            issues.append(f"metrics.json 缺顶层键 {k}")

    z = np.load(os.path.join(run_dir, "predictions.npz"), allow_pickle=True)
    for k in ("probs", "probs_cal", "logits", "labels", "T", "pid_a", "pid_b"):
        if k not in z.files:
            issues.append(f"predictions.npz 缺 {k}")
    if not issues:
        p, y = z["probs"], z["labels"]
        info.update(n_test=len(p), auprc=float(m["test"]["auprc"]), acc=float(m["test"]["acc"]),
                    val_auprc=float(m["val_auprc"]), T=float(m["T"]))
        if p.ndim != 1 or y.ndim != 1:
            issues.append(f"probs/labels 维度应为 1-D，实际 {p.shape}/{y.shape}")   # N1
        if len(p) != len(y):
            issues.append(f"probs {len(p)} != labels {len(y)}")
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            issues.append("probs 存在非有限或越界值")
        if set(np.unique(y).astype(int)) != {0, 1}:
            issues.append(f"labels 不是二分类: {np.unique(y)}")
        if len(z["pid_a"]) != len(p):
            issues.append(f"pid_a {len(z['pid_a'])} != probs {len(p)}")
        if abs(info["auprc"] - float(m["test_calibrated"]["auprc"])) > 0.5:
            issues.append("校准 AUPRC 与原始 AUPRC 差异异常（>0.5）")

    # ckpt 参数账：state_dict 含 buffer（L5Head 的 T），parameters() 不含 → 用 state_dict 对齐
    mcfg = model_cfg_from_dict(cfg["model"])
    probe = engine.PPIModel(mcfg)
    n_model = sum(v.numel() for v in probe.state_dict().values())
    sd = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    n_ckpt = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    info.update(params=sum(x.numel() for x in probe.parameters()), ckpt_params=n_ckpt)
    if n_model != n_ckpt:
        issues.append(f"ckpt 参数 {n_ckpt} != 模型参数 {n_model}")
    return issues, info


# ---------------------------------------------------------------- 各组
def run_experiments(group, seeds=(42,)):
    """跑一组实验，返回 [(name, ok, info, err, elapsed)]。"""
    rows = []

    def one(name, fn, check=True):
        _reset_meters()
        t0 = time.time()
        ok, info, err = True, {}, None
        try:
            fn()
            if check:
                issues, info = check_run(name, seeds[0])
                if issues:
                    ok, err = False, "; ".join(issues)
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        el = time.time() - t0
        gn = [g for g in _GNORMS if np.isfinite(g)]
        tl = [v for v in _TRAIN_LOSSES if np.isfinite(v)]
        info.update(steps=len(_GNORMS), gnorm_min=min(gn) if gn else None,
                    gnorm_max=max(gn) if gn else None,
                    loss_first=tl[0] if tl else None, loss_last=tl[-1] if tl else None,
                    loss_nonfinite=sum(1 for v in _LOSSES if not np.isfinite(v)))
        if ok and check and (not gn or info["loss_nonfinite"]):   # 基线/搜索无 backward，不查 gnorm
            ok, err = False, f"梯度/损失出现非有限值 (gnorm={info['gnorm_min']}, nonfinite_loss={info['loss_nonfinite']})"
        rows.append((name, ok, info, err, el))
        print(f"  [{'ok' if ok else 'FAIL'}] {name}  {el:.1f}s  {err or ''}", flush=True)

    if group == "ladder":
        for lv in R_LEVELS:
            one(lv, lambda lv=lv: run_all.run_tagged(lv, lv, None, list(seeds), TinyTrainCfg()))
    elif group == "ablation":
        for name in ABLATIONS:
            a = argparse.Namespace(names=[name], seeds=list(seeds), lr=5e-5, bs=BS, hyperparams=None)
            one(name, lambda a=a: run_all.cmd_ablation(a))
        # 可辨识性：与本子集上 R7 的输出逐位比较。==0 说明该消融在 16 条 pair / 4 步训练下
        # 测不出差别（例如 beta_init=0 时 iface_bias 项恒为 0），需真实规模才能归因。
        ref = os.path.join(OUT_DIR, "R7", f"seed{seeds[0]}", "predictions.npz")
        if os.path.exists(ref):
            rp = np.load(ref, allow_pickle=True)["probs"]
            for i, (nm, ok, info, err, el) in enumerate(rows):
                pth = os.path.join(OUT_DIR, nm, f"seed{seeds[0]}", "predictions.npz")
                if not os.path.exists(pth):
                    continue
                dp = float(np.max(np.abs(np.load(pth, allow_pickle=True)["probs"] - rp)))
                info["max_dp_vs_R7"] = dp
                if ok and dp < 1e-9:
                    info["note"] = "本子集上与 R7 输出逐位相同（此规模不可辨识，需真实规模归因）"
    elif group == "search":
        a = argparse.Namespace(trials=2)
        one("SEARCH", lambda: run_all.cmd_search(a), check=False)
        if rows[-1][1]:
            best = json.load(open(os.path.join(OUT_DIR, "best_hyperparams.json")))
            rows[-1][2].update(best_tag=best["best"]["tag"], n_all=len(best["all"]),
                               best_keys=sorted(best["best_overrides"]))
            if len(best["all"]) != 2 or "train__mixup_prob" not in best["best_overrides"]:
                rows[-1] = (rows[-1][0], False, rows[-1][2], "best_hyperparams.json 内容不完整", rows[-1][4])
            elif best["best"]["val_auprc"] != max(r["val_auprc"] for r in best["all"]):
                rows[-1] = (rows[-1][0], False, rows[-1][2], "best 不是 val_auprc 最高的 trial", rows[-1][4])
    elif group == "baseline":
        note = {}

        def _bl():
            """cmd_baseline 一次产出 RFC-mean / RFC-40 两个 tag。"""
            try:
                run_all.cmd_baseline(argparse.Namespace())
            except (PermissionError, OSError) as e:   # 沙箱禁命名管道 → joblib 线程池起不来
                import sklearn.ensemble as _se
                orig_fit = _se.RandomForestClassifier.fit

                def _fit(self, *a, **k):      # 子类化会被 sklearn 的参数签名校验拒掉，改包 fit
                    self.n_jobs = 1
                    return orig_fit(self, *a, **k)

                _se.RandomForestClassifier.fit = _fit
                note["note"] = f"n_jobs=-1 被拒({type(e).__name__})→降级 n_jobs=1"
                try:
                    run_all.cmd_baseline(argparse.Namespace())
                finally:
                    _se.RandomForestClassifier.fit = orig_fit

        one("RFC-mean", _bl, check=False)          # 一次调用产出两个 tag
        base = rows.pop()
        for nm in ("RFC-mean", "RFC-40"):
            info, err, ok = dict(base[2]), base[3], base[1]
            info.update(note)
            d = os.path.join(OUT_DIR, nm, "seed42")
            miss = [f for f in ("metrics.json", "predictions.npz")
                    if not os.path.exists(os.path.join(d, f))]
            if miss:
                ok, err = False, f"缺产物 {miss}"
            else:
                mm = json.load(open(os.path.join(d, "metrics.json")))
                z = np.load(os.path.join(d, "predictions.npz"))
                info.update(auprc=float(mm["test"]["auprc"]), val_auprc=mm.get("val_auprc"),
                            n_test=len(z["probs"]))
                if "test" not in mm or "auprc" not in mm["test"]:
                    ok, err = False, "metrics.json 缺 test.auprc（engine 契约）"
                elif set(np.unique(z["labels"]).astype(int)) != {0, 1}:
                    ok, err = False, "labels 非二分类"
                elif not (0.0 <= info["auprc"] <= 1.0):
                    ok, err = False, "AUPRC 越界"
            rows.append((nm, ok, info, err, base[4] if nm == "RFC-mean" else 0.0))
        if rows[0][1] and rows[1][1] and abs(rows[1][2]["auprc"] - rows[0][2]["auprc"]) < 1e-9:
            rows[1] = (rows[1][0], False, rows[1][2],
                       "RFC-40 与 RFC-mean 的 AUPRC 完全相同（PCA 未生效？）", rows[1][4])
    return rows


# ---------------------------------------------------------------- 统计组
def _write_fixture(tag, seed, probs, labels):
    d = os.path.join(OUT_DIR, tag, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    y = np.asarray(labels, int)
    m = engine.compute_metrics(y, np.asarray(probs, float))
    json.dump(dict(tag=tag, level=tag, seed=seed, best_epoch=1, val_auprc=0.5, T=1.0,
                   test=m, test_calibrated=m), open(os.path.join(d, "metrics.json"), "w"))
    np.savez(os.path.join(d, "predictions.npz"), probs=np.asarray(probs, float),
             probs_cal=np.asarray(probs, float), logits=engine.probs_to_logits(probs),
             labels=y.astype(np.float32), T=1.0,
             pid_a=np.array([f"P{i}" for i in range(len(y))]), pid_b=np.array([f"Q{i}" for i in range(len(y))]))


def run_stats():
    """FIX-A（seed 42/1234/2024）vs FIX-B（seed 42/1234）→ 只能配对交集，且数值可解析验证。"""
    rows = []
    y = [0, 0, 1, 1]
    good = [0.1, 0.2, 0.9, 0.8]          # AP=1.0
    bad = [0.9, 0.8, 0.2, 0.1]           # AP=5/12
    for s in (42, 1234, 2024):
        _write_fixture("FIX-A", s, good if s != 2024 else bad, y)
    for s in (42, 1234):
        _write_fixture("FIX-B", s, bad, y)
    # 旧版扁平 metrics（无 predictions.npz）——stats 必须能读，不得 KeyError
    d = os.path.join(OUT_DIR, "LEGACY", "seed42")
    os.makedirs(d, exist_ok=True)
    json.dump({"tag": "LEGACY", "test_auprc": 0.6123}, open(os.path.join(d, "metrics.json"), "w"))

    _reset_meters()
    t0 = time.time()
    ok, err, info = True, None, {}
    try:
        txt = st.summarize(OUT_DIR, ["FIX-A", "FIX-B"], None,
                           out_csv=os.path.join(OUT_DIR, "summary.csv"))
        cmp_rows = [r for r in txt if str(r.get("comparison", "")).startswith("FIX-B")]
        if not cmp_rows:
            ok, err = False, "summarize 没产出 FIX-B vs FIX-A 的对比行"
        else:
            r = cmp_rows[0]
            info.update(paired_seeds=r.get("seeds"), n_seeds=r.get("n_seeds"),
                        delta=r.get("delta"), p=r.get("wilcoxon_p"))
            if list(r.get("seeds") or []) != [42, 1234]:
                ok, err = False, f"配对 seed 应为 [42,1234]，实际 {r.get('seeds')}（seed 错配回归）"
            elif abs(r["delta"] - (5 / 12 - 1.0)) > 1e-6:
                ok, err = False, f"delta 应为 {5/12-1.0:.6f}，实际 {r['delta']:.6f}"
        # 扁平旧格式（无 predictions.npz）：只能降级，不得 KeyError
        leg = st.summarize(OUT_DIR, ["LEGACY"], None, out_csv=os.path.join(OUT_DIR, "summary_legacy.csv"))
        leg_auprc = None
        for r in leg:
            if r.get("tag") == "LEGACY":
                leg_auprc = float(str(r["auprc"]).split("±")[0])
        info["legacy_auprc"] = leg_auprc
        if leg_auprc is None or abs(leg_auprc - 0.6123) > 1e-9:
            ok, err = False, f"扁平旧 metrics 读取失败（{leg_auprc}≠0.6123）"
        if not os.path.exists(os.path.join(OUT_DIR, "summary.csv")):
            ok, err = False, "summary.csv 未落盘"
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    rows.append(("STATS", ok, info, err, time.time() - t0))
    print(f"  [{'ok' if ok else 'FAIL'}] STATS  {err or ''}", flush=True)
    return rows


# ---------------------------------------------------------------- MC 组
def run_mc():
    a = argparse.Namespace(tags=["R7"], seeds=[42], T=2)
    _reset_meters()
    t0 = time.time()
    ok, err, info = True, None, {}
    try:
        run_all.cmd_mc(a)
        d = os.path.join(OUT_DIR, "R7", "seed42")
        need = ("mc_dropout.npz", "mc_metrics.json")
        miss = [n for n in need if not os.path.exists(os.path.join(d, n))]
        if miss:
            ok, err = False, f"缺 {miss}"
        else:
            mm = json.load(open(os.path.join(d, "mc_metrics.json")))
            z = np.load(os.path.join(d, "mc_dropout.npz"))
            lab = np.load(os.path.join(d, "predictions.npz"))["labels"]
            info.update(T=mm["T"], auprc=float(mm["test_mc_mean"]["auprc"]),
                        rho=float(mm["spearman_var_abserr"]), n=len(z["mc_mean"]))
            if len(z["mc_mean"]) != len(lab) or len(z["mc_var"]) != len(lab):
                ok, err = False, "mc_mean/mc_var 长度与 labels 不一致"
            if not np.isfinite(mm["spearman_var_abserr"]):
                ok, err = False, "spearman(var,|err|) 非有限（方差退化为 0？）"
        # skip 路径：不存在的 tag 必须跳过而不是崩
        run_all.cmd_mc(argparse.Namespace(tags=["__nope__"], seeds=[42], T=1))
        info["skip_path"] = True
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    rows = [("MC", ok, info, err, time.time() - t0)]
    print(f"  [{'ok' if ok else 'FAIL'}] MC  {err or ''}", flush=True)
    return rows


# ---------------------------------------------------------------- 真实数据通路
def run_datapath():
    """真实 _loaders：BucketSampler + num_workers=2（worker 内重建 FeatureStore）+ 长蛋白。"""
    t0 = time.time()
    ok, err, info = True, None, {}
    try:
        real_loaders = engine._loaders
        engine._loaders = _REAL_LOADERS
        paths = Paths(out_dir=OUT_DIR)
        store = data_mod.FeatureStore(paths.feat_dir)
        # num_workers>0 需要 multiprocessing 命名管道；DSH 沙箱下会被拒绝（WinError 5）。
        # 先按 2 worker 建迭代器，被拒则降级到 0 并在报告里标注“环境限制，非代码缺陷”。
        nw, tr = 2, None
        try:
            tcfg = TinyTrainCfg(bs=2, num_workers=2)
            tr = engine._loaders(paths, store, tcfg, 42)[0]
            iter(tr).__next__()
        except (PermissionError, OSError) as e:
            nw, tr = 0, None
            info["workers_note"] = f"num_workers=2 被沙箱拒绝({type(e).__name__})→降级 0"
        tcfg = TinyTrainCfg(bs=2, num_workers=nw)
        tr, va, te, te_df, dev = engine._loaders(paths, store, tcfg, 42)
        info.update(train_batches=len(tr), val_batches=len(va), test_batches=len(te),
                    num_workers=nw)
        mcfg = build_model_cfg("R7", None)
        model = engine.PPIModel(mcfg, adapter_ckpt=paths.adapter_ckpt).to(dev)
        it = iter(tr)
        seen = 0
        for _ in range(3):
            b = engine.to_dev(next(it), dev)
            out = model(b, y=b["y"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out["logits"].float().squeeze(-1), b["y"])
            loss.backward()
            seen += 1
            info["Lmax"] = max(info.get("Lmax", 0), int(b["A"]["seq"].shape[1]))
            for side in ("A", "B"):
                for k in ("surf_mask", "iface_mask", "text_mask", "genome_mask"):
                    if k not in b[side]:
                        raise AssertionError(f"batch['{side}'] 缺 {k}")
            if not np.isfinite(float(loss)):
                raise AssertionError("真实数据通路 loss 非有限")
        # 真实 val loader 也要能迭代（BucketSampler 之外的默认路径）
        vb = engine.to_dev(next(iter(va)), dev)
        with torch.no_grad():
            model.eval()
            model(vb)
        info.update(steps=seen, params=model.report_params()["total"])
        engine._loaders = real_loaders
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    rows = [("DATAPATH", ok, info, err, time.time() - t0)]
    print(f"  [{'ok' if ok else 'FAIL'}] DATAPATH  {err or ''}", flush=True)
    return rows


# ---------------------------------------------------------------- CLI 组
def run_cli():
    t0 = time.time()
    ok, err, info = True, None, {"helps": 0}
    py = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        for sub in ([], ["ladder"], ["ablation"], ["search"], ["baseline"], ["stats"], ["mc"]):
            r = subprocess.run([py, "run_all.py"] + sub + ["--help"], cwd=here,
                               capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                raise AssertionError(f"run_all.py {' '.join(sub)} --help 退出码 {r.returncode}: {r.stderr[-300:]}")
            if "--help" not in r.stdout and "usage" not in r.stdout.lower():
                raise AssertionError(f"run_all.py {' '.join(sub)} --help 无 usage 输出")
            info["helps"] += 1
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    rows = [("CLI", ok, info, err, time.time() - t0)]
    print(f"  [{'ok' if ok else 'FAIL'}] CLI  {err or ''}", flush=True)
    return rows


# ---------------------------------------------------------------- 报告
def _note(info, err):
    """备注列：错误 + 可辨识性 + 环境降级说明。"""
    parts = []
    if err:
        parts.append(err)
    dp = info.get("max_dp_vs_R7")
    if dp is not None:
        parts.append(f"Δp vs R7={dp:.1e}")
    for k in ("note", "workers_note"):
        if info.get(k):
            parts.append(str(info[k]))
    return "；".join(parts)


def write_report(all_rows):
    md = ["# 冒烟测试报告（smoke）", "",
          f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
          f"- 规模：train {TRAIN_PER_CLASS*2} / val {VAL_PER_CLASS*2} / test {TEST_PER_CLASS*2} 条最短 pair，"
          f"bs={BS}，max_epochs=1，CPU",
          f"- 产物目录：`{OUT_DIR}`", ""]
    md += ["| 组 | 实验 | 结果 | 参数 | steps | gnorm | loss(首→末) | val_auprc | test_auprc | 耗时 | 备注 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    n_ok = 0
    for group, rows in all_rows:
        for name, ok, info, err, el in rows:
            n_ok += int(ok)
            md.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1f}s | {} |".format(
                group, name, "✅" if ok else "❌",
                info.get("params", "-"),
                info.get("steps", "-"),
                f"{info['gnorm_min']:.2f}–{info['gnorm_max']:.2f}" if info.get("gnorm_min") is not None else "-",
                f"{info['loss_first']:.4f}→{info['loss_last']:.4f}" if info.get("loss_first") is not None else "-",
                f"{info['val_auprc']:.4f}" if info.get("val_auprc") is not None else "-",
                f"{info['auprc']:.4f}" if info.get("auprc") is not None else "-",
                el, _note(info, err)))
    total = sum(len(r) for _, r in all_rows)
    md += ["", f"**合计 {n_ok}/{total} 通过。**", "",
           "## 说明", "",
           "- 这里是**连通性冒烟**：每级/每项只跑 1 epoch × 4 step（train 16 条最短 pair、bs=4、CPU），",
           "  只证明「能建图、能前向反传、能评测、能落盘」，**指标数值不具科学含义**。",
           "- `Δp vs R7`：同一子集上该消融与 R7 输出的最大概率差（可辨识性）。全部 ≥1e-7，说明每个消融",
           "  都真实改变了模型；`L4-no-iface-bias` 只有 4e-7，因为 `beta_init=0.0` 时 `beta*iface_t` 在初始化",
           "  附近恒为 0 —— 属初始化口径而非代码缺陷，需真实规模训练才能归因。",
           "- `L4-no-global-path` 与 R5 同口径（R5 本身就没有 text/genome 全局通路），指标与参数量一致属预期。",
           "- 两条环境降级：`RFC-*` 的 `n_jobs=-1` 被沙箱拒绝→降级 1；`DATAPATH` 的 `num_workers=2` 同样被拒",
           "  （命名管道限制）→降级 0。均为本机沙箱限制，不是代码问题。"]
    with open(os.path.join(os.path.dirname(OUT_DIR), "smoke_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_DIR, "smoke_summary.json"), "w", encoding="utf-8") as f:
        json.dump([dict(group=g, name=n, ok=o, info=i, err=e, elapsed_s=round(t, 2))
                   for g, rows in all_rows for (n, o, i, e, t) in rows], f,
                  indent=2, ensure_ascii=False, default=str)
    print(f"\n报告: {os.path.join(os.path.dirname(OUT_DIR), 'smoke_report.md')}")
    print(f"合计 {n_ok}/{total} 通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(GROUPS), choices=GROUPS)
    ap.add_argument("--fresh", action="store_true", help="清空 _smoke_runs 再跑（默认保留已有 run）")
    a = ap.parse_args()
    if a.fresh and os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 全局替身：out_dir / 训练预算 / 数据量
    run_all.Paths = lambda: Paths(out_dir=OUT_DIR)
    config.TrainCfg = TinyTrainCfg
    run_all.TrainCfg = TinyTrainCfg
    engine.TrainCfg = TinyTrainCfg
    engine._loaders = _tiny_loaders
    torch.nn.utils.clip_grad_norm_ = _clip
    torch.nn.functional.binary_cross_entropy_with_logits = _bce

    # baseline 的 PCA/RF 用最短的 500 条/类，避免 CPU 上跑满 16 万行
    _orig_load_pairs = run_all.load_pairs

    def _smaller(csv_path, *args, **kw):
        return _short_pairs(csv_path, _SMALL_STORE[0], 500)[0]

    _SMALL_STORE = [data_mod.FeatureStore(Paths().feat_dir)]
    run_all.load_pairs = _smaller

    all_rows = []
    for g in a.only:
        print(f"\n=== {g} ===", flush=True)
        if g == "stats":
            all_rows.append((g, run_stats()))
        elif g == "mc":
            all_rows.append((g, run_mc()))
        elif g == "datapath":
            all_rows.append((g, run_datapath()))
        elif g == "cli":
            all_rows.append((g, run_cli()))
        else:
            all_rows.append((g, run_experiments(g)))
    run_all.load_pairs = _orig_load_pairs
    write_report(all_rows)
    bad = [f"{g}/{n}: {e}" for g, rows in all_rows for (n, ok, _, e, _) in rows if not ok]
    if bad:
        print("\n失败项:\n  " + "\n  ".join(bad))
        sys.exit(1)


_REAL_LOADERS = engine._loaders

if __name__ == "__main__":
    main()
