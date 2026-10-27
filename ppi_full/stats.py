"""统计检验：Wilcoxon(exact) + Cohen's dz + 精确 McNemar + Stouffer + bootstrap CI。

两条约定（修复旧实现的两个坑）：
1) seed 按目录名 `seed<N>` 的**数值**对齐，不再依赖 glob 顺序直接 zip
   （旧实现：R2 有 2 个 seed、R1 有 3 个时会错配成 R2/1234↔R1/1234、R2/42↔R1/2024）。
2) metrics.json 容忍两种契约——engine 产出的嵌套 `{"test": {"auprc": ...}}` 与外部基线的
   扁平 `{"test_auprc": ...}`；predictions.npz 缺失时自动降级（跳过 McNemar / 配对 bootstrap）。
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score


def _seed_of_dir(d):
    name = os.path.basename(os.path.abspath(d))
    if not name.startswith("seed"):
        return None
    try:
        return int(name[4:])
    except ValueError:
        # seed<非数字>：仍可统计，用稳定负序号占位（保证可排序、不与其他 seed 冲突）
        return -(abs(hash(name)) % 10 ** 6)


def _auprc_from_metrics(m):
    t = m.get("test")
    if isinstance(t, dict) and t.get("auprc") is not None:
        return float(t["auprc"])
    for k in ("test_auprc", "auprc"):
        if m.get(k) is not None:
            return float(m[k])
    return None


def _acc_from_metrics(m):
    t = m.get("test")
    if isinstance(t, dict) and t.get("acc") is not None:
        return float(t["acc"])
    for k in ("test_acc", "acc"):
        if m.get(k) is not None:
            return float(m[k])
    return float("nan")


def load_runs(out_dir, tag):
    """→ {seed: {seed, dir, metrics, preds|None, auprc}}，按 seed 数值键，不依赖 glob 顺序。
    只含“能取到 AUPRC”的 seed；被中断的半成品（如只有 cfg.json 的 seed2024）会被跳过并告警。"""
    runs, skipped = {}, []
    for d in sorted(glob.glob(os.path.join(out_dir, tag, "seed*"))):
        if not os.path.isdir(d):
            continue
        seed = _seed_of_dir(d)
        if seed is None:
            continue
        mp = os.path.join(d, "metrics.json")
        pp = os.path.join(d, "predictions.npz")
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        preds = np.load(pp, allow_pickle=True) if os.path.exists(pp) else None
        r = dict(seed=seed, dir=d, metrics=m, preds=preds, auprc=_auprc_from_metrics(m))
        if r["auprc"] is None and preds is not None and "probs" in preds and "labels" in preds:
            r["auprc"] = float(average_precision_score(
                np.asarray(preds["labels"]).astype(int), np.asarray(preds["probs"], float)))
        if r["auprc"] is None:
            skipped.append(f"seed{seed}")
            continue
        runs[seed] = r
    if skipped:
        print(f"[warn] {tag}: 跳过不完整 run {skipped}（无 test.auprc 且无 predictions.npz）")
    if not runs:
        raise FileNotFoundError(f"{tag}: {os.path.join(out_dir, tag)} 下没有可用的 seed 结果")
    withp = [r for r in runs.values()
             if r["preds"] is not None and "labels" in r["preds"]]
    for r in withp[1:]:                                   # 配对统计前提：测试标签顺序一致
        assert np.array_equal(withp[0]["preds"]["labels"], r["preds"]["labels"]), \
            f"{tag}: test label order differs across seeds"
    return runs


def bootstrap_auprc_ci(y, p, B=1000, seed=0):
    """单个 run 的 AUPRC 95% CI（测试样本重采样）。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    n = len(y)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, y[i].size):
            continue
        vals.append(average_precision_score(y[i], p[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_mean_ci(runs, B=1000, seed=0):
    """跨 seed **均值**的 95% CI：对测试样本重采样，每个 run 在同一批样本上算 AUPRC 再取均值。
    这样 CI 与上报的 mean±std 是同一口径（旧实现拿单 seed 的 CI 去配跨 seed 均值）。"""
    ok = [r for r in runs if r["preds"] is not None and "probs" in r["preds"]
          and "labels" in r["preds"]]
    if not ok:                                            # 没有任何预测文件：只能按 seed 重采样
        if len(runs) < 2:
            return float("nan"), float("nan")
        v = np.array([r["auprc"] for r in runs], float)
        rng = np.random.default_rng(seed)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    y = np.asarray(ok[0]["preds"]["labels"]).astype(int)
    ps = [np.asarray(r["preds"]["probs"], float) for r in ok]
    n = len(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, n):
            continue
        vals.append(float(np.mean([average_precision_score(y[i], p[i]) for p in ps])))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar_exact(p1, p2, labels):
    c1 = (np.asarray(p1) >= 0.5) == (np.asarray(labels) == 1)
    c2 = (np.asarray(p2) >= 0.5) == (np.asarray(labels) == 1)
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    if b + c == 0:
        return 1.0, b, c
    return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), b, c


def stouffer(ps):
    ps = np.clip(np.asarray(ps, float), 1e-12, 1 - 1e-12)
    z = stats.norm.isf(ps)
    Z = float(np.sum(z) / np.sqrt(len(z)))
    return float(stats.norm.sf(Z))


def compare(out_dir, tag_a, tag_b):
    """返回 a 相对 b 的增量（delta>0 ⇒ a 更好），只用两者**共有的 seed** 配对。"""
    ra, rb = load_runs(out_dir, tag_a), load_runs(out_dir, tag_b)
    seeds = sorted(set(ra) & set(rb))
    if not seeds:
        raise AssertionError(f"{tag_a} 与 {tag_b} 没有共同 seed："
                             f"{sorted(ra)} vs {sorted(rb)}")
    A = [ra[s] for s in seeds]
    B = [rb[s] for s in seeds]
    a = np.array([r["auprc"] for r in A], float)
    b = np.array([r["auprc"] for r in B], float)
    n = len(seeds)
    d = a - b
    if n >= 2 and np.any(np.abs(d) > 0):
        try:
            w_p = float(stats.wilcoxon(a, b, method="exact").pvalue)
        except ValueError:                                # 全等值等退化情形
            w_p = float("nan")
    else:
        w_p = float("nan")
    dz = float(d.mean() / d.std(ddof=1)) if n > 1 and d.std(ddof=1) > 0 else float("nan")
    mps, mseed = [], []
    for s in seeds:
        pa, pb = ra[s]["preds"], rb[s]["preds"]
        if pa is None or pb is None or "probs" not in pa or "probs" not in pb:
            continue
        mps.append(mcnemar_exact(pa["probs"], pb["probs"], pa["labels"])[0])
        mseed.append(s)
    lo, hi = bootstrap_mean_ci(A)
    return dict(delta=float(a.mean() - b.mean()),
                auprc_a=f"{a.mean():.4f}±{a.std():.4f}",
                auprc_b=f"{b.mean():.4f}±{b.std():.4f}",
                n_seeds=n, seeds=seeds,
                wilcoxon_p=w_p, cohens_dz=dz,
                mcnemar_per_seed=[round(x, 4) for x in mps],
                mcnemar_seeds=mseed,
                mcnemar_stouffer_p=stouffer(mps) if mps else float("nan"),
                boot95_a=(lo, hi))


def summarize(out_dir, tags, baseline=None, out_csv=None):
    rows = []
    if baseline:
        for t in tags:
            if t == baseline:
                continue
            try:
                rows.append(dict(comparison=f"{t} vs {baseline}",
                                 **compare(out_dir, t, baseline)))
            except Exception as e:                        # 缺 run / 缺 preds / 无共同 seed 都只跳过
                print(f"[skip] {t} vs {baseline}: {e}")
    else:
        for a, b in zip(tags[:-1], tags[1:]):
            try:
                rows.append(dict(comparison=f"{b} vs {a}", **compare(out_dir, b, a)))
            except Exception as e:
                print(f"[skip] {b} vs {a}: {e}")
    for t in tags:
        try:
            runs = load_runs(out_dir, t)
        except Exception as e:
            print(f"[skip] {t}: {e}")
            continue
        rs = [runs[s] for s in sorted(runs)]
        lo, hi = bootstrap_mean_ci(rs)
        au = np.array([r["auprc"] for r in rs], float)
        rows.append(dict(tag=t, n_seeds=len(rs), seeds=sorted(runs),
                         auprc=f"{au.mean():.4f}±{au.std():.4f}",
                         boot95=(lo, hi),
                         acc=f"{np.nanmean([_acc_from_metrics(r['metrics']) for r in rs]):.4f}"))
    if out_csv:
        try:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        except OSError as e:                              # workdir 只读时不要让统计结果丢失
            print(f"[warn] summary.csv 写入失败（{e}）；结果仅打印到 stdout")
    return rows
