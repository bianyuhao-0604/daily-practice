# -*- coding: utf-8 -*-
"""数据资产总检 check_data_all.py v2（只读，不改任何数据）。

v4 2026-10-22：重复索引自动判定——uniprot_primary_map.txt 背书则 PASS（构建期别名），否则 FAIL。v3 2026-10-22：重复索引/未引用键点名（区分构建期去重别名 vs 特征错位）。v2 2026-10-22：①指纹表改为 state_dict 语义（R1=3,823,875、R2+=4,155,523，+1=L4 内标量 buffer，源机实测校准）；②pid2idx 深检：死键(WARN) vs 被引用越界/重复(FAIL) 分级判定。

检查五层实验资产：
  1) 数据划分  workdir/data/{train,val,test}.csv —— 行数/标签/自环/重复对/P7 蛋白不相交/类别比
  2) 特征库    workdir/features/ —— meta 一致性/形状/ dtype/NaN/Inf/非平凡性/越界 pLDDT/padding 零
  3) SSL 适配器 workdir/ssl/adapter.pt —— 可加载/键与形状/参数量=331,648
  4) runs 锚点  workdir/runs/（--runs 开启）—— 产物齐全/labels 跨 seed 一致/参数指纹
  5) 打包清单  MANIFEST（--manifest 开启）—— 流式 sha256/md5 对账

用法（在代码根目录，即含 workdir/ 的目录）：
  python check_data_all.py                     # 快速体检（每通道采样 512 蛋白，~1 分钟）
  python check_data_all.py --full-nan          # 全库 NaN/Inf 扫描（62GB 约 5-15 分钟）
  python check_data_all.py --runs              # 追加 runs 产物检查
  python check_data_all.py --manifest MANIFEST.csv   # 追加打包清单哈希对账
  python check_data_all.py --root D:\\pii_root  # 指定代码根（默认当前目录）
退出码：0=全部 PASS（WARN 不影响）；1=存在 FAIL（必须先解决再发车）。
报告：写入 <root>/data_check_report.md 与 .json。
"""
import argparse, glob, hashlib, json, os, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None

MAX_LEN = 1022
RES_CH = {"seq": 1280, "str": 1024, "surf": 80}
PROT_CH = {"text": 128, "genome": 256}
EXP_DTYPE = {"seq": "float16", "str": "float16", "surf": "float16",
             "iface": "float16", "plddt": "float16", "text": "float32", "genome": "float32"}
# ckpt 是 state_dict 语义（参数+buffer）：L4 内含 1 个标量 buffer，故 R1=3,823,874+1、R2+=4,155,522+1；
# parameters() 语义仍为审计值 3,823,874 / 4,155,522。源机实测（2026-10-22）验证：R0=1,045,569、
# R1 三 seed=3,823,875、R2=4,155,523，且 R2-R1=331,648=adapter，完全自洽。
EXP_PARAMS = {"R0": 1_045_569, "R1": 3_823_875}
for _lv in ("R2", "R3", "R4", "R5", "R6", "R7"):
    EXP_PARAMS[_lv] = 4_155_523

ITEMS = []
def add(stage, name, status, detail=""):
    ITEMS.append(dict(stage=stage, name=name, status=status, detail=str(detail)))
    mark = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[INFO]"}[status]
    print(f"{mark} {stage:>10s} | {name} | {detail}", flush=True)


def load_mmap(path):
    """mmap 打开（只读 npy 头，不读数据体）。缺失返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        return np.load(path, mmap_mode="r")
    except Exception as e:
        print(f"[FAIL] {'features':>10s} | np.load 失败 | {path}: {repr(e)[:120]}")
        return None


def check_splits(root):
    S = "data"
    ddir = os.path.join(root, "workdir", "data")
    dfs = {}
    for name in ("train", "val", "test"):
        p = os.path.join(ddir, f"{name}.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            add(S, f"{name}.csv 存在", "FAIL", f"缺失或为空: {p}")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            add(S, f"{name}.csv 可读", "FAIL", repr(e)[:120])
            continue
        need = {"pid_a", "pid_b", "label"}
        if not need <= set(df.columns):
            add(S, f"{name}.csv 列名", "FAIL", f"缺 {need - set(df.columns)}，现有 {list(df.columns)}")
            continue
        dfs[name] = df
        add(S, f"{name}.csv 行数", "INFO", f"{len(df)} 行, 列={list(df.columns)}")
        bad_na = int(df[["pid_a", "pid_b", "label"]].isna().sum().sum())
        add(S, f"{name} NaN", "PASS" if bad_na == 0 else "FAIL", f"NaN={bad_na}")
        lab = set(np.unique(df["label"].to_numpy()).tolist())
        ok_lab = lab <= {0, 1, 0.0, 1.0}
        add(S, f"{name} 标签域", "PASS" if ok_lab else "FAIL", f"取值={sorted(lab)}")
        loops = int((df.pid_a == df.pid_b).sum())
        add(S, f"{name} 自环", "PASS" if loops == 0 else "FAIL", f"pid_a==pid_b 共 {loops} 行")
        dup = len(df) - len({frozenset((a, b)) for a, b in zip(df.pid_a, df.pid_b)})
        add(S, f"{name} 重复无序对", "WARN" if dup else "PASS", f"{dup} 行重复")
        pos = float(df["label"].mean())
        add(S, f"{name} 阳性率", "INFO" if 0.05 <= pos <= 0.95 else "WARN",
            f"pos={int(df.label.sum())} neg={int((df.label == 0).sum())} ratio={pos:.4f}")
        if "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            add(S, f"{name} 时间范围", "INFO", f"{d.min()} → {d.max()}（B3 截断参考）")
    if len(dfs) == 3:
        ps = {k: (set(dfs[k].pid_a) | set(dfs[k].pid_b)) for k in dfs}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = ps[a] & ps[b]
            add("data", f"P7 蛋白不相交 {a}&{b}", "PASS" if not ov else "FAIL",
                f"交集 {len(ov)} 个" + (f", e.g. {sorted(ov)[:3]}" if ov else ""))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            pa = {frozenset((x, y)) for x, y in zip(dfs[a].pid_a, dfs[a].pid_b)}
            pb = {frozenset((x, y)) for x, y in zip(dfs[b].pid_a, dfs[b].pid_b)}
            ov = len(pa & pb)
            add("data", f"配对重叠 {a}&{b}", "PASS" if ov == 0 else "FAIL", f"{ov} 对")
    return dfs


def check_features(root, sample, full_nan):
    S = "features"
    fdir = os.path.join(root, "workdir", "features")
    if not os.path.isdir(fdir):
        add(S, "features 目录", "FAIL", f"缺失: {fdir}")
        return None
    mp = os.path.join(fdir, "meta.json")
    if not os.path.exists(mp):
        add(S, "meta.json", "FAIL", "缺失")
        return None
    meta = json.load(open(mp))
    n = int(meta["n"]); pid2idx = meta["pid2idx"]; ht = meta["has_text"]; hg = meta["has_genome"]
    add(S, "meta.n", "PASS" if (len(pid2idx) == n and len(ht) == n and len(hg) == n) else "WARN",
        f"n={n}, pid2idx={len(pid2idx)}, has_text={sum(1 for x in ht if x)}/{n}, has_genome={sum(1 for x in hg if x)}/{n}"
        + ("（pid2idx 与 n 不一致，见下面的死键/越界判定）" if len(pid2idx) != n else ""))
    bad_range = {k: int(v) for k, v in pid2idx.items() if not (0 <= int(v) < n)}
    dup_all = len(pid2idx) - len({int(v) for v in pid2idx.values()})
    if not bad_range and dup_all == 0:
        add(S, "pid2idx 值域/唯一", "PASS", f"全部索引在 [0,{n}) 且无重复")
    total_gb = sum(os.path.getsize(os.path.join(fdir, f)) for f in os.listdir(fdir)
                   if os.path.isfile(os.path.join(fdir, f))) / 2**30
    add(S, "磁盘占用", "INFO", f"{total_gb:.2f} GiB")

    # 全部划分的 pid 必须都在库里，且其索引必须落在 [0,n) 内且唯一
    from collections import Counter, defaultdict
    miss = 0; tot = 0; used_pids = set()
    for name in ("train", "val", "test"):
        p = os.path.join(root, "workdir", "data", f"{name}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        tot += len(df)
        miss += sum(1 for a, b in zip(df.pid_a, df.pid_b) if a not in pid2idx or b not in pid2idx)
        used_pids.update(df.pid_a.tolist()); used_pids.update(df.pid_b.tolist())
    add(S, "划分 pid 全覆盖", "PASS" if miss == 0 else "FAIL", f"{tot} 行中缺特征 {miss} 行")
    fatal_oob = set(bad_range) & used_pids
    if bad_range:
        add(S, "越界键判定", "FAIL" if fatal_oob else "WARN",
            f"{len(bad_range)} 个越界键" + (f"，被划分引用(致命): {sorted(fatal_oob)[:5]}"
            if fatal_oob else f"，全部未被引用(死键，无害): {sorted(set(bad_range))[:5]}"))
    if dup_all:
        rev = defaultdict(list)
        for k, v in pid2idx.items():
            rev[int(v)].append(k)
        dup_idx_all = [i for i, ps in rev.items() if len(ps) > 1]
        cnt = Counter(int(pid2idx[k]) for k in used_pids if k in pid2idx)
        dup_used = sorted(i for i, c in cnt.items() if c > 1)
        # 别名背书：uniprot_primary_map.txt（次级→主登录号），构建脚本据此生成别名
        alias_pairs = {}
        mf = os.path.join(root, "workdir", "features", "uniprot_primary_map.txt")
        if os.path.exists(mf):
            for line in open(mf, encoding="utf-8"):
                t = line.rstrip("\n").split("\t")
                if len(t) >= 2 and t[0] and t[1] and t[0] != t[1]:
                    alias_pairs[t[0]] = t[1]

        def canon(x, depth=8):
            for _ in range(depth):
                y = alias_pairs.get(x)
                if not y or y == x:
                    return x
                x = y
            return x

        if dup_used:
            ok, bad_list = [], []
            for i in dup_used:
                ps = sorted(set(rev[i]) & used_pids) or sorted(rev[i])
                groups = {}
                for q in ps:
                    groups.setdefault(canon(q), []).append(q)
                if len(groups) == 1 and len(ps) > 1:
                    ok.append((i, ps))
                else:
                    bad_list.append((i, ps))
            if not bad_list:
                add(S, "被引用索引重复", "PASS",
                    f"全库重复索引 {len(dup_idx_all)} 处，均为 uniprot_primary_map.txt 背书的构建期别名"
                    "（同 UniProt 条目主/次登录号，同序列，功能无损）: "
                    + "; ".join(f"idx {i} ← {ps}" for i, ps in ok[:5]))
            else:
                add(S, "被引用索引重复", "FAIL",
                    f"未获 map 文件背书的冲突 {len(bad_list)} 处: "
                    + "; ".join(f"idx {i} ← {ps}" for i, ps in bad_list[:5])
                    + "（两 pid 序列不同→特征错位，须人工比对或重建映射；map 文件缺失也会导致本判定）")
        else:
            add(S, "被引用索引重复", "WARN",
                f"全库重复索引 {len(dup_idx_all)} 处，均未被划分引用: "
                + "; ".join(f"idx {i} ← {sorted(rev[i])}" for i in dup_idx_all[:5]))
    add(S, "被引用索引值域", "PASS" if all(0 <= int(pid2idx[k]) < n for k in used_pids if k in pid2idx) else "FAIL",
        f"{len(used_pids)} 个被引用蛋白的索引全部在 [0,{n})")
    unused_keys = sorted(set(pid2idx) - used_pids)
    if unused_keys:
        add(S, "未被引用的键", "WARN", f"{len(unused_keys)} 个: {unused_keys[:5]}")

    idx = np.linspace(0, n - 1, num=min(sample or n, n)).astype(int) if n else np.array([], int)
    L = np.asarray(load_mmap(os.path.join(fdir, "lengths.npy")))
    add(S, "lengths", "PASS" if (L.shape == (n,) and L.min() > 0 and L.max() <= MAX_LEN) else "FAIL",
        f"min={int(L.min())} med={int(np.median(L))} max={int(L.max())} 触顶={int((L == MAX_LEN).sum())}")
    P = load_mmap(os.path.join(fdir, "plddt.npy"))
    if P.shape != (n, MAX_LEN):
        add(S, "plddt 形状", "FAIL", f"{P.shape} != ({n},{MAX_LEN})")
    else:
        Ps = np.asarray(P[idx], np.float32)
        add(S, "plddt 值域", "PASS" if (Ps.min() >= 0 and Ps.max() <= 100) else "FAIL",
            f"采样 min={Ps.min():.1f} max={Ps.max():.1f} mean={Ps.mean():.1f}")

    for ch, d in list(RES_CH.items()) + [("iface", None)]:
        p = os.path.join(fdir, ch + ".npy")
        shape = (n, MAX_LEN, d) if d else (n, MAX_LEN)
        arr = load_mmap(p)
        if arr is None:
            add(S, f"{ch}.npy", "FAIL", "缺失或损坏")
            continue
        add(S, f"{ch}.npy 形状", "PASS" if tuple(arr.shape) == shape else "FAIL",
            f"{tuple(arr.shape)}（期望 {shape}）")
        if str(arr.dtype) != EXP_DTYPE[ch]:
            add(S, f"{ch}.npy dtype", "WARN", f"{arr.dtype}（建议 {EXP_DTYPE[ch]}，功能等价可接受）")
        nan = inf = bad_pad = 0; nz = 0.0
        for s0 in range(0, len(idx), 64):        # 分块：内存峰值 ~64×4.9MB
            sel = idx[s0:s0 + 64]
            sub = np.asarray(arr[sel], np.float32)
            nan += int(np.isnan(sub).sum()); inf += int(np.isinf(sub).sum())
            nz = max(nz, float(np.abs(sub).max()))
            for k, i in enumerate(sel):
                Li = int(L[i]); tail = sub[k, Li:]
                if tail.size and float(np.abs(tail).max()) > 0:
                    bad_pad += 1
        add(S, f"{ch} 采样 NaN/Inf", "PASS" if (nan == 0 and inf == 0) else "FAIL", f"NaN={nan} Inf={inf}")
        add(S, f"{ch} 非平凡性", "PASS" if nz > 1e-6 else "FAIL", f"|max|={nz:.4f}")
        add(S, f"{ch} padding=0（采样）", "PASS" if bad_pad == 0 else "FAIL",
            f"{bad_pad}/{len(idx)} 行越界非零")
        if full_nan:
            step = 512
            nbad = 0
            for s in range(0, n, step):
                blk = np.asarray(arr[s:s + step], np.float32)
                nbad += int(np.isnan(blk).sum()) + int(np.isinf(blk).sum())
            add(S, f"{ch} 全库 NaN/Inf", "PASS" if nbad == 0 else "FAIL", f"{nbad} 处")

    for ch, d in PROT_CH.items():
        p = os.path.join(fdir, ch + ".npy")
        arr = load_mmap(p)
        if arr is None:
            add(S, f"{ch}.npy", "FAIL", "缺失或损坏")
            continue
        add(S, f"{ch}.npy 形状", "PASS" if tuple(arr.shape) == (n, d) else "FAIL",
            f"{tuple(arr.shape)}（期望 ({n},{d})）")
        if str(arr.dtype) != EXP_DTYPE[ch]:
            add(S, f"{ch}.npy dtype", "WARN", f"{arr.dtype}（建议 {EXP_DTYPE[ch]}）")
        has = np.array(ht if ch == "text" else hg, bool)
        nan = inf = mism = 0
        for s0 in range(0, len(idx), 64):
            sel = idx[s0:s0 + 64]
            sub = np.asarray(arr[sel], np.float32)
            nan += int(np.isnan(sub).sum()); inf += int(np.isinf(sub).sum())
            for k, i in enumerate(sel):
                nz = float(np.abs(sub[k]).max())
                if has[i] != (nz > 1e-6):
                    mism += 1
        add(S, f"{ch} 与 has 标记一致（采样）", "PASS" if mism == 0 else "FAIL",
            f"{mism}/{len(idx)} 行不符（has=1 应非零 / has=0 应全零）")
        if full_nan:
            nbad = 0
            for s in range(0, n, 512):
                blk = np.asarray(arr[s:s + 512], np.float32)
                nbad += int(np.isnan(blk).sum()) + int(np.isinf(blk).sum())
            add(S, f"{ch} 全库 NaN/Inf", "PASS" if nbad == 0 else "FAIL", f"{nbad} 处")
    return dict(n=n, pid2idx=pid2idx)


def check_adapter(root):
    S = "ssl"
    p = os.path.join(root, "workdir", "ssl", "adapter.pt")
    if not os.path.exists(p):
        add(S, "adapter.pt", "FAIL", f"缺失: {p}（R2+ 全部依赖）")
        return
    try:
        import torch
    except Exception as e:
        add(S, "adapter.pt", "WARN", f"torch 不可用，跳过深度检查（{repr(e)[:60]}）")
        return
    try:
        sd = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(p, map_location="cpu")
    num = sum(v.numel() for v in sd.values())
    add(S, "adapter 参数量", "PASS" if num == 331_648 else "FAIL",
        f"{num}（审计指纹 331,648）")
    need = {"down.weight": (128, 1280), "up.weight": (1280, 128),
            "ln.weight": (1280,), "ln.bias": (1280,)}
    bad = [k for k, sh in need.items() if k not in sd or tuple(sd[k].shape) != sh]
    add(S, "adapter 键与形状", "PASS" if not bad else "FAIL", f"异常键: {bad or '无'}")


def check_runs(root):
    S = "runs"
    rdir = os.path.join(root, "workdir", "runs")
    if not os.path.isdir(rdir) or not os.listdir(rdir):
        add(S, "runs 目录", "INFO", "为空（首轮全新训练时正常）")
        return
    try:
        import torch
    except Exception:
        add(S, "runs 深度检查", "WARN", "torch 不可用，仅查文件")
        torch = None
    for tag in sorted(os.listdir(rdir)):
        tdir = os.path.join(rdir, tag)
        if not os.path.isdir(tdir):
            continue
        labels_ref = None
        for sd_ in sorted(glob.glob(os.path.join(tdir, "seed*"))):
            tag2 = f"{tag}/{os.path.basename(sd_)}"
            for f in ("metrics.json", "predictions.npz", "cfg.json", "ckpt.pt"):
                if not os.path.exists(os.path.join(sd_, f)):
                    add(S, f"{tag2} 产物", "FAIL", f"缺 {f}")
            mp = os.path.join(sd_, "metrics.json")
            if os.path.exists(mp):
                try:
                    m = json.load(open(mp))
                    a = float(m["test"]["auprc"])
                    add(S, f"{tag2} metrics", "PASS" if 0.0 <= a <= 1.0 else "FAIL", f"test auprc={a:.4f}")
                except Exception as e:
                    add(S, f"{tag2} metrics", "FAIL", repr(e)[:100])
            pp = os.path.join(sd_, "predictions.npz")
            if os.path.exists(pp):
                z = np.load(pp, allow_pickle=True)
                y = z["labels"]
                same = (labels_ref is None or np.array_equal(labels_ref, y))
                labels_ref = y if labels_ref is None else labels_ref
                add(S, f"{tag2} labels 对齐", "PASS" if same else "FAIL",
                    f"n={len(y)}, 与首 seed 一致={same}")
            ck = os.path.join(sd_, "ckpt.pt")
            if torch is not None and os.path.exists(ck):
                try:
                    st = torch.load(ck, map_location="cpu", weights_only=True)
                except TypeError:
                    st = torch.load(ck, map_location="cpu")
                num = sum(v.numel() for v in st.values())
                exp = EXP_PARAMS.get(tag)
                if exp is None:
                    add(S, f"{tag2} 参数量", "INFO", f"{num}（非主阶梯标签，跳过指纹）")
                else:
                    add(S, f"{tag2} 参数指纹", "PASS" if num == exp else "FAIL", f"{num}（期望 {exp}）")


def check_manifest(root, mpath):
    S = "manifest"
    if not os.path.exists(mpath):
        add(S, "MANIFEST", "FAIL", f"清单不存在: {mpath}")
        return
    bad = 0
    with open(mpath, encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\r\n")
    cols = re_split(header)
    def col(key):
        for i, c in enumerate(cols):
            if key in c.lower():
                return i
        return None
    ic_hash = col("sha256") or col("sha") or col("md5")
    ic_file = col("path") or col("file") or col("name")
    if ic_hash is None or ic_file is None:
        add(S, "MANIFEST 列解析", "FAIL", f"表头 {cols} 中找不到 文件/哈希 列")
        return
    alg = "md5" if "md5" in cols[ic_hash].lower() else "sha256"
    n_ok = n_bad = n_miss = 0
    with open(mpath, encoding="utf-8", errors="replace") as f:
        next(f)
        for ln in f:
            parts = re_split(ln.rstrip("\r\n"))
            if len(parts) <= max(ic_hash, ic_file) or not parts[ic_file]:
                continue
            rel, want = parts[ic_file], parts[ic_hash].strip().lower()
            p = rel if os.path.isabs(rel) else os.path.join(root, rel)
            if not os.path.exists(p):
                n_miss += 1; add(S, f"{rel}", "FAIL", "清单内文件缺失"); continue
            h = hashlib.new(alg)
            with open(p, "rb") as fp:
                for blk in iter(lambda: fp.read(1 << 20), b""):
                    h.update(blk)
            if h.hexdigest().lower() == want:
                n_ok += 1
            else:
                n_bad += 1; add(S, f"{rel}", "FAIL", f"{alg} 不匹配")
    add(S, f"MANIFEST 对账（{alg}）", "PASS" if (n_bad == 0 and n_miss == 0) else "FAIL",
        f"ok={n_ok} bad={n_bad} missing={n_miss}")


def re_split(line):
    return [c.strip() for c in (line.split("\t") if "\t" in line else line.split(","))]


def check_env(root):
    S = "env"
    try:
        import torch
        add(S, "torch", "INFO", torch.__version__)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            add(S, "GPU", "INFO", f"{p.name}, {p.total_memory / 2**30:.1f} GiB, cc {p.major}.{p.minor}")
            add(S, "bf16 支持", "INFO" if torch.cuda.is_bf16_supported() else "WARN",
                str(torch.cuda.is_bf16_supported()))
        else:
            add(S, "CUDA", "WARN", "不可用（训练需 GPU）")
    except Exception as e:
        add(S, "torch/CUDA", "WARN", f"不可用: {repr(e)[:80]}")
    free = __import__("shutil").disk_usage(root).free / 2**30
    add(S, "磁盘剩余", "INFO" if free > 200 else "WARN", f"{free:.0f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--sample", type=int, default=512, help="每通道深度扫描的蛋白采样数，0=全部")
    ap.add_argument("--full-nan", action="store_true", help="全库 NaN/Inf 扫描（慢）")
    ap.add_argument("--runs", action="store_true", help="检查 runs 产物")
    ap.add_argument("--manifest", default=None, help="MANIFEST 文件路径（哈希对账，慢）")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        print(f"[FAIL] root 不存在: {root}"); sys.exit(1)
    t0 = time.time()
    print(f"=== 数据资产总检 root={root} sample={a.sample or '全部'} full_nan={a.full_nan} ===")
    check_env(root)
    check_splits(root)
    check_features(root, a.sample, a.full_nan)
    check_adapter(root)
    if a.runs:
        check_runs(root)
    if a.manifest:
        check_manifest(root, a.manifest if os.path.isabs(a.manifest) else os.path.join(root, a.manifest))
    n_fail = sum(1 for i in ITEMS if i["status"] == "FAIL")
    n_warn = sum(1 for i in ITEMS if i["status"] == "WARN")
    n_pass = sum(1 for i in ITEMS if i["status"] == "PASS")
    print(f"\n=== 汇总: PASS={n_pass} WARN={n_warn} FAIL={n_fail} 用时 {time.time() - t0:.0f}s ===")
    print("结论: " + ("可以发车" if n_fail == 0 else "存在 FAIL，先解决再发车"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    for ext in ("md", "json"):
        rep = os.path.join(root, f"data_check_report.{ext}")
        if ext == "md":
            with open(rep, "w", encoding="utf-8") as f:
                f.write(f"# 数据体检 {ts}\n\nPASS={n_pass} WARN={n_warn} FAIL={n_fail}\n\n")
                for i in ITEMS:
                    f.write(f"- [{i['status']}] {i['stage']}/{i['name']}: {i['detail']}\n")
        else:
            json.dump(dict(time=ts, root=root, items=ITEMS), open(rep, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
