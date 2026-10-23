"""build_str_channels.py v3 —— 将散装 per-protein npy 打包为 store 结构侧大文件（str/surf/iface/plddt）。

用法（源机项目根目录下）:
    python build_str_channels.py --channels str
    python build_str_channels.py --channels plddt,surf --missing workdir\\afdb\\af404_pids.txt --short-zero
    python build_str_channels.py --root D:\\path\\to\\ppi_full --channels str,surf,iface

行为约定:
  - 行序 = workdir/features/features_node.tsv 的 primaries 顺序（与 seq collect 完全一致）;
  - 写前硬校验: n 一致、每个 primaries[i] 的 meta.pid2idx[pid] == i、
    每个散装文件存在且 len(散装) >= lengths[i]（collate 按 lengths 切片，短了会读到 padding）;
  - 缺件白名单（--missing）: 清单内的 pid 允许无散装文件 -> 对应 memmap 行保持全零
    （open_memmap("w+") 初始化即零，不写即零），并以通道记录到
    workdir/features/missing_pids.json 供 QA 双向一致校验;
    清单外缺件 -> 仍然 FATAL 拒绝写盘（绝不写半成品）;
  - v3 新增 --short-zero: 散装文件存在但比 lengths 短时（如 AFDB v6 模型序列因
    UniProt 版本漂移短于本地 fasta），不再 FATAL，而是与缺件同池处理——
    该行写零 + has=false，明细（src_len/expected）记入 missing_pids.json 的
    short_zero 段; 不加此开关保持 v2 行为（FATAL）;
  - 只新增 str.npy/surf.npy/iface.npy/plddt.npy 与 missing_pids.json，
    不碰 meta.json、不删散装文件;
  - 写完对 首/中/尾 3 行做逐位回读校验（零行改校验全零）。
"""
import os, sys, json, argparse
import numpy as np

RES_TAIL = {"str": (1024,), "surf": (80,), "iface": (), "plddt": ()}


def load_missing(path):
    """每行首列取 pid（兼容 pid\t... 格式与纯单列清单），跳过空行与 # 注释。"""
    pids = set()
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        pid = s.split("\t")[0].split()[0]
        if pid:
            pids.add(pid)
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--channels", default="str,surf,iface")
    ap.add_argument("--missing", default=None,
                    help="缺件白名单文件（每行首列 pid，如 workdir/afdb/af404_pids.txt）")
    ap.add_argument("--short-zero", action="store_true",
                    help="散装比 lengths 短时置零该行并记录明细（否则 FATAL，v2 行为）")
    a = ap.parse_args()

    try:
        from config import MAX_LEN
    except Exception:
        MAX_LEN = 1022
    FEAT = os.path.join(a.root, "workdir", "features")

    meta = json.load(open(os.path.join(FEAT, "meta.json")))
    n = int(meta["n"]); pid2idx = meta["pid2idx"]
    lines = open(os.path.join(FEAT, "features_node.tsv")).read().splitlines()
    primaries = [l.split("\t")[0] for l in lines[1:]]
    assert len(primaries) == n, f"features_node.tsv 行数 {len(primaries)} != meta.n {n}"

    missing_allowed = load_missing(a.missing) if a.missing else set()
    prim_set = set(primaries)
    dropped = missing_allowed - prim_set
    if dropped:
        print(f"提示: --missing 清单中 {len(dropped)} 个 pid 不在节点表内，忽略 e.g. {sorted(dropped)[:5]}")
    missing_allowed &= prim_set

    lengths = np.load(os.path.join(FEAT, "lengths.npy"), mmap_mode="r")
    chans = [c.strip() for c in a.channels.split(",") if c.strip()]
    for c in chans:
        assert c in RES_TAIL, f"未知通道 {c}（可选: {list(RES_TAIL)}）"

    # ---- 预检 1: 行序与 meta 一致 ----
    bad_idx = [(i, p, pid2idx.get(p)) for i, p in enumerate(primaries) if pid2idx.get(p) != i]
    if bad_idx:
        raise SystemExit(f"FATAL 行序失配 {len(bad_idx)} 处 e.g. {bad_idx[:3]}\n"
                         f"（primaries 顺序必须与建库时一致，禁止重排 features_node.tsv）")

    # ---- 预检 2: 散装文件齐全（白名单内允许缺）且长度够 ----
    # short 条目结构: (row_idx, pid, src_len, expected_len)
    missing, short = [], []
    missing_by_chan = {c: [] for c in chans}
    short_by_chan = {c: [] for c in chans}
    for c in chans:
        for i, p in enumerate(primaries):
            f = os.path.join(FEAT, f"{c}_{p}.npy")
            if not os.path.exists(f):
                if p in missing_allowed:
                    missing_by_chan[c].append(p)
                    continue
                missing.append(f"{c}:{p}")
                continue
            L = int(lengths[i])
            arr = np.load(f, mmap_mode="r")
            if arr.shape[0] < L:
                short_by_chan[c].append((i, p, int(arr.shape[0]), L))
            if arr.ndim - 1 != len(RES_TAIL[c]):
                raise SystemExit(f"FATAL {c}_{p}.npy ndim={arr.ndim}（期望 {1+len(RES_TAIL[c])} 维）")
    if missing:
        print(f"缺件 {len(missing)}: {missing[:10]}")
        raise SystemExit("FATAL 白名单外缺件，拒绝写盘")
    has_short = any(short_by_chan.values())
    if has_short:
        flat = [f"{c}:{p} len={s} < lengths[{i}]={L}"
                for c in chans for (i, p, s, L) in short_by_chan[c]]
        if a.short_zero:
            print(f"过短置零 {len(flat)} 条（--short-zero，与缺件同池处理）: {flat[:10]}")
        else:
            print(f"过短 {len(flat)}: {flat[:10]}")
            raise SystemExit("FATAL 散装过短，拒绝写盘（加 --short-zero 可置零同池处理）")

    # ---- 零行集合（缺件白名单 + 过短置零）----
    zero_pids = {c: set(missing_by_chan[c]) for c in chans}
    if a.short_zero:
        for c in chans:
            zero_pids[c] |= {p for (_, p, _, _) in short_by_chan[c]}
    excused = {c: len(v) for c, v in zero_pids.items() if v}
    if excused:
        print(f"零行合计（缺件{('+' if has_short and a.short_zero else '')}过短置零）: {excused}")
    if missing_allowed:
        stale = [p for p in missing_allowed if all(os.path.exists(os.path.join(FEAT, f"{c}_{p}.npy")) for c in chans)]
        if stale:
            print(f"提示: 清单中 {len(stale)} 个 pid 散装文件实际存在（清单可能过时），按存在文件打包 e.g. {stale[:5]}")

    # ---- 打包 ----
    for c in chans:
        tail = RES_TAIL[c]
        shape = (n, MAX_LEN) + tail if tail else (n, MAX_LEN)
        out = os.path.join(FEAT, f"{c}.npy")
        mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=shape)
        for i, p in enumerate(primaries):
            f = os.path.join(FEAT, f"{c}_{p}.npy")
            if not os.path.exists(f) or p in zero_pids[c]:
                if i % 1000 == 0:
                    print(f"{c}: {i}/{n} (zero-row)", flush=True)
                continue  # 白名单缺件 / 过短置零 -> 该行保持全零
            src = np.load(f, mmap_mode="r")
            L = int(lengths[i])
            blk = np.asarray(src[:L], np.float16)
            mm[i, :L] = blk
            if i % 1000 == 0:
                print(f"{c}: {i}/{n}", flush=True)
        mm.flush(); del mm
        # ---- 抽样位校验 ----
        packed = np.load(out, mmap_mode="r")
        for i in {0, n // 2, n - 1}:
            p = primaries[i]; L = int(lengths[i])
            src_path = os.path.join(FEAT, f"{c}_{p}.npy")
            if not os.path.exists(src_path) or p in zero_pids[c]:
                z = np.asarray(packed[i])
                assert z.max() == 0, f"{c} 行 {i} ({p}) 应为零行但非全零"
                continue
            src = np.asarray(np.load(src_path, mmap_mode="r")[:L], np.float16)
            row = np.asarray(packed[i, :L])
            assert np.array_equal(row, src), f"{c} 行 {i} ({p}) 回读不一致"
            if L < MAX_LEN:
                z = np.asarray(packed[i, L:])
                assert z.max() == 0, f"{c} 行 {i} padding 非零"
        print(f"{c}.npy 打包+校验完成 {shape} 零行 {len(zero_pids[c])}", flush=True)

    # ---- 缺件/过短报告（QA 双向一致校验用）----
    report = {
        "channels": {c: sorted(missing_by_chan[c]) for c in chans if missing_by_chan[c]},
        "short_zero": {c: [{"pid": p, "src_len": s, "expected": L}
                           for (_, p, s, L) in short_by_chan[c]]
                       for c in chans if short_by_chan[c]},
        "allowed_list_size": len(missing_allowed),
        "note": "零行 pid（白名单缺件 + short_zero 过短置零）对应 memmap 行全零"
                "（fp16 初始化即零，未写入）; 模型侧按 has 掩码跳过结构通道",
    }
    with open(os.path.join(FEAT, "missing_pids.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print("missing_pids.json 已写出", flush=True)

    print("BUILD_CHANNELS_DONE —— 重跑 check_data_all.py 复核", flush=True)


if __name__ == "__main__":
    main()
