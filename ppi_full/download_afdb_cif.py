"""AFDB cif 并发下载器 v4（限流退避 + 回退修复版）。

变更（相对 v3）：
  1. 修复 404 回退 bug：v3 中 `last.split(":")[-1]` 得到 "HTTP404" 而非 "404"，
     导致 v6 404 后从不回退 v5/v4。v4 修正判断。
  2. 429 限流退避：优先读 Retry-After 响应头，否则 10s/20s/30s 指数等待；
     tries 提到 3。
  3. workers 默认 16 → 6（EBI 主站对高频并发限流严格，6 并发一般可全程无 429）。

其余同 v3：v6→v5→v4 逐条目回退（仅 404/empty 时）；落盘固定 AF-{pid}-F1-model_v4.cif
兼容 geo_worker.py / extract_surf.py；>1KB 跳过可续传；失败写 _dl_failed.txt。

用法（ppi_full 根目录）:
  python download_afdb_cif.py --pids workdir\features\pids.txt
  全程 429 不断 → --workers 8；仍 429 → --workers 4
"""
import argparse, os, ssl, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL_TPL = "https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_{ver}.cif"
HDRS = {"User-Agent": "Mozilla/5.0 (research batch downloader)"}


def fetch(pid, out_dir, tries=3, timeout=90, versions=("v6", "v5", "v4")):
    dst = os.path.join(out_dir, f"AF-{pid}-F1-model_v4.cif")   # 固定 v4 名：下游硬编码兼容
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return pid, "skip"
    last = ""
    for ver in versions:
        for t in range(tries):
            try:
                req = urllib.request.Request(URL_TPL.format(pid=pid, ver=ver), headers=HDRS)
                with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
                    f.write(r.read())
                if os.path.getsize(dst) > 1000:
                    return pid, f"ok:{ver}"
                last = f"{ver}:empty"
                break
            except urllib.error.HTTPError as e:
                last = f"{ver}:HTTP{e.code}"
                if e.code == 404:
                    break                                   # 此版本不存在 → 回退更旧版本
                if e.code == 429:                           # 限流：尊重 Retry-After / 指数等待
                    ra = (e.headers or {}).get("Retry-After", "")
                    time.sleep(min(int(ra), 60) if ra.isdigit() else 10 * (t + 1))
                else:
                    time.sleep(2 * (t + 1))
            except Exception as e:
                last = f"{ver}:{type(e).__name__}"
                time.sleep(2 * (t + 1))
        tail = last.split(":")[-1]
        if "404" in tail or tail == "empty":
            continue                                        # 该版本确实没有 → 回退（v4 修复点）
        break                                               # 网络类错误 → 换版本无意义
    return pid, f"fail:{last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="workdir/features/pids.txt")
    ap.add_argument("--out", default="workdir/afdb")
    ap.add_argument("--workers", type=int, default=6, help="EBI 限流敏感；429 不断就升 8，反之降 4")
    ap.add_argument("--vers", default="v6,v5,v4", help="版本回退顺序，逗号分隔")
    ap.add_argument("--proxy", default=None, help="代理地址，如 http://127.0.0.1:7890")
    ap.add_argument("--no-proxy", action="store_true", help="强制直连，忽略系统残留代理")
    ap.add_argument("--insecure", action="store_true", help="跳过 HTTPS 证书校验")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 pid（试探用）")
    a = ap.parse_args()

    handlers = []
    if a.proxy:
        handlers.append(urllib.request.ProxyHandler({"http": a.proxy, "https": a.proxy}))
        print(f"proxy = {a.proxy}")
    elif a.no_proxy:
        handlers.append(urllib.request.ProxyHandler({}))
        print("proxy = <disabled，强制直连>")
    else:
        print(f"proxy = 系统默认 {urllib.request.getproxies() or '（无）'}")
    if a.insecure:
        handlers.append(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
        print("TLS 校验 = 关闭（--insecure）")
    if handlers:
        urllib.request.install_opener(urllib.request.build_opener(*handlers))

    versions = tuple(v.strip() for v in a.vers.split(",") if v.strip())
    os.makedirs(a.out, exist_ok=True)
    pids = [l.strip() for l in open(a.pids) if l.strip()]
    if a.limit:
        pids = pids[:a.limit]
    print(f"download {len(pids)} cif -> {a.out} (workers={a.workers}, 版本回退 {versions})")

    ok = skip = fail = 0
    fails = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, p, a.out, 3, 90, versions): p for p in pids}
        for i, fut in enumerate(as_completed(futs), 1):
            pid, st = fut.result()
            if st.startswith("ok"):
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  FIRST-FAIL {pid}: {st}")
                fails.append(f"{pid}\t{st}")
            if i % 200 == 0:
                print(f"{i}/{len(pids)} ok={ok} skip={skip} fail={fail}")
    with open(os.path.join(a.out, "_dl_failed.txt"), "w") as f:
        f.write("\n".join(fails))
    print(f"DONE ok={ok} skip={skip} fail={fail} -> _dl_failed.txt")
    not404 = [l for l in fails if "404" not in l.split("\t")[-1].split(":")[-1]]
    if fails:
        print(f"其中 {len(fails) - len(not404)} 个为全版本 404（AFDB 未收录/已移除，走排除清单流程）；"
              f"{len(not404)} 个为其他错误（可重跑本脚本补漏）。")


if __name__ == "__main__":
    main()
