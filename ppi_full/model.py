"""L3 SSL adapter + L4 双通路融合（硬开关）+ L5 对称头 + R0 锚点（v2.3）。
本轮修复：N3(_global_asym 四段拼接对齐 4d)。携带已审计补丁：P0-2/P0-3/P1-6/D1/C2。
参数账（init 打印核对）：L3=331,648 / L4=3,294,721 / L5=529,153 / 总=4,155,522 / R0=1,045,569。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelCfg


def _pmask(p, key, ndim=2):
    """蛋白级可用性掩码 → [B,1]（ndim=3 时 [B,1,1]）；旧 batch/旧 ckpt 无该键时返回 1.0。"""
    m = p.get(key)
    if m is None:
        return 1.0
    return m.reshape(m.shape[0], *([1] * (ndim - 1)))


class BiasedAttention(nn.Module):
    def __init__(self, d, n_heads, w_drop=0.0):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dh = d // n_heads
        self.w_drop = w_drop
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, q_in, k_in, v_in, pad_q=None, pad_k=None, bias=None, need_w=False):
        B, Lq, D = q_in.shape
        Lk = k_in.shape[1]
        qh = self.q(q_in).view(B, Lq, self.h, self.dh).transpose(1, 2)   # [B,h,Lq,dh]
        kh = self.k(k_in).view(B, Lk, self.h, self.dh).transpose(1, 2)
        vh = self.v(v_in).view(B, Lk, self.h, self.dh).transpose(1, 2)
        att = (qh @ kh.transpose(-1, -2)) / math.sqrt(self.dh)           # [B,h,Lq,Lk]
        if bias is not None:                                             # [B,1,Lk] → [B,1,1,Lk]
            att = att + bias.unsqueeze(1)
        if pad_k is not None:
            att = att.masked_fill(pad_k[:, None, None, :], float("-inf"))
        w = att.softmax(-1)                                              # pre-dropout（供 s 提取）
        o = F.dropout(w, self.w_drop, self.training) @ vh
        o = o.transpose(1, 2).reshape(B, Lq, D)
        o = self.out(o)
        if pad_q is not None:
            o = o.masked_fill(pad_q[:, :, None], 0.0)
        return o, (w.mean(1) if need_w else None)

class AttnBlock(nn.Module):
    """Pre-norm block。self-attention: x_kv=None；cross: 显式传 kv。"""
    def __init__(self, d, n_heads, ffn_mult, p_drop):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = BiasedAttention(d, n_heads, p_drop)
        self.ff = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(),
                                nn.Dropout(p_drop), nn.Linear(d * ffn_mult, d))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x_q, x_kv=None, pad_q=None, pad_k=None, bias=None, need_w=False):
        kv = x_q if x_kv is None else x_kv
        h, w = self.attn(self.ln_q(x_q), self.ln_kv(kv), self.ln_kv(kv),
                         pad_q, pad_k, bias, need_w)
        x = x_q + self.drop(h)
        x = x + self.drop(self.ff(self.ln_f(x)))
        return x, w

def masked_mean(x, pad):
    m = (~pad).float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)

class PreNormBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, d), nn.Linear(d, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(self.n1(x))))
        return x + self.drop(self.fc2(self.n2(h)))

class SeqAdapter(nn.Module):
    """L3 bottleneck adapter：1280→128→1280 残差；ESM-2 主体在数据/SSL 层冻结。"""
    def __init__(self, d=1280, b=128):
        super().__init__()
        self.down, self.up = nn.Linear(d, b), nn.Linear(b, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.ln(x))))

class L4Fusion(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.seq_proj = nn.Linear(cfg.dims["seq"], d)
        self.str_proj = nn.Linear(cfg.dims["str"], d)
        self.alpha = nn.Parameter(torch.zeros(d))
        if cfg.seq_proj_pos == "late":
            self.surf_proj = nn.Linear(cfg.dims["surf"], d)
            self.surf_gate = nn.Linear(cfg.dims["surf"], d)
        else:
            self.surf_proj = nn.Linear(cfg.dims["surf"], cfg.dims["seq"])
            self.surf_gate = nn.Linear(cfg.dims["surf"], cfg.dims["seq"])
        self.cross = AttnBlock(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
        self.self_ctx = AttnBlock(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout) if cfg.selfattn_ctx else None
        self.pair_mlp = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(),
                                      nn.Dropout(cfg.dropout), nn.Linear(2 * d, d))
        self.text_proj = nn.Linear(cfg.dims["text"], d)   # 参数常驻（C2/C15：开关=输入置零）
        self.beta = nn.Parameter(torch.tensor(float(cfg.beta_init)))
        g_in = 6 * d if cfg.global_sym else 4 * d
        self.global_mlp = nn.Sequential(nn.Linear(g_in, 2 * d), nn.GELU(),
                                        nn.Dropout(cfg.dropout), nn.Linear(2 * d, d))

    def _embed(self, p):                                  # Step 4.2 表面门控 + 4.3 α 融合
        # 通道开关(sw.surf) 与 门控开关(surface_gate) 解耦：关 gate 只是把 sigmoid 权重拿掉，
        # surf 加法项保留，否则 L4-no-surface-gate 退化成“整条 surf 通道被移除”（= R4 口径）。
        surf_term = None
        if self.cfg.sw.surf:
            surf_term = torch.tanh(self.surf_proj(p["surf"])) * _pmask(p, "surf_mask", 3)
            if self.cfg.surface_gate:
                surf_term = surf_term * torch.sigmoid(self.surf_gate(p["surf"]))
        if self.cfg.seq_proj_pos == "late":
            u = self.seq_proj(p["seq"])
            if surf_term is not None:
                u = u + surf_term
        else:
            h = p["seq"] + surf_term if surf_term is not None else p["seq"]
            u = self.seq_proj(h)
        if self.cfg.zero_seq:
            u = torch.zeros_like(u)
        if self.cfg.sw.struct:
            if self.cfg.alpha_mode == "weighted":
                w = torch.sigmoid(self.alpha)
                u = w * u + (1 - w) * self.str_proj(p["str"])
            else:
                u = u + self.str_proj(p["str"])
        return u                                          # [B,L,256]

    def _interact(self, u_src, u_tgt, src, tgt):          # Step 4.4/4.5/4.6
        B = u_src.shape[0]
        iface_t, bias = None, None
        if self.cfg.sw.iface:
            iface_t = tgt["iface"] * (tgt["iface"] > self.cfg.iface_tau).float() \
                * _pmask(tgt, "iface_mask")   # τ 硬阈值 + 整蛋白缺结构掩码
            if self.cfg.iface_bias:
                bias = self.beta * iface_t[:, None, :]    # [B,1,Lk]
        ctx, w = self.cross(u_src, u_tgt, src["pad"], tgt["pad"], bias=bias, need_w=True)
        if self.cfg.sw.iface and self.cfg.siface_out and w is not None:
            s_q = (w * iface_t[:, None, None, :]).sum(-1).mean(1)   # [B,h,Lq,Lk]→[B,Lq]
            keep = (~src["pad"]).float()
            s = (s_q * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True).clamp(min=1.0)  # [B,1]
        else:
            s = u_src.new_zeros(B, 1)
        if self.self_ctx is not None:                     # Step 4.6
            ctx, _ = self.self_ctx(ctx, None, src["pad"], src["pad"])
        p_s = masked_mean(ctx, src["pad"])
        p_t = masked_mean(u_tgt, tgt["pad"])
        z = self.pair_mlp(torch.cat([p_s, p_t, p_s - p_t, p_s * p_t], -1))
        return z, s

    def _global_sym(self, A, B):                          # Step 4.8/4.9（C10 对称）
        d = self.cfg.d_model
        B0 = A["seq"].shape[0]
        if self.cfg.sw.text:
            ta = self.text_proj(A["text"]) * A["text_mask"][:, None]
            tb = self.text_proj(B["text"]) * B["text_mask"][:, None]
            ft = torch.cat([ta + tb, ta * tb, (ta - tb).abs()], -1)   # [B,3d]
        else:
            ft = A["seq"].new_zeros(B0, 3 * d)            # 硬零槽
        if self.cfg.sw.genome:
            ga = A["genome"] * A["genome_mask"][:, None]
            gb = B["genome"] * B["genome_mask"][:, None]
            fg = torch.cat([ga + gb, ga * gb, (ga - gb).abs()], -1)
        else:
            fg = A["seq"].new_zeros(B0, 3 * d)
        return self.global_mlp(torch.cat([ft, fg], -1))   # [B,6d]→[B,d]

    def _global_asym(self, P, Q):                         # N3 修复：P/Q 两侧四段拼接 = 4d
        d = self.cfg.d_model
        B0 = P["seq"].shape[0]
        tp = self.text_proj(P["text"]) * P["text_mask"][:, None] if self.cfg.sw.text \
            else P["seq"].new_zeros(B0, d)
        gp = P["genome"] * P["genome_mask"][:, None] if self.cfg.sw.genome \
            else P["seq"].new_zeros(B0, d)
        tq = self.text_proj(Q["text"]) * Q["text_mask"][:, None] if self.cfg.sw.text \
            else P["seq"].new_zeros(B0, d)
        gq = Q["genome"] * Q["genome_mask"][:, None] if self.cfg.sw.genome \
            else P["seq"].new_zeros(B0, d)
        return self.global_mlp(torch.cat([tp, gp, tq, gq], -1))   # [B,4d]→[B,d]

    def forward(self, A, B):
        uA, uB = self._embed(A), self._embed(B)
        z_ab, s_ab = self._interact(uA, uB, A, B)
        if self.cfg.bidirectional:
            z_ba, s_ba = self._interact(uB, uA, B, A)
        else:                                             # L4-unidirectional
            z_ba, s_ba = z_ab, torch.zeros_like(s_ab)
        if self.cfg.global_path and (self.cfg.sw.text or self.cfg.sw.genome):
            if self.cfg.global_sym:
                g = self._global_sym(A, B)
                z_ab = z_ab + g
                z_ba = z_ba + g
            else:
                z_ab = z_ab + self._global_asym(A, B)
                z_ba = z_ba + self._global_asym(B, A)
        return z_ab, z_ba, s_ab, s_ba                     # [B,256] / [B,1]

class L5Head(nn.Module):
    """plain 与 blocks 双模块常驻构造（N4）→ L5-plain-mlp 消融参数量守恒。"""
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.proj = nn.Linear(2 * d + 1, d)
        self.blocks = nn.ModuleList([PreNormBlock(d, cfg.dropout) for _ in range(2)])
        self.plain = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.register_buffer("T", torch.ones(1))          # 训练期恒 1；校准走 post-hoc

    def forward(self, z_ab, z_ba, s_ab, s_ba, y=None, mix_pair=None):
        if self.cfg.l5_concat_asym:
            z_pair = torch.cat([z_ab, z_ba], -1)          # [B,512]
        else:
            z_sym = 0.5 * (z_ab + z_ba)
            z_asym = (z_ab - z_ba).abs()
            if self.cfg.l5_no_asym:
                z_asym = torch.zeros_like(z_asym)
            z_pair = torch.cat([z_sym, z_asym], -1)
        s_ch = 0.5 * (s_ab + s_ba) if self.cfg.l5_siface else torch.zeros_like(s_ab)
        z_in = torch.cat([z_pair, s_ch], -1)              # [B,513]
        y_mix = None
        if mix_pair is not None and y is not None:        # D1：perm/λ 外部注入、双前向共享
            perm, lam = mix_pair
            z_in = lam * z_in + (1 - lam) * z_in[perm]
            y_mix = lam * y + (1 - lam) * y[perm]
        h = self.proj(z_in)
        if self.cfg.l5_plain_mlp:
            h = self.plain(h)
        else:
            for blk in self.blocks:
                h = blk(h)
        return self.out(self.ln(h)) / self.T, y_mix       # [B,1]

class Reim2D(nn.Module):
    """R0 复现锚点：共享编码器 + 分块外积 + 1×1 conv + masked max（P0-3）。"""
    def __init__(self, din=1280, chunk=8):
        super().__init__()
        self.chunk = chunk
        self.enc = nn.Sequential(nn.Linear(din, 640), nn.ReLU(),
                                 nn.Linear(640, 320), nn.ReLU(), nn.Linear(320, 64))
        self.conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, A, B):
        eA, eB = self.enc(A["seq"]), self.enc(B["seq"])
        logits = []
        for i in range(0, eA.shape[0], self.chunk):
            a, b = eA[i:i + self.chunk], eB[i:i + self.chunk]
            M = a[:, :, None, :] * b[:, None, :, :]       # [b,LA,LB,64]
            m = self.conv(M.permute(0, 3, 1, 2)).squeeze(1)
            keep = (~A["pad"][i:i + self.chunk])[:, :, None] & (~B["pad"][i:i + self.chunk])[:, None, :]
            m = m.masked_fill(~keep, -1e4)
            logits.append(m.amax(dim=(1, 2)))
        return torch.cat(logits, dim=0)[:, None], None         # [B,1]

class PPIModel(nn.Module):
    def __init__(self, mcfg: ModelCfg, adapter_ckpt=None, strict_adapter=True):
        super().__init__()
        self.cfg = mcfg
        if mcfg.kind == "reim":
            self.core = Reim2D(mcfg.dims["seq"])
            self.l4 = self.l5 = self.adapter = None
        else:
            self.adapter = SeqAdapter(mcfg.dims["seq"]) if mcfg.use_ssl else None
            if self.adapter is not None and adapter_ckpt:
                self.adapter.load_state_dict(torch.load(adapter_ckpt, map_location="cpu"),
                                             strict=strict_adapter)
            self.l4 = L4Fusion(mcfg)
            self.l5 = L5Head(mcfg)
            self.core = None

    def forward(self, batch, y=None, mix_pair=None):
        if self.core is not None:
            logits, _ = self.core(batch["A"], batch["B"])
            return {"logits": logits, "y_mix": None}
        bA, bB = dict(batch["A"]), dict(batch["B"])
        if self.adapter is not None:                      # P1-6：双侧同 adapter（对称性）
            bA["seq"] = self.adapter(bA["seq"])
            bB["seq"] = self.adapter(bB["seq"])
        z_ab, z_ba, s_ab, s_ba = self.l4(bA, bB)
        logits, y_mix = self.l5(z_ab, z_ba, s_ab, s_ba, y, mix_pair)
        return {"logits": logits, "y_mix": y_mix, "s_ab": s_ab, "s_ba": s_ba}

    def report_params(self):
        parts = {}
        if self.adapter is not None:
            parts["L3"] = sum(p.numel() for p in self.adapter.parameters())
        if self.l4 is not None:
            parts["L4"] = sum(p.numel() for p in self.l4.parameters())
        if self.l5 is not None:
            parts["L5"] = sum(p.numel() for p in self.l5.parameters())
        if self.core is not None:
            parts["R0"] = sum(p.numel() for p in self.core.parameters())
        parts["total"] = sum(p.numel() for p in self.parameters())
        return parts
