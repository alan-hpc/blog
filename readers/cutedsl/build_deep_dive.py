# -*- coding: utf-8 -*-
"""Generate a self-contained deep-dive HTML for the MHC pre-norm-fn bwd CuteDSL kernel.

Output: cutedsl_review/mhc_bwd_cutedsl_deep_dive.html
House style: Min Yang's progressive Layer/figure deep-dive (golden prologue, red
layer banners, green deep-dive boxes, hand-drawn SVG figures with the fixed palette).
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "mhc_bwd_cutedsl_deep_dive.html")

# palette
RED = "#b85450"      # old / bad
GREEN = "#5fa55f"    # new / good
BLUE = "#4a6fd3"     # neutral
ORANGE = "#e0b300"   # hardware
PURPLE = "#a33ea1"   # numerics
INK = "#1f2328"
MUT = "#55606b"

DEFS = (
    '<defs>'
    '<marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
    '<path d="M0,0 L7,3 L0,6 Z" fill="#55606b"/></marker>'
    '<marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
    f'<path d="M0,0 L7,3 L0,6 Z" fill="{RED}"/></marker>'
    '<marker id="arg" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
    f'<path d="M0,0 L7,3 L0,6 Z" fill="{GREEN}"/></marker>'
    '<marker id="arb" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
    f'<path d="M0,0 L7,3 L0,6 Z" fill="{BLUE}"/></marker>'
    '</defs>'
)


def fig(fid, svg_body, cap, vh=360):
    return (
        f'<figure class="fig"><svg viewBox="0 0 1000 {vh}" xmlns="http://www.w3.org/2000/svg">'
        f'{DEFS}{svg_body}</svg>'
        f'<figcaption><b>{fid}</b>　{cap}</figcaption></figure>'
    )


def box(x, y, w, h, fill, stroke, rx=7, sw=1.6):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def txt(x, y, s, size=13, fill=INK, anchor="middle", weight="normal", mono=False, italic=False):
    fam = '"SF Mono",ui-monospace,Menlo,monospace' if mono else '-apple-system,"PingFang SC","Microsoft YaHei",sans-serif'
    st = ' font-style="italic"' if italic else ''
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family=\'{fam}\'{st}>{s}</text>')


def line(x1, y1, x2, y2, stroke="#55606b", sw=1.6, marker="ar", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" marker-end="url(#{marker})"{d}/>'


# ---------------------------------------------------------------- figures
def F1():
    b = ''
    b += txt(500, 26, 'mHC 超连接层：残差被扩成 n_hc=4 路，pre-norm-fn 预测动态混合系数', 14, INK, weight='bold')
    # 4 residual streams
    for i in range(4):
        y = 70 + i * 46
        b += box(60, y, 150, 32, '#eef3ff', BLUE)
        b += txt(135, y + 21, f'residual 流 {i}  (hidden=2048)', 11.5, '#1a2d55')
    b += txt(135, 70 + 4 * 46 + 6, 'x : (token, n_hc=4, 2048)  =  (token, 8192)', 11.5, MUT)
    # arrow to fn proj
    b += line(215, 140, 360, 140, BLUE, 2, 'arb')
    b += box(360, 96, 200, 92, '#fff8e7', ORANGE)
    b += txt(460, 120, 'pre-norm-fn', 13.5, '#7a4e00', weight='bold')
    b += txt(460, 142, 'mapping_proj', 11.5, '#7a4e00', mono=True)
    b += txt(460, 162, 'Linear(8192 -> 24)', 11, '#7a4e00', mono=True)
    b += txt(460, 180, '+ RMSNorm', 11, '#7a4e00')
    b += line(565, 140, 700, 140, ORANGE, 2, 'ar')
    # outputs -> mixing coeffs
    b += box(700, 88, 240, 104, '#f4faf4', GREEN)
    b += txt(820, 110, 'mixes : (token, 24)', 12.5, '#1a3d1a', weight='bold')
    b += txt(820, 132, '24 = n_hc^2 + 2*n_hc', 11, '#1a3d1a', mono=True)
    b += txt(820, 152, 'A (残差强度)  C (输入缩放)', 11, '#1a3d1a')
    b += txt(820, 172, '-> Sinkhorn -> Birkhoff 混合矩阵', 10.5, '#1a3d1a')
    b += txt(500, 250, '本文聚焦 pre-norm-fn 的【反向】kernel：已知 mixes 的梯度，求 x 和 fn 的梯度', 13, RED, weight='bold')
    b += txt(500, 275, '(DeepSeek-V4 设置 n_hc=4；归一化在整段 8192 上联合做，即 n_rms_group = 1)', 11.5, MUT)
    return fig('F1', b, '<b>mHC 与 pre-norm-fn 的位置</b>：4 路残差拼平成 8192 维，经一个线性投影 + RMSNorm 得到 24 维动态混合系数。本文优化的是它的反向传播 kernel。', 300)


def F2():
    b = ''
    b += txt(500, 26, 'pre-norm-fn 前向：一次矩阵乘 + 一次 RMS 归一化', 14, INK, weight='bold')
    b += box(70, 70, 150, 60, '#eef3ff', BLUE)
    b += txt(145, 95, 'x  (residual)', 12.5, '#1a2d55', weight='bold')
    b += txt(145, 114, '(token, 8192) bf16', 11, '#1a2d55', mono=True)
    b += box(70, 170, 150, 60, '#fff8e7', ORANGE)
    b += txt(145, 195, 'fn  (权重)', 12.5, '#7a4e00', weight='bold')
    b += txt(145, 214, '(24, 8192) f32', 11, '#7a4e00', mono=True)
    # matmul
    b += line(222, 100, 330, 120, BLUE, 1.8, 'arb')
    b += line(222, 200, 330, 150, ORANGE, 1.8, 'ar')
    b += box(330, 100, 170, 56, '#f4faf4', GREEN)
    b += txt(415, 124, 'out_mul = x @ fnᵀ', 12.5, '#1a3d1a', weight='bold', mono=True)
    b += txt(415, 144, '(token, 24)', 11, '#1a3d1a', mono=True)
    # sqrsum
    b += line(222, 110, 330, 230, BLUE, 1.4, 'arb', dash='4 3')
    b += box(330, 210, 170, 50, '#f9eef8', PURPLE)
    b += txt(415, 232, 'sqrsum = ∑ x²', 12.5, '#4a1a48', weight='bold', mono=True)
    b += txt(415, 250, '(token,)  -> rms', 11, '#4a1a48', mono=True)
    # combine
    b += line(502, 128, 600, 150, GREEN, 1.8, 'arg')
    b += line(502, 235, 600, 175, PURPLE, 1.8, 'ar')
    b += box(600, 138, 200, 60, '#f4faf4', GREEN)
    b += txt(700, 162, 'out = out_mul * rms', 12.5, '#1a3d1a', weight='bold', mono=True)
    b += txt(700, 182, 'rms = rsqrt(sqrsum/G + ε)', 10.5, '#1a3d1a', mono=True)
    b += line(802, 168, 900, 168, GREEN, 1.8, 'arg')
    b += box(820, 138, 130, 60, '#eef7ee', GREEN)
    b += txt(885, 165, 'mixes', 12.5, '#1a3d1a', weight='bold')
    b += txt(885, 184, '(token, 24)', 10.5, '#1a3d1a', mono=True)
    b += txt(500, 295, 'G = 8192 = rms_group_size（n_rms_group=1，整段联合归一化）', 11.5, MUT)
    return fig('F2', b, '<b>F2 前向计算图</b>：out_mul 是 x 与 fn 的矩阵乘（24 宽），sqrsum 是 x 的平方和给出 rms，二者相乘得到 mixes。反向需对这三步求导。', 320)


def F3():
    b = ''
    b += txt(500, 24, 'H100 执行层级 与 显存层级（本 kernel 用到的部分）', 14, INK, weight='bold')
    # exec hierarchy
    cols = [('Grid', '2D: (token tile, hidden tile)'), ('CTA / Block', '128 线程 = 1 warpgroup'),
            ('Warpgroup', 'wgmma m64nNk16'), ('Warp', '32 线程')]
    x = 50
    for i, (a, c) in enumerate(cols):
        b += box(x, 60, 200, 64, '#eef3ff', BLUE)
        b += txt(x + 100, 86, a, 13, '#1a2d55', weight='bold')
        b += txt(x + 100, 107, c, 10.5, '#1a2d55', mono=True)
        if i < 3:
            b += line(x + 202, 92, x + 228, 92, BLUE, 1.6, 'arb')
        x += 228
    # memory
    b += txt(500, 168, '显存层级（容量 / 带宽）', 13, INK, weight='bold')
    mem = [('寄存器 RMEM', '255/线程', '最快', '#f4faf4', GREEN),
           ('共享内存 SMEM', '228 KB/SM', '~19 TB/s', '#fff8e7', ORANGE),
           ('L2 Cache', '50 MB', '~7 TB/s', '#eef3ff', BLUE),
           ('HBM3 全局', '80 GB', '~3.3 TB/s', '#fff5f0', RED)]
    x = 90
    for name, cap, bw, fill, st in mem:
        b += box(x, 190, 195, 70, fill, st)
        b += txt(x + 97, 214, name, 12.5, INK, weight='bold')
        b += txt(x + 97, 234, cap, 11, MUT, mono=True)
        b += txt(x + 97, 252, bw, 11, MUT, mono=True)
        x += 205
    b += txt(500, 300, '本 kernel 受限于 SMEM/L1 指令吞吐，而非 HBM 带宽（见 Layer 4）', 11.5, RED, weight='bold')
    return fig('F3', b, '<b>F3 硬件层级</b>：CTA = 1 warpgroup（128 线程）跑 wgmma 张量核；数据沿 HBM→SMEM→寄存器 流动。优化的核心是把数据尽量留在片上并喂满张量核。', 325)


def _F4_unused():  # superseded by the RMSNorm∘Linear computation graph below
    b = ''
    b += txt(500, 24, '反向推导：从 out 的梯度 out_grad 反推三条路径', 14, INK, weight='bold')
    b += box(60, 60, 150, 50, '#fff5f0', RED)
    b += txt(135, 82, 'out_grad', 12.5, '#4a1515', weight='bold', mono=True)
    b += txt(135, 100, '(token, 24)', 10.5, '#4a1515', mono=True)
    # path1 out_mul_grad
    b += line(212, 78, 320, 78, RED, 1.6, 'arr')
    b += box(320, 56, 200, 48, '#f9eef8', PURPLE)
    b += txt(420, 76, 'out_mul_grad = out_grad * rms', 11, '#4a1a48', weight='bold', mono=True)
    b += txt(420, 94, '(归一化梯度, 经 tf32/bf16)', 10, '#4a1a48')
    # to fn_grad and x_grad
    b += line(522, 70, 640, 56, PURPLE, 1.6, 'ar')
    b += box(640, 36, 300, 44, '#f4faf4', GREEN)
    b += txt(790, 56, 'fn_grad = out_mul_gradᵀ @ x', 11.5, '#1a3d1a', weight='bold', mono=True)
    b += txt(790, 73, 'GEMM1 : 沿 token 维归约', 10, '#1a3d1a')
    b += line(522, 88, 640, 110, PURPLE, 1.6, 'ar')
    b += box(640, 92, 300, 44, '#f4faf4', GREEN)
    b += txt(790, 112, 'x_grad += out_mul_grad @ fn', 11.5, '#1a3d1a', weight='bold', mono=True)
    b += txt(790, 129, 'GEMM2 : 沿 24 维归约', 10, '#1a3d1a')
    # path2 rms->sqrsum_grad
    b += line(135, 112, 135, 170, RED, 1.6, 'arr')
    b += box(60, 170, 230, 48, '#f9eef8', PURPLE)
    b += txt(175, 190, 'rms_grad = ∑(out_grad * out_mul)', 10.5, '#4a1a48', weight='bold', mono=True)
    b += txt(175, 208, '(out = out_mul*rms 的另一支)', 10, '#4a1a48')
    b += line(292, 194, 380, 194, PURPLE, 1.6, 'ar')
    b += box(380, 170, 250, 48, '#f9eef8', PURPLE)
    b += txt(505, 190, 'sqrsum_grad', 11.5, '#4a1a48', weight='bold', mono=True)
    b += txt(505, 208, '= -rms_grad*rms/2(sqrsum+εG)', 9.5, '#4a1a48', mono=True)
    b += line(630, 200, 720, 230, PURPLE, 1.6, 'ar')
    b += box(640, 218, 300, 46, '#f4faf4', GREEN)
    b += txt(790, 240, 'x_grad += 2 * x * sqrsum_grad', 11.5, '#1a3d1a', weight='bold', mono=True)
    b += txt(790, 257, '(RMS 对 x 的修正项)', 10, '#1a3d1a')
    b += txt(500, 300, '融合 kernel 一次算完：out_mul_grad + 两个 GEMM + sqrsum 修正，写出 x_grad 和 fn_grad', 12, RED, weight='bold')
    return fig('F4', b, '<b>F4 反向数据流</b>：out_grad 分三条路径——经 rms 得 out_mul_grad 驱动两个 GEMM（fn_grad、x_grad），另一支经 rms_grad→sqrsum_grad 给出 x_grad 的 RMS 修正项。', 325)


def _bignode(x, y, w, h, fill, st, t1, t2=None, t3=None):
    s = box(x, y, w, h, fill, st, rx=8, sw=2)
    cy = y + h / 2
    if t3:
        s += txt(x + w / 2, cy - 9, t1, 14, INK, weight='bold')
        s += txt(x + w / 2, cy + 8, t2, 11, MUT, mono=True)
        s += txt(x + w / 2, cy + 24, t3, 10, MUT, mono=True)
    elif t2:
        s += txt(x + w / 2, cy - 3, t1, 14, INK, weight='bold')
        s += txt(x + w / 2, cy + 15, t2, 11, MUT, mono=True)
    else:
        s += txt(x + w / 2, cy + 5, t1, 14, INK, weight='bold')
    return s


def F4a():
    b = ''
    b += txt(500, 30, 'pre-norm-fn 前向：RMSNorm 之后接一个 Linear', 16, INK, weight='bold')
    y = 80
    b += _bignode(40, y, 150, 90, '#eef3ff', BLUE, 'x', '(T, nC)', 'residual, bf16')
    b += _bignode(300, y, 175, 90, '#fff8e7', ORANGE, 'RMSNorm', 'r = rsqrt(mean x²+ε)', 'x̄ = x · r')
    b += _bignode(575, y, 150, 90, '#eef3ff', BLUE, 'x̄', '(T, nC)', '归一化残差')
    b += _bignode(810, y, 150, 90, '#f4faf4', GREEN, 'mixes', '(T, 24)', 'x̄ · fnᵀ')
    # param fn
    b += _bignode(575, y + 150, 150, 64, '#fff8e7', ORANGE, 'fn (=φᵀ)', '(24, nC)')
    # arrows
    b += line(192, y + 45, 298, y + 45, BLUE, 2.4, 'arb')
    b += line(477, y + 45, 573, y + 45, ORANGE, 2.4, 'ar')
    b += line(727, y + 45, 808, y + 45, GREEN, 2.4, 'arg')
    b += txt(525, y + 38, 'Linear', 12, GREEN, weight='bold')
    b += line(650, y + 148, 650, y + 92, ORANGE, 2.0, 'ar')
    b += txt(760, y + 130, '24 = n_hc² + 2·n_hc', 11, MUT, mono=True)
    return fig('F4a', b, '<b>F4a 前向</b>：pre-norm-fn 就是 RMSNorm（把残差按逆 RMS 缩放成 x̄）再接一个线性层（权重 fn=φᵀ），产出 24 维 raw 混合系数 mixes。', 290)


def F4b():
    b = ''
    b += txt(500, 30, 'pre-norm-fn 反向：从右往左过 Linear 与 RMSNorm 两层', 16, INK, weight='bold')
    y = 80
    b += _bignode(810, y, 150, 86, '#fff5f0', RED, 'out_grad', '(T, 24)', 'g = ∂L/∂mixes')
    b += _bignode(515, y, 230, 86, '#f9eef8', PURPLE, 'Linear 反向', 'x̄_grad = g · fn', 'fn_grad = gᵀ · x̄')
    b += _bignode(250, y, 200, 86, '#f9eef8', PURPLE, 'RMSNorm 反向', 'x_grad = r(x̄_grad', '− (c/G) x̄)')
    b += _bignode(40, y, 150, 86, '#f4faf4', GREEN, 'x_grad', '(T, nC)')
    # outputs
    b += _bignode(560, y + 150, 150, 60, '#f4faf4', GREEN, 'fn_grad', '(24, nC)')
    # arrows (right to left, red)
    b += line(808, y + 43, 747, y + 43, RED, 2.4, 'arr')
    b += line(513, y + 43, 452, y + 43, PURPLE, 2.4, 'ar')
    b += txt(483, y + 36, 'x̄_grad', 10.5, PURPLE, mono=True)
    b += line(248, y + 43, 192, y + 43, GREEN, 2.4, 'arg')
    b += line(600, y + 88, 620, y + 148, PURPLE, 2.0, 'ar')
    b += txt(700, y + 125, 'c = ⟨x̄_grad, x̄⟩（每 token 标量）', 10.5, MUT, mono=True)
    return fig('F4b', b, '<b>F4b 反向</b>：复合的反向 = 先 Linear 反向（得 x̄_grad 与 fn_grad），再 RMSNorm 反向（把 x̄_grad 还原成 x_grad，含 −(c/G)x̄ 的正交修正项）。', 290)


def FCG():
    b = ''
    b += txt(500, 22, 'pre-norm-fn = RMSNorm ∘ Linear：两个标准层的复合', 14, INK, weight='bold')
    b += txt(500, 42, '上行蓝实线 = 前向；下行红虚线 = 反向误差信号（VJP）', 11, MUT)
    def node(x, y, w, h, fill, st, t1, t2):
        s = box(x, y, w, h, fill, st)
        s += txt(x + w / 2, y + h / 2 - 3, t1, 13, INK, weight='bold', mono=True)
        if t2:
            s += txt(x + w / 2, y + h / 2 + 14, t2, 9.5, MUT, mono=True)
        return s
    yF = 95   # forward row
    b += node(40, yF, 110, 54, '#eef3ff', BLUE, 'x', '(T, nC) residual')
    b += node(250, yF, 150, 54, '#fff8e7', ORANGE, 'x̄ = RMSNorm(x)', 'x̄ = x·rms')
    b += node(500, yF, 150, 54, '#f4faf4', GREEN, 'mixes = x̄·φᵀ', 'Linear（φ=fn）')
    b += node(750, yF, 110, 54, '#fff5f0', RED, 'L', 'loss')
    # param fn
    b += node(250, yF + 130, 150, 44, '#fff8e7', ORANGE, 'φ (=fn)', '(2n+n², nC)')
    # forward arrows
    b += line(152, yF + 27, 248, yF + 27, BLUE, 1.9, 'arb')
    b += line(402, yF + 27, 498, yF + 27, GREEN, 1.9, 'arg')
    b += line(652, yF + 27, 748, yF + 27, RED, 1.6, 'arr' if False else 'ar')
    b += line(325, yF + 128, 540, yF + 56, ORANGE, 1.6, 'ar')
    # backward arrows (below, red dashed)
    yB = yF + 78
    b += line(748, yB, 652, yB, RED, 1.5, 'arr', dash='5 3')
    b += txt(700, yB + 16, 'out_grad = ∂L/∂mixes', 9.5, RED, mono=True)
    b += line(498, yB, 402, yB, RED, 1.5, 'arr', dash='5 3')
    b += txt(450, yB + 16, 'Linear bwd', 9.5, RED, mono=True)
    b += txt(450, yB + 30, 'x̄_grad = g·fn', 9, RED, mono=True)
    b += line(248, yB, 152, yB, RED, 1.5, 'arr', dash='5 3')
    b += txt(200, yB + 16, 'RMSNorm bwd', 9.5, RED, mono=True)
    b += txt(200, yB + 30, 'x_grad', 9, RED, mono=True)
    # fn_grad branch
    b += line(560, yF + 56, 360, yF + 128, RED, 1.4, 'arr', dash='5 3')
    b += txt(500, yF + 104, 'fn_grad = gᵀ·x̄', 9.5, RED, mono=True, anchor='start')
    b += txt(500, 300, 'Linear 反向给出 x̄_grad 和 fn_grad；RMSNorm 反向再把 x̄_grad 变成 x_grad', 11.5, MUT)
    return fig('F4', b, '<b>F4 计算图</b>：pre-norm-fn = RMSNorm 后接一个 Linear（φ=fn）。前向蓝线、反向红虚线。Linear 反向产出 x̄_grad 与 fn_grad，RMSNorm 反向再把 x̄_grad 还原成 x_grad。', 320)


def F5():
    b = ''
    b += txt(500, 24, '两个 GEMM 的形状：一个沿 24 归约，一个沿 token 归约', 14, INK, weight='bold')
    # GEMM2 x_grad
    b += txt(250, 58, 'GEMM2 · x_grad', 13, '#1a3d1a', weight='bold')
    b += box(70, 80, 70, 90, '#f9eef8', PURPLE)
    b += txt(105, 128, 'omg', 12, '#4a1a48', weight='bold', mono=True)
    b += txt(105, 182, 'tok×24', 10, MUT, mono=True)
    b += txt(160, 130, '@', 16, INK, weight='bold')
    b += box(180, 80, 150, 50, '#fff8e7', ORANGE)
    b += txt(255, 110, 'fn  24×8192', 11, '#7a4e00', weight='bold', mono=True)
    b += txt(255, 150, '=', 16, INK, weight='bold')
    b += box(180, 140, 230, 36, '#f4faf4', GREEN)
    b += txt(295, 163, 'x_grad  tok×8192', 11, '#1a3d1a', weight='bold', mono=True)
    b += txt(250, 198, '收缩维 K = 24（小）· 输出大（token×8192）', 10.5, MUT)
    # GEMM1 fn_grad
    b += txt(740, 58, 'GEMM1 · fn_grad', 13, '#1a3d1a', weight='bold')
    b += box(560, 80, 60, 70, '#f9eef8', PURPLE)
    b += txt(590, 118, 'omgᵀ', 12, '#4a1a48', weight='bold', mono=True)
    b += txt(590, 162, '24×tok', 10, MUT, mono=True)
    b += txt(636, 120, '@', 16, INK, weight='bold')
    b += box(656, 80, 150, 70, '#eef3ff', BLUE)
    b += txt(731, 118, 'x  tok×8192', 11, '#1a2d55', weight='bold', mono=True)
    b += txt(731, 165, '=', 16, INK, weight='bold')
    b += box(656, 175, 230, 34, '#f4faf4', GREEN)
    b += txt(771, 197, 'fn_grad  24×8192', 11, '#1a3d1a', weight='bold', mono=True)
    b += txt(740, 232, '收缩维 K = token=4096（大，跨所有 token 归约）', 10.5, MUT)
    b += txt(500, 272, 'omg = out_mul_grad。两个 GEMM 共享 omg，分别作正常 A 和转置 Aᵀ', 12, RED, weight='bold')
    return fig('F5', b, '<b>F5 两个 GEMM</b>：x_grad = omg @ fn（K=24，输出巨大）；fn_grad = omgᵀ @ x（K=token=4096，跨所有 token 归约）。两者都用同一个 omg，决定了后续 tiling 与归约策略。', 300)


def F6():
    b = ''
    b += txt(500, 24, '基础实现（v1, 724µs）：每 CTA 一个 hidden 列块，串行循环所有 token 块', 13.5, INK, weight='bold')
    b += box(60, 55, 120, 200, '#fff5f0', RED)
    b += txt(120, 75, 'CTA (hidden 块)', 11, '#4a1515', weight='bold')
    for i in range(4):
        b += box(75, 90 + i * 38, 90, 30, '#fff', RED, rx=4, sw=1)
        b += txt(120, 110 + i * 38, f'token 块 {i}', 10, '#4a1515', mono=True)
    b += txt(120, 248, '串行 for 循环', 10, RED, italic=True)
    b += line(185, 150, 250, 150, RED, 1.6, 'arr')
    b += box(250, 90, 200, 120, '#fff8e7', ORANGE)
    b += txt(350, 112, '每个 token 块:', 11.5, '#7a4e00', weight='bold')
    b += txt(350, 134, '• 标量 bf16 读 x', 10.5, '#7a4e00')
    b += txt(350, 152, '• 标量 smem FMA 算两个 GEMM', 10.5, '#7a4e00')
    b += txt(350, 170, '• fn_grad 寄存器累加', 10.5, '#7a4e00')
    b += txt(350, 188, '• 标量 bf16 写 x_grad', 10.5, '#7a4e00')
    b += line(455, 150, 520, 150, '#55606b', 1.6, 'ar')
    b += box(520, 95, 420, 110, '#fff5f0', RED)
    b += txt(730, 120, '为什么慢 = 189 GB/s', 12.5, '#4a1515', weight='bold')
    b += txt(730, 144, '• 标量 16-bit 访存：无向量化，访存效率 1/8', 11, '#4a1515')
    b += txt(730, 164, '• 标量 smem GEMM：bank 冲突 + 无寄存器复用', 11, '#4a1515')
    b += txt(730, 184, '• 仅 128 个 CTA，串行 64 次 token 迭代', 11, '#4a1515')
    b += txt(500, 240, '正确，但离带宽下限（~45µs）差 16×。后面逐项优化。', 12, MUT)
    return fig('F6', b, '<b>F6 基础版</b>：最直接的映射——CTA 拥有一个 hidden 列块、串行循环所有 token 块，全程标量访存与标量 smem GEMM。正确但仅 189 GB/s，是后续优化的起点。', 270)


def F7():
    b = ''
    b += txt(500, 22, '优化①：128-bit 向量化 IO（724→457µs）', 14, INK, weight='bold')
    # before
    b += txt(250, 52, '❌ 之前：标量 bf16', 12.5, RED, weight='bold')
    for i in range(8):
        b += box(90 + i * 40, 70, 34, 34, '#fff5f0', RED, rx=3, sw=1)
        b += txt(107 + i * 40, 92, 't', 9, '#4a1515', mono=True)
    b += txt(250, 124, '每线程 1 个 16-bit load → 访存事务多 8×', 10.5, MUT)
    # after
    b += txt(750, 52, '✅ 之后：128-bit 向量', 12.5, GREEN, weight='bold')
    for i in range(2):
        b += box(590 + i * 160, 70, 154, 34, '#f4faf4', GREEN, rx=3, sw=1.4)
        b += txt(667 + i * 160, 92, '8×bf16 = 128 bit', 10, '#1a3d1a', mono=True)
    b += txt(750, 124, '每线程 1 个 128-bit load → 合并访存', 10.5, MUT)
    b += txt(500, 165, '用 TV-layout tiled-copy 把 x / x_grad 以 128-bit 向量在 HBM↔寄存器↔swizzled smem 间搬运', 12, INK)
    b += box(250, 190, 500, 60, '#fff4e0', ORANGE)
    b += txt(500, 213, '结果：189 → 299 GB/s', 12.5, '#5a3f00', weight='bold')
    b += txt(500, 234, '但仍远离带宽下限 → 说明瓶颈不只是访存，还有计算/staging', 11, '#5a3f00')
    return fig('F7', b, '<b>F7 向量化 IO</b>：把标量 16-bit 访存换成 128-bit 向量化 tiled-copy（每线程搬 8 个 bf16）。带宽 189→299 GB/s，但仍未触顶，提示计算侧才是大头。', 270)


def F8():
    b = ''
    b += txt(500, 22, '优化②：bf16 wgmma 张量核（457→447µs）—— 计算变快，但引入 swizzle staging', 13, INK, weight='bold')
    b += txt(135, 56, '标量 FMA GEMM', 11.5, RED, weight='bold')
    b += box(60, 70, 150, 70, '#fff5f0', RED)
    b += txt(135, 100, '线程逐元素', 11, '#4a1515')
    b += txt(135, 120, '乘加（慢）', 11, '#4a1515')
    b += line(215, 105, 290, 105, '#55606b', 1.6, 'ar')
    b += box(290, 60, 200, 90, '#f4faf4', GREEN)
    b += txt(390, 84, 'wgmma m64nNk16', 12, '#1a3d1a', weight='bold', mono=True)
    b += txt(390, 106, 'bf16 张量核', 11, '#1a3d1a')
    b += txt(390, 126, '矩阵乘几乎免费', 10.5, '#1a3d1a')
    b += line(495, 105, 560, 105, GREEN, 1.6, 'arg')
    b += box(560, 55, 380, 100, '#fff4e0', ORANGE)
    b += txt(750, 78, '代价：wgmma 算子要求 swizzled smem', 11.5, '#5a3f00', weight='bold')
    b += txt(750, 100, '每个 token 块都要：寄存器片 → cute.copy → swizzled smem', 10.5, '#5a3f00')
    b += txt(750, 120, '+ 多次 __syncthreads + x 转置 + omg hi/lo 双拷', 10.5, '#5a3f00')
    b += txt(750, 140, '→ staging 开销吃掉了张量核收益', 10.5, '#5a3f00')
    # numeric note
    b += box(180, 180, 640, 70, '#f9eef8', PURPLE)
    b += txt(500, 202, '数值技巧：fn_grad 跨 4096 token 归约，单 bf16 精度不够', 11.5, '#4a1a48', weight='bold')
    b += txt(500, 224, 'omg 拆成 bf16 高位 + bf16 低位（2×bf16 ≈ tf32），喂两遍 wgmma 累加', 11, '#4a1a48')
    b += txt(500, 242, '（x_grad 收缩维只有 24，单 bf16 即可）', 10, '#4a1a48')
    return fig('F8', b, '<b>F8 wgmma 张量核</b>：两个 GEMM 改用 bf16 wgmma，矩阵乘几乎免费——但 wgmma 强制 swizzled-smem 算子，带来 staging 开销；fn_grad 用 hi/lo 2×bf16 维持精度。', 270)


def F9():
    b = ''
    b += txt(500, 22, '优化③：2D-grid 架构重写（447→282µs，首次超过 TileLang）', 13.5, INK, weight='bold')
    # before serial
    b += txt(250, 52, '❌ 之前：128 CTA + 串行 token 循环', 11.5, RED, weight='bold')
    b += box(120, 66, 120, 150, '#fff5f0', RED)
    b += txt(180, 84, 'CTA = hidden 块', 10, '#4a1515', weight='bold')
    for i in range(4):
        b += box(132, 96 + i * 26, 96, 20, '#fff', RED, rx=3, sw=1)
        b += txt(180, 110 + i * 26, f'token 块 {i} (串行)', 9, '#4a1515', mono=True)
    b += txt(180, 208, '每 CTA 重读/重算 per-token', 9, RED, italic=True)
    # after 2D
    b += txt(750, 52, '✅ 之后：token×hidden 二维 grid', 11.5, GREEN, weight='bold')
    for r in range(3):
        for c in range(4):
            b += box(560 + c * 50, 66 + r * 34, 44, 28, '#f4faf4', GREEN, rx=3, sw=1)
    b += txt(640, 178, '每 CTA 只做 1 个 tile，无串行', 9.5, '#1a3d1a')
    b += txt(640, 196, '8192 个 CTA → 满占用率', 9.5, '#1a3d1a')
    # key points
    b += box(120, 235, 760, 95, '#eef7ee', GREEN)
    b += txt(500, 257, '三个关键改动', 12.5, '#1a3d1a', weight='bold')
    b += txt(500, 278, '① Kernel A 一次性预计算 out_mul_grad / sqrsum_grad 到 workspace（消除每块重算）', 11, '#1a3d1a')
    b += txt(500, 297, '② Kernel B 二维 grid，无串行 token 循环 → 占用率拉满', 11, '#1a3d1a')
    b += txt(500, 316, '③ fn_grad 跨 token 块用 f32 atomicAdd 归约（不额外占 HBM workspace）', 11, '#1a3d1a')
    return fig('F9', b, '<b>F9 2D-grid 重写</b>：把"串行 token 循环"换成 token×hidden 二维 grid（每 CTA 一个 tile），per-token 量预计算一次，fn_grad 用 atomicAdd 跨块归约。282µs，首次超过 TileLang。', 350)


def F10():
    b = ''
    b += txt(500, 22, '优化④：ncu 引导砍 shared memory（282→168µs）', 14, INK, weight='bold')
    b += txt(500, 46, 'ncu 发现占用率被 smem 压死：205 KB/block → 每 SM 只能放 1 个 block', 11.5, RED)
    # occupancy bars
    b += txt(250, 78, '占用率', 12, INK, weight='bold')
    b += box(120, 92, 260, 30, '#f6f8fa', '#d0d7de', rx=4, sw=1)
    b += box(120, 92, int(260 * 0.0625), 30, RED, RED, rx=4, sw=1)
    b += txt(250, 112, '之前 6.25%（smem 205KB → 1 block/SM）', 10, '#fff' if False else INK)
    b += box(120, 132, 260, 30, '#f6f8fa', '#d0d7de', rx=4, sw=1)
    b += box(120, 132, int(260 * 0.1875), 30, GREEN, GREEN, rx=4, sw=1)
    b += txt(250, 152, '之后 18.75%（smem 62KB → 3 block/SM）', 10, INK)
    # how
    b += box(420, 80, 470, 95, '#eef7ee', GREEN)
    b += txt(655, 102, '怎么砍的', 12.5, '#1a3d1a', weight='bold')
    b += txt(655, 123, '• hidden 块 256 → 128（所有 buffer 减半）', 11, '#1a3d1a')
    b += txt(655, 143, '• 去掉重复的 plain staging buffer', 11, '#1a3d1a')
    b += txt(655, 163, '• 直接由寄存器片填 swizzled smem', 11, '#1a3d1a')
    b += box(200, 195, 600, 56, '#fff4e0', ORANGE)
    b += txt(500, 217, '结果：282 → 168µs / 818 GB/s', 13, '#5a3f00', weight='bold')
    b += txt(500, 238, '寄存器 255→162，smem 205→62 KB，占用率 6.25%→18.75%', 11, '#5a3f00')
    return fig('F10', b, '<b>F10 砍 smem 抬占用率</b>：ncu 显示 smem（205KB/block）把占用率压到 6.25%。减小 hidden 块、去掉重复 staging 后 smem→62KB、占用率→18.75%，168µs。', 270)


def F11():
    b = ''
    b += txt(500, 24, '优化瀑布：从基础版到最终版（H100, num_tokens=4096, hidden=8192）', 13.5, INK, weight='bold')
    data = [('TileLang\n基线', 326, BLUE), ('v1 标量', 724, RED), ('+向量化IO', 457, ORANGE),
            ('+wgmma', 447, ORANGE), ('2D-grid', 282, GREEN), ('+砍smem\n(最终)', 168, GREEN)]
    maxv = 760
    base_y = 300
    bw = 110
    gap = 38
    x = 70
    for name, us, col in data:
        h = int((us / maxv) * 230)
        y = base_y - h
        b += box(x, y, bw, h, '#fff', col, rx=4, sw=2)
        b += box(x, y, bw, min(h, 8), col, col, rx=4, sw=0)
        b += txt(x + bw / 2, y - 8, f'{us}µs', 12.5, col, weight='bold')
        for li, ln in enumerate(name.split('\n')):
            b += txt(x + bw / 2, base_y + 18 + li * 15, ln, 10.5, INK)
        x += bw + gap
    b += line(60, base_y, 940, base_y, '#999', 1.2, dash='2 2')
    # floor line
    fy = base_y - int((45 / maxv) * 230)
    b += line(60, fy, 940, fy, GREEN, 1.4, dash='6 4')
    b += txt(880, fy - 6, '带宽下限 ~45µs', 10, GREEN, anchor='end')
    b += txt(500, 348, '最终 168µs：1.93× TileLang 融合；2.4× 真实运行的拆分路径（见 Layer 5）', 12, RED, weight='bold')
    return fig('F11', b, '<b>F11 优化瀑布</b>：724µs(标量) → 457(向量化) → 447(wgmma) → 282(2D-grid) → 168(砍smem)。最终比 TileLang 融合(326µs)快 1.93×。虚线为 HBM 带宽下限。', 380)


def F12():
    b = ''
    b += txt(500, 24, 'ncu 瓶颈分解：为什么停在 168µs', 14, INK, weight='bold')
    bars = [('L1/TEX 指令吞吐', 87, RED), ('SMEM pipe', 83, RED), ('DRAM 带宽', 21, GREEN), ('Compute (SM)', 21, GREEN)]
    y = 60
    for name, pct, col in bars:
        b += txt(180, y + 19, name, 11.5, INK, anchor='end')
        b += box(200, y, 600, 26, '#f6f8fa', '#d0d7de', rx=4, sw=1)
        b += box(200, y, int(600 * pct / 100), 26, col, col, rx=4, sw=0)
        b += txt(810, y + 19, f'{pct}%', 11.5, col, weight='bold', anchor='start')
        y += 40
    b += box(150, 232, 700, 96, '#fff5f0', RED)
    b += txt(500, 254, '诊断：L1/TEX 指令发射受限，不是 HBM、不是算力', 12, '#4a1515', weight='bold')
    b += txt(500, 275, '指令来自喂给 24 宽细长 GEMM 的算子 staging：x 转置 + omg hi/lo 填充', 11, '#4a1515')
    b += txt(500, 295, '占用率硬卡 3 block/SM（寄存器 162 与 smem 62KB 双重限制）', 11, '#4a1515')
    b += txt(500, 315, '三轮再优化（token 分组 / 向量化填充 / 消除 x 转置）均无改进 → 架构地板', 11, '#4a1515')
    return fig('F12', b, '<b>F12 瓶颈分解</b>：L1/TEX 与 SMEM pipe 高达 83-87%，而 DRAM/算力仅 21%——是片上指令受限。算子 staging 是根因，且经三轮验证不可再降，构成架构地板。', 350)


def F13():
    b = ''
    b += txt(500, 24, '真实运行（nsys）：当前生产走的是【拆分】路径，可被融合 cute 替换', 13, INK, weight='bold')
    b += txt(250, 56, '❌ 真实运行：拆分 kernel', 12, RED, weight='bold')
    b += box(110, 72, 280, 40, '#fff5f0', RED)
    b += txt(250, 96, 'bwd_mul  398µs + bwd_norm  4µs', 11, '#4a1515', weight='bold', mono=True)
    b += txt(250, 130, '≈ 402µs（两次 launch + 中间张量落显存）', 10.5, MUT)
    b += line(250, 150, 250, 185, '#55606b', 1.6, 'ar')
    b += txt(750, 56, '✅ 融合 CuteDSL', 12, GREEN, weight='bold')
    b += box(610, 72, 280, 40, '#f4faf4', GREEN)
    b += txt(750, 96, 'norm+mul 融合一次  168µs', 11, '#1a3d1a', weight='bold', mono=True)
    b += txt(750, 130, '吃同样的 saved tensors，drop-in 替换', 10.5, MUT)
    b += line(750, 150, 750, 185, GREEN, 1.6, 'arg')
    b += box(250, 190, 500, 80, '#fff4e0', ORANGE)
    b += txt(500, 214, '收益', 13, '#5a3f00', weight='bold')
    b += txt(500, 236, '168µs  vs  实跑 402µs  =  2.4×', 12.5, '#5a3f00', weight='bold')
    b += txt(500, 256, '通过 env 宏 TILE_KERNELS_MHC_BWD_BACKEND=cute 启用', 11, '#5a3f00')
    b += txt(500, 300, 'nsys 确认形状一致：num_tokens=4096, hidden=8192, n_rms_group=1', 11.5, MUT)
    return fig('F13', b, '<b>F13 真实场景</b>：nsys 显示生产当前走拆分路径（bwd_mul 398µs + bwd_norm 4µs ≈ 402µs）。融合 cute kernel 吃同样张量、drop-in 替换 → 2.4× 加速。', 320)


# ---------------------------------------------------------------- page
CSS = r"""
:root{--fg:#1f2328;--muted:#55606b;--bg:#fff;--bg-alt:#f6f8fa;--border:#d0d7de;--link:#004276;--accent:#b85450;--ok:#5fa55f;--warn:#e0b300;--prol-bg:#fff8e7;--prol-border:#e0b300;--prol-ink:#4a3500;--dd-bg:#eef7ee;--dd-border:#5fa55f;--std-bg:#fff5f0;--std-border:#b85450;--sm-bg:#f4faf4;--sm-border:#5fa55f}
*{box-sizing:border-box}
html{font-size:15px;line-height:1.65;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Roboto,Arial,sans-serif;color:var(--fg);background:var(--bg)}
body{margin:0;padding:0}
.container{max-width:1080px;margin:0 auto;padding:28px 32px 90px}
.masthead{border-bottom:3px solid var(--accent);padding-bottom:14px;margin-bottom:8px}
.masthead h1{font-size:30px;margin:0 0 6px;line-height:1.25}
.masthead .sub{color:var(--muted);font-size:15px}
.masthead .meta{color:var(--muted);font-size:12.5px;margin-top:8px}
h2{font-size:24px;border-bottom:2px solid var(--border);padding-bottom:6px;margin:1.8em 0 .6em}
h3{font-size:19px;margin:1.4em 0 .4em}
p{margin:.5em 0 .8em}
a{color:var(--link)}
code{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:86%;background:var(--bg-alt);padding:.12em .36em;border-radius:3px}
table{border-collapse:collapse;margin:.8em 0;font-size:14px;width:100%}
th,td{border:1px solid var(--border);padding:6px 12px;vertical-align:top;text-align:left}
th{background:var(--bg-alt);font-weight:600}
tr:nth-child(even) td{background:#fafbfc}
.prologue{background:var(--prol-bg);border:1px solid var(--prol-border);border-left:5px solid var(--prol-border);border-radius:4px;padding:22px 28px;margin:22px 0 30px;color:var(--prol-ink)}
.prologue h2.prologue-title{font-size:22px;margin:0 0 10px;color:#7a4e00;border-bottom:2px solid var(--prol-border);padding-bottom:6px}
.prologue h3.prologue-h3{color:#7a4e00;margin:20px 0 8px;font-size:16px;border-bottom:1px dashed var(--prol-border);padding-bottom:3px}
.prologue code{background:#fff1c4;color:#5a3f00}
.prologue table{font-size:13px}
.prologue th,.prologue td{border:1px solid #d9b860}
.prologue th{background:#fff1c4;color:#5a3f00}
.prologue td{background:#fffcf1}
.prologue-toc{background:#fffcf1;border:1px solid #d9b860;border-radius:4px;padding:12px 20px;margin:12px 0 18px;font-size:13.5px;line-height:1.85}
.prologue-toc a{color:#7a4e00;font-weight:600;text-decoration:none}
.deep-dive{background:var(--dd-bg);border-left:4px solid var(--dd-border);margin:18px 0;padding:14px 18px;border-radius:4px;font-size:14.5px;color:#1a3d1a;line-height:1.75}
.deep-dive .dd-label{display:inline-block;background:var(--dd-border);color:#fff;font-size:11.5px;font-weight:700;padding:2px 10px;border-radius:3px;letter-spacing:.5px;margin-bottom:8px}
.deep-dive strong{display:block;font-size:1.04em;color:#0f3d0f;margin:.2em 0 .4em}
.deep-dive code{background:#d7e8d7;color:#0f3d0f}
.formula-box{margin:10px 0 14px;padding:12px 18px;border-radius:4px;font-size:14px;line-height:1.8}
.formula-box.std-box{background:var(--std-bg);border:1px solid var(--std-border);border-left:4px solid var(--std-border);color:#4a1515}
.formula-box.sm-box{background:var(--sm-bg);border:1px solid var(--sm-border);border-left:4px solid var(--sm-border);color:#1a3d1a}
.formula-label{display:inline-block;font-weight:700;font-size:12px;padding:2px 10px;border-radius:3px;margin-bottom:8px}
.std-box .formula-label{background:var(--std-border);color:#fff}
.sm-box .formula-label{background:var(--sm-border);color:#fff}
.formula-box code{background:rgba(0,0,0,.08);color:inherit}
.tip{background:#eef7ff;border-left:4px solid #4a90e2;padding:10px 16px;margin:14px 0;color:#1a3a5c;border-radius:4px;font-size:14px}
.warn{background:#fff4e0;border-left:4px solid var(--warn);padding:10px 16px;margin:14px 0;color:#5a3f00;border-radius:4px;font-size:14px}
figure.fig{margin:18px 0 26px}
figure.fig svg{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--border);border-radius:6px}
figure.fig figcaption{color:var(--muted);font-size:12.8px;padding:8px 4px 0;line-height:1.55}
figure.fig figcaption b{color:#333}
.layer-banner{margin:38px 0 14px;padding:14px 18px;border-left:5px solid var(--accent);background:linear-gradient(90deg,#fff5f5 0,#fff 80%);border-radius:0 6px 6px 0}
.layer-banner .tag{display:inline-block;background:var(--accent);color:#fff;font-size:11.5px;font-weight:700;padding:2px 9px;border-radius:3px;letter-spacing:.08em}
.layer-banner h2.t{font-size:22px;font-weight:700;margin:4px 0 0;padding:0;border:none;line-height:1.3}
.layer-banner .s{color:var(--muted);font-size:14px;margin-top:2px}
.opt-pill{display:inline-block;background:#eef7ee;border:1px solid #5fa55f;color:#1a3d1a;font-size:12px;padding:2px 9px;border-radius:999px;margin:2px 4px 2px 0;font-weight:600}
.opt-pill.mem{background:#eef3ff;border-color:#4a6fd3;color:#1a2d55}
.opt-pill.num{background:#f9eef8;border-color:#a33ea1;color:#4a1a48}
.opt-pill.sched{background:#fff4e0;border-color:#e0b300;color:#5a3f00}
"""


def banner(tag, title, sub='', color=None):
    style = '' if color is None else f' style="border-left-color:{color};background:linear-gradient(90deg,#f6f8fa 0,#fff 80%)"'
    tagstyle = '' if color is None else f' style="background:{color}"'
    return (f'<div class="layer-banner"{style}><div class="tag"{tagstyle}>{tag}</div>'
            f'<h2 class="t">{title}</h2><div class="s">{sub}</div></div>')


def main():
    H = []
    H.append('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
    H.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    H.append('<title>mHC pre-norm-fn 反向 kernel · CuteDSL 优化全解析</title>')
    H.append(r'<script>window.MathJax={tex:{inlineMath:[["$","$"],["\\(","\\)"]],displayMath:[["$$","$$"],["\\[","\\]"]]},svg:{fontCache:"global"}};</script>')
    # Inline MathJax (tex-svg, self-contained fonts) so the page renders math OFFLINE.
    _mj = os.path.join(os.path.dirname(__file__), 'mathjax-tex-svg-full.js')
    with open(_mj, encoding='utf-8') as _f:
        _mjjs = _f.read().replace('</script>', '<\\/script>')
    H.append('<script>' + _mjjs + '</script>')
    H.append(f'<style>{CSS}</style></head><body><main class="container">')

    # masthead
    H.append('<div class="masthead"><h1>DeepSeek-V4 mHC pre-norm-fn 反向 kernel<br>CuteDSL 优化全解析</h1>')
    H.append('<div class="sub">从 mHC 前向推导反向，再从朴素实现逐步优化到超越 TileLang 的 Hopper bf16-wgmma 融合 kernel</div>')
    H.append('<div class="meta">目标硬件：NVIDIA H100 (sm90) · 基准形状：num_tokens=4096, n_hc=4, hidden=2048 (mhc_hidden=8192) · 最终 168µs / 818 GB/s（1.93× TileLang）</div></div>')

    # prologue
    H.append('<section class="prologue"><h2 class="prologue-title">\U0001F4D6 Prologue · 背景知识与符号定义</h2>')
    H.append('<p>本节先交代 mHC 是什么、pre-norm-fn 前向在算什么、以及 H100 上的硬件层级，为后面的反向推导与逐层优化打底。</p>')
    H.append('<div class="prologue-toc"><b style="color:#7a4e00">\U0001F4D1 目录</b><ol>'
             '<li><a href="#pr1">① mHC 与 pre-norm-fn 的位置</a></li>'
             '<li><a href="#pr2">② pre-norm-fn 前向公式</a></li>'
             '<li><a href="#pr3">③ H100 执行 / 显存层级</a></li>'
             '<li><a href="#pr4">④ 符号速查表</a></li></ol>'
             '<b style="color:#7a4e00">章节</b>：Layer 1 反向推导 · Layer 2 基础实现 · Layer 3 优化之路 · Layer 4 收敛分析 · Layer 5 真实收益 · Appendix</div>')

    H.append('<h3 class="prologue-h3" id="pr1">① mHC 与 pre-norm-fn 的位置</h3>')
    H.append('<p>mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）是 DeepSeek-V4 的核心架构创新之一。它把标准残差扩成 <code>n_hc=4</code> 路通道，并用一个双随机（Birkhoff）混合矩阵在层间传递信息。这个混合矩阵是<strong>动态</strong>的——由 <code>pre-norm-fn</code> 从当前残差预测出来。本文优化的就是 pre-norm-fn 的<strong>反向</strong> kernel。</p>')
    H.append(F1())

    H.append('<h3 class="prologue-h3" id="pr2">② pre-norm-fn 前向公式</h3>')
    H.append(r'''<p>一句话：<strong>pre-norm-fn = RMSNorm 后接一个 Linear</strong>。把 4 路残差拼平成 8192 维 <span>$x$</span>，先归一化 <span>$\bar x=\mathrm{RMSNorm}(x)$</span>，再过线性层 <span>$fn$</span> 产出 24 维 raw 混合系数（<code>24 = n_hc² + 2·n_hc</code>）。整段 8192 作为<strong>单个</strong>归一化组（<code>n_rms_group=1</code>，已由真实模型代码与 nsys 双重确认）。Layer 1 给出完整前后向推导。</p>''')
    H.append(F2())
    H.append(r'''<div class="formula-box sm-box"><div class="formula-label">前向（n_rms_group=1，G=8192）</div>
$$r[t]=\Big(\tfrac1G\textstyle\sum_j x[t,j]^2+\varepsilon\Big)^{-1/2}\,,\quad \bar x=x\cdot r\,,\quad \boxed{\;\mathrm{mixes}=\bar x\,fn^{\top}\;}$$
<div style="font-size:12.5px;margin-top:6px">实现上 kernel 保存未归一化的 <code>out_mul = x·fnᵀ</code> 与 <code>sqrsum = ∑x²</code>，则 <code>mixes = out_mul·rms</code>（代数等价，前向少存一份 x̄）。</div></div>''')

    H.append('<h3 class="prologue-h3" id="pr3">③ H100 执行 / 显存层级</h3>')
    H.append(F3())

    H.append('<h3 class="prologue-h3" id="pr4">④ 符号速查表</h3>')
    H.append('<table><tr><th>符号</th><th>含义</th><th>典型值</th></tr>'
             '<tr><td><code>n_hc / mhc_mult</code></td><td>超连接残差流数</td><td>4</td></tr>'
             '<tr><td><code>mhc_mult3</code></td><td>输出宽度 = n_hc²+2n_hc</td><td>24</td></tr>'
             '<tr><td><code>mhc_hidden_size</code></td><td>拼平后的隐藏维 = n_hc×hidden</td><td>8192</td></tr>'
             '<tr><td><code>n_rms_group</code></td><td>RMS 归一化分组数</td><td>1（联合归一化）</td></tr>'
             '<tr><td><code>num_tokens</code></td><td>token 数（s×b）</td><td>4096</td></tr>'
             '<tr><td><code>out_mul_grad</code> (omg)</td><td>out_mul 的梯度 = out_grad·rms</td><td>(token,24)</td></tr>'
             '<tr><td><code>token_block / hidden_block</code></td><td>kernel tile（wgmma M 固定 64）</td><td>64 / 128</td></tr></table>')
    H.append('</section>')

    # Layer 1
    H.append(banner('Layer 1', '从 mHC 整体到 pre-norm-fn：把它看成 RMSNorm ∘ Linear', '一旦看清是两个标准层的复合，前后向就是 Linear backward + RMSNorm backward'))

    H.append('<h3>1.1 mHC 整体与 pre-norm-fn 的位置</h3>')
    H.append(r'''<p>沿用 DeepSeek-V4 论文记号：每个 token 的残差是 <span>$n$</span> 路流 <span>$x_l\in\mathbb{R}^{n\times C}$</span>（<span>$n=4$</span>，<span>$C$</span> 为单流隐藏维，<span>$nC=8192$</span>）。单层 mHC 把它们混合后再走子模块 <span>$F$</span>：</p>''')
    H.append(r'''<div class="formula-box sm-box"><div class="formula-label">mHC 单层（论文 Eq.3）</div>
$$x_{l+1}=H^{\mathrm{res}}_l\,x_l+\big(H^{\mathrm{post}}_l\big)^{\!\top} F\big(H^{\mathrm{pre}}_l\,x_l,\;W_l\big)$$</div>''')
    H.append(r'''<p>三个混合系数 <span>$H^{\mathrm{pre}},H^{\mathrm{post}}\!\in\!\mathbb{R}^{1\times n}$</span>、<span>$H^{\mathrm{res}}\!\in\!\mathbb{R}^{n\times n}$</span> 是<strong>动态</strong>的：先把残差归一化 <span>$\bar x=\mathrm{RMSNorm}(x_l)$</span>，再过一个轻量线性层 <span>$\varphi\in\mathbb{R}^{(nC)\times(2n+n^2)}$</span> 得到 <span>$2n+n^2=24$</span> 个 raw logits，最后各自加 <span>$\alpha$</span> 缩放/偏置并经 sigmoid / 2·sigmoid / Sinkhorn 激活（后者投影到 Birkhoff 双随机流形）。</p>''')
    H.append(r'''<div class="tip">本文优化的 <code>pre-norm-fn</code> 正是产出这 24 个 raw logits 的那一步——也就是 <strong>RMSNorm 之后接一个 Linear</strong>（权重 <span>$\varphi$</span>，在 kernel 里记作 <span>$fn=\varphi^{\top}\!\in\!\mathbb{R}^{24\times 8192}$</span>）。下游的 <span>$\alpha$</span>/激活/Sinkhorn 是另外的 kernel（<code>head_compute_mix</code>、<code>sinkhorn</code> 等），不在本文范围。F1 给出它在数据流中的位置。</div>''')

    H.append(r'''<p>先看<strong>整层</strong> mHC 的反向全景（下图，与论文前向结构图同款画法）：上游梯度 <span>$\delta=\partial L/\partial x_{l+1}$</span> 沿 <strong>pre / post / res</strong> 三条路径回传，最终汇总到 <span>$\partial L/\partial x_l$</span>。红框圈出的 <strong>RMSNorm + Linear</strong> 就是本文 CuteDSL kernel 负责的 pre-norm-fn 部分；其余（激活 / Sinkhorn / Layer F / 流混合）是相邻 kernel。</p>''')
    with open(os.path.join(os.path.dirname(__file__), 'mhc_backward_fullflow.svg'), encoding='utf-8') as _bf:
        _bwd_svg = _bf.read()
    H.append('<figure class="fig">' + _bwd_svg +
             '<figcaption><b>F-mHC 反向全图</b>　整层 mHC 的反向：δ 经求和节点拷贝到 post / res 两支，post 路反传得 v_grad→u_grad→pre 路，三条路径（res、pre、预测）梯度按 path-sum 汇回 x_l。红框 = pre-norm-fn（本文 kernel）。</figcaption></figure>')
    H.append('<h3>1.2 前向：RMSNorm ∘ Linear</h3>')
    H.append(r'''<p>把每个 token 的 <span>$n$</span> 路残差拼平成一个 <span>$nC=8192$</span> 维向量 <span>$x$</span>（<code>n_rms_group=1</code>，整段联合归一化，已由真实模型代码与 nsys 确认）。前向就是两步标准操作：</p>''')
    H.append(r'''<div class="formula-box sm-box"><div class="formula-label">pre-norm-fn · forward pass</div>
$$\textbf{RMSNorm：}\quad r[t]=\Big(\tfrac1G\textstyle\sum_j x[t,j]^2+\varepsilon\Big)^{-1/2}\,,\qquad \bar x[t,j]=x[t,j]\cdot r[t]$$
$$\textbf{Linear：}\quad \boxed{\;\mathrm{mixes}[t,m]=\sum_j \bar x[t,j]\,fn[m,j]=\big(\bar x\,fn^{\top}\big)[t,m]\;}\quad(M=24)$$</div>''')
    H.append(F4a())
    H.append(r'''<div class="deep-dive"><span class="dd-label">实现对照</span><strong>kernel 里的变量怎么对应</strong>
<p>为复用，kernel 保存的是<strong>未归一化</strong>的投影 <span>$\mathrm{out\_mul}=x\,fn^{\top}$</span> 和 <span>$\mathrm{sqrsum}=\sum_j x^2$</span>；于是 <span>$\mathrm{rms}=r$</span>、<span>$\mathrm{mixes}=\mathrm{out\_mul}\cdot\mathrm{rms}$</span>。换言之 <span>$\bar x\,fn^{\top}=r\,(x\,fn^{\top})$</span>——把 RMS 缩放从输入挪到输出，纯属代数等价，方便前向少存一份 <span>$\bar x$</span>。</p></div>''')

    H.append('<h3>1.3 反向 = Linear backward + RMSNorm backward</h3>')
    H.append(r'''<p>记上游梯度 <span>$g=\partial L/\partial\,\mathrm{mixes}\in\mathbb{R}^{T\times M}$</span>（即 <code>out_grad</code>）。既然 pre-norm-fn 是两个标准层的复合，反向就<strong>从右往左</strong>过这两层（F4b）。</p>''')
    H.append(F4b())
    H.append(r'''<p><b>(a) Linear 反向。</b> <span>$\mathrm{mixes}=\bar x\,fn^{\top}$</span> 是普通矩阵乘，标准结果（对参数求和在 token 维、对输入求和在 <span>$M$</span> 维）：</p>
$$\boxed{\;\mathrm{fn\_grad}=g^{\top}\bar x\;}\;(\textbf{GEMM1，沿 token 归约})\,,\qquad \boxed{\;\bar x\_\mathrm{grad}=g\,fn\;}\;(\textbf{GEMM2，沿 }M\textbf{ 归约})\,.$$''')
    H.append(r'''<p><b>(b) RMSNorm 反向。</b> 给定 <span>$\bar x\_\mathrm{grad}$</span>，对 <span>$\bar x[t,j]=x[t,j]\,r[t]$</span>（其中 <span>$r$</span> 又依赖整行 <span>$x$</span>）求导。设每 token 标量 <span>$c[t]=\langle \bar x\_\mathrm{grad}[t],\,\bar x[t]\rangle=\sum_j \bar x\_\mathrm{grad}[t,j]\,\bar x[t,j]$</span>，标准 RMSNorm 反向给出：</p>
$$\boxed{\;x\_\mathrm{grad}[t,j]=r[t]\Big(\bar x\_\mathrm{grad}[t,j]-\tfrac{c[t]}{G}\,\bar x[t,j]\Big)\;(+\ \text{初值})\;}$$
<p>第一项 <span>$r\,\bar x\_\mathrm{grad}$</span> 是"直通"梯度，第二项 <span>$-\tfrac{r\,c}{G}\bar x$</span> 是 RMS 把整行耦合起来产生的修正——这正是 <span>$\bar x$</span> 必须减去自身投影分量的原因（梯度要与归一化方向正交）。</p>''')
    H.append(r'''<div class="deep-dive"><span class="dd-label">DEEP DIVE</span><strong>为什么第二项长这样（一步到位的推导）</strong>
<p><span>$\dfrac{\partial \bar x[t,k]}{\partial x[t,j]}=r\,\delta_{kj}+x[t,k]\dfrac{\partial r}{\partial x[t,j]}$</span>，而 <span>$\dfrac{\partial r}{\partial x[t,j]}=-\dfrac{r^3}{G}x[t,j]$</span>（由 <span>$r=a^{-1/2},\,a=\tfrac1G\sum x^2+\varepsilon$</span>）。代入 <span>$x\_\mathrm{grad}[t,j]=\sum_k \bar x\_\mathrm{grad}[t,k]\,\partial\bar x[t,k]/\partial x[t,j]$</span> 并用 <span>$x=\bar x/r$</span> 整理，即得上式。关键：<span>$\partial r/\partial x$</span> 里<strong>只有一个 <span>$r^3$</span></strong>（等价于最终式里的单 <span>$r$</span> 因子）。</p></div>''')
    H.append(r'''<p>把以上结果翻译成 kernel 实际使用的变量（<span>$\mathrm{omg}=g\odot r$</span>、<span>$\mathrm{rms\_grad}=\sum_m g\odot\mathrm{out\_mul}=\langle x,\bar x\_\mathrm{grad}\rangle$</span>）：</p>''')
    H.append(r'''<div class="formula-box sm-box"><div class="formula-label">pre-norm-fn · backward pass（正确形式 · 以 kernel 变量表示）</div>
$$\mathrm{omg}=\mathrm{out\_grad}\odot\mathrm{rms}\,,\qquad \mathrm{fn\_grad}=\mathrm{omg}^{\top}x\,,\qquad \mathrm{rms\_grad}=\textstyle\sum_m \mathrm{out\_grad}\odot\mathrm{out\_mul}$$
$$\mathrm{sqrsum\_grad}=-\dfrac{\mathrm{rms\_grad}\odot\mathrm{rms}}{2(\mathrm{sqrsum}+\varepsilon G)}\,,\qquad \mathrm{x\_grad}=\underbrace{\mathrm{omg}\,fn}_{=\,r\,\bar x\_\mathrm{grad}}+\underbrace{2\,x\odot\mathrm{sqrsum\_grad}}_{=\,-\,r\,c\,\bar x/G}\;(+\ \text{初值})$$</div>''')
    H.append(r'''<p>两个花括号正好对应 RMSNorm 反向的两项，<span>$\mathrm{fn\_grad}=\mathrm{omg}^{\top}x=g^{\top}\bar x$</span> 与 Linear 反向一致。<strong>注意</strong>：上式 <span>$\mathrm{sqrsum\_grad}$</span> 含<strong>单个</strong> <span>$\mathrm{rms}$</span>——这是标准 RMSNorm 反向的正确系数；而仓库现有代码用了 <span>$\mathrm{rms}^2$</span>，详见 1.5。</p>''')

    H.append('<h3>1.4 两个 GEMM 的形状</h3>')
    H.append(r'''<p>Step ②③ 是反向的两块重活，它们<strong>共享同一个 <span>$\mathrm{omg}$</span></strong>，分别作正常算子和转置算子：</p>''')
    H.append(F5())
    H.append('<div class="tip">关键观察：<code>x_grad</code> 的收缩维只有 24（很小，输出巨大），<code>fn_grad</code> 的收缩维是 token=4096（要跨所有 token 归约）。这个不对称决定了后面的 tiling 与归约策略——尤其是 fn_grad 为什么要用 atomicAdd。</div>')

    H.append('<h3>1.5 数值核验：一个被掩盖的 rms 因子</h3>')
    H.append(r'''<p>推导给出 <span>$\mathrm{sqrsum\_grad}$</span> 含<strong>单个</strong> <span>$\mathrm{rms}$</span>。但仓库现有的 torch 参考实现与两个 kernel（TileLang 与本文 cute）都用了 <span>$\mathrm{rms}^2$</span>。用 PyTorch autograd 在 <span>$\mathrm{rms}\neq1$</span> 的数据上对照（把 <span>$x$</span> 放大使 <span>$\mathrm{rms}\in[0.23,0.46]$</span>）：</p>''')
    H.append('<table><tr><th>梯度项</th><th>对 autograd 的最大误差</th><th>结论</th></tr>'
             '<tr><td>fn_grad、x_grad 的两个 GEMM 部分</td><td>0.0</td><td>完全正确</td></tr>'
             '<tr><td>x_grad 修正项（<b>rms²</b>，仓库现用）</td><td>0.15</td><td style="color:#b85450">偏差</td></tr>'
             '<tr><td>x_grad 修正项（<b>rms¹</b>，本文推导）</td><td>1e-16</td><td style="color:#5fa55f">与 autograd 一致</td></tr></table>')
    H.append(r'''<div class="side-by-side">
<div class="formula-box std-box"><div class="formula-label">❌ 仓库现用（ref / TileLang / cute）</div>
$$\mathrm{sqrsum\_grad}=-\frac{\mathrm{rms\_grad}\cdot\mathrm{rms}^{\,2}}{2(\mathrm{sqrsum}+\varepsilon G)}$$</div>
<div class="formula-box sm-box"><div class="formula-label">✅ autograd 验证正确</div>
$$\mathrm{sqrsum\_grad}=-\frac{\mathrm{rms\_grad}\cdot\mathrm{rms}}{2(\mathrm{sqrsum}+\varepsilon G)}$$</div></div>''')
    H.append(r'''<div class="warn"><b>为什么测试都过、平时看不出来：</b> 两式仅相差一个因子 <span>$\mathrm{rms}$</span>，而 residual 近似单位方差时 <span>$\mathrm{sqrsum}\approx G\Rightarrow\mathrm{rms}\approx1$</span>，二者数值上几乎相等。只有当 residual 方差明显偏离 1（训练中段 residual 增长）时，这一项的 <code>x_grad</code> 修正才会出现系统性偏差（误差正比于 <span>$\mathrm{rms}$</span> 偏离 1 的程度）。</div>''')
    H.append(r'''<div class="deep-dive"><span class="dd-label">DEEP DIVE</span><strong>影响范围与处理建议</strong>
<p>受影响的<strong>只有</strong> <span>$\mathrm{sqrsum}\to x$</span> 的修正项；<span>$\mathrm{out\_mul\_grad}$</span>、<span>$\mathrm{fn\_grad}$</span> 与 <span>$x\_grad$</span> 的两个 GEMM 部分都<strong>严格正确</strong>。本文的 cute kernel 是<strong>忠实复现</strong>了既有 TileLang/ref 的行为（不是新引入的回归）。是否把系数改成正确的 <span>$\mathrm{rms}^1$</span>，建议与训练团队确认真实 residual 的 RMS 分布后再定——若长期 <span>$\mathrm{rms}\approx1$</span> 则影响可忽略，否则应统一修正 ref 与两个 kernel。</p></div>''')

    # Layer 2
    H.append(banner('Layer 2', '基础实现 —— 朴素 CuteDSL 版（724µs）'))
    H.append('<h3>2.1 最直接的映射</h3>')
    H.append('<p>第一版只追求<strong>正确</strong>：每个 CTA 拥有一个 hidden 列块，串行循环所有 token 块，把 x 和 fn 读进 smem，用线程内标量乘加完成两个 GEMM（无跨线程归约），fn_grad 在寄存器里跨 token 块累加。</p>')
    H.append(F6())
    H.append('<h3>2.2 为什么慢</h3>')
    H.append('<div class="formula-box std-box"><div class="formula-label">❌ 基础版三大成本（189 GB/s）</div>'
             '<ol><li>标量 16-bit 访存：每线程一次只搬 2 字节，访存事务比 128-bit 多 8×</li>'
             '<li>标量 smem GEMM：逐元素乘加，bank 冲突重、无寄存器复用</li>'
             '<li>仅 128 个 CTA，每个串行跑 64 次 token 迭代，延迟无法被掩盖</li></ol></div>')
    H.append('<div class="tip">基础版离 HBM 带宽下限（~45µs）差 16×。接下来逐项拆解：先解决访存，再解决计算，最后解决并行度与占用率。</div>')

    # Layer 3
    H.append(banner('Layer 3', '优化之路 —— 逐步逼近带宽（724 → 168µs）'))
    H.append('<h3>3.1 优化① 128-bit 向量化 IO（→457µs）</h3>')
    H.append('<p>x（residual）和 x_grad 是 bf16、占主导的访存量（各 67MB）。把标量访存换成 128-bit 向量化 tiled-copy，每线程一次搬 8 个 bf16。</p>')
    H.append(F7())
    H.append('<h3>3.2 优化② bf16 wgmma 张量核（→447µs）</h3>')
    H.append('<p>两个 GEMM 改用 Hopper bf16 wgmma 张量核。矩阵乘本身几乎免费——但 wgmma 要求算子放在 swizzled smem，引入了"寄存器片→cute.copy→swizzled smem"的 staging 开销，几乎抵消了张量核收益。</p>')
    H.append(F8())
    H.append('<div class="deep-dive"><span class="dd-label">DEEP DIVE</span><strong>为什么 fn_grad 要 hi/lo 双 bf16</strong>'
             '<p><code>fn_grad</code> 跨 4096 个 token 累加，单 bf16（7 位尾数）误差会被放大到 1e35 量级。把 <code>out_mul_grad</code> 拆成 bf16 高位 + bf16 低位（合起来 ≈14 位尾数 ≈ tf32），喂两遍 wgmma 累加，即可满足 7e-2 容差。x_grad 收缩维只有 24，单 bf16 足够。这正是参考实现里 <code>round_tf32</code> 的等价精度处理。</p></div>')
    H.append('<h3>3.3 优化③ 2D-grid 架构重写（→282µs，首超 TileLang）</h3>')
    H.append('<p>前两步证明瓶颈不在访存也不在算力，而在<strong>并行度与冗余工作</strong>：基础结构只有 128 个 CTA、每个串行跑 64 次迭代、且每个 hidden-CTA 都重读/重算 per-token 量。彻底重写成两段式：</p>')
    H.append(F9())
    H.append('<h3>3.4 优化④ ncu 引导砍 shared memory（→168µs）</h3>')
    H.append('<p>2D-grid 后用 ncu profiling，发现占用率只有 6.25%——被 shared memory（205 KB/block）压死，每个 SM 只能放 1 个 block。减小 hidden 块、去掉重复 staging buffer 后，smem 降到 62KB、占用率升到 18.75%。</p>')
    H.append(F10())
    H.append(F11())

    # Layer 4
    H.append(banner('Layer 4', '收敛分析 —— 为什么停在 168µs'))
    H.append('<p>继续用 ncu 定位下一个瓶颈，并做了三轮针对性再优化，全部无改进，最终确认 168µs 是这条 wgmma 路线在该问题形状下的架构地板。</p>')
    H.append(F12())
    H.append('<div class="formula-box std-box"><div class="formula-label">❌ 三轮再优化均无改进（已验证的死路）</div>'
             '<ol><li><b>token 分组</b>（每 CTA 累加 K 个 token 块再 atomicAdd）：中性 —— 证明原子操作不是瓶颈</li>'
             '<li><b>向量化算子填充</b>：被 swizzle 布局锁死，smem 存边只能 16-bit</li>'
             '<li><b>消除 x 转置</b>：用 wgmma readback 测试<strong>严格证明</strong>——swizzle 会打乱 token 坐标（k=0 对、k=5 错），转置在做必要的坐标重整，不可省</li></ol></div>')
    H.append('<div class="warn">占用率被寄存器（162/线程）和 smem（62KB）<strong>双重</strong>卡在 3 block/SM；要到 4 block 需同时把两者降下来，而 hi/lo fn_grad 精度路径又是必需的。综合下来 168µs 是该架构的合理地板。</div>')

    # Layer 5
    H.append(banner('Layer 5', '真实场景与收益'))
    H.append('<h3>5.1 nsys 实跑分析</h3>')
    H.append('<p>用真实训练的 nsys profile（<code>h100_v4_8ep_nograph</code>）核对：mHC 反向当前走的是<strong>拆分</strong>路径（<code>bwd_mul</code> 398µs + <code>bwd_norm</code> 4µs），而非融合 kernel。形状与本文一致（4096 / 8192 / n_rms_group=1）。</p>')
    H.append(F13())
    H.append('<h3>5.2 收益与接入</h3>')
    H.append('<table><tr><th>路径</th><th>反向耗时</th><th>相对</th></tr>'
             '<tr><td>真实运行：bwd_mul + bwd_norm（拆分）</td><td>≈402µs</td><td>1.0×</td></tr>'
             '<tr><td>TileLang 融合 kernel</td><td>326µs</td><td>1.23×</td></tr>'
             '<tr><td><b>CuteDSL 融合 kernel（本文）</b></td><td><b>168µs</b></td><td><b>2.4×</b></td></tr></table>')
    H.append('<div class="tip">接入是<strong>可选后端</strong>，默认行为不变。设环境变量 <code>TILE_KERNELS_MHC_BWD_BACKEND=cute</code> 即全局启用；<code>auto</code> 在 n_rms_group=1 且 CuteDSL 可用时自动启用、否则回退 TileLang。</div>')

    # Appendix
    H.append(banner('Appendix', '运行、源码与优化点清单', '', color='#555'))
    H.append('<h3>A.1 运行</h3>')
    H.append('<pre># 启用 cute 后端（环境变量宏）\nexport TILE_KERNELS_MHC_BWD_BACKEND=cute\n\n# 单测（H100 + nvidia-cutlass-dsl）\npytest tests/mhc/test_norm_fn_bwd_fused.py</pre>')
    H.append('<h3>A.2 源码路径</h3>')
    H.append('<table><tr><th>文件</th><th>内容</th></tr>'
             '<tr><td><code>tile_kernels/mhc/norm_fn_cute_kernel.py</code></td><td>CuteDSL 融合 kernel（_pre_norm_grad + _matmul_grad）</td></tr>'
             '<tr><td><code>tile_kernels/mhc/norm_fn_kernel.py</code></td><td>后端调度器 + 环境变量宏</td></tr>'
             '<tr><td><code>tile_kernels/modeling/mhc/ops/norm_fn.py</code></td><td>MHCPreNormFn.backward 接入</td></tr></table>')
    H.append('<h3>A.3 优化点清单</h3>')
    H.append('<p><span class="opt-pill">2D-grid 满占用率</span><span class="opt-pill">预计算消除重算</span>'
             '<span class="opt-pill mem">128-bit 向量化 IO</span><span class="opt-pill mem">砍 smem 抬占用率</span>'
             '<span class="opt-pill sched">bf16 wgmma 张量核</span><span class="opt-pill sched">atomicAdd 归约</span>'
             '<span class="opt-pill num">hi/lo 2×bf16 精度</span></p>')
    H.append('<h3>A.4 参考文献</h3>')
    H.append('<ul style="font-size:13.5px;line-height:1.9">'
             '<li>Laurent Boué, <i>Deep learning for pedestrians: backpropagation in CNNs</i>, arXiv:1811.11987 —— 反向推导范式（计算图 / VJP / 误差信号回传，forward&amp;backward pass 结果框）</li>'
             '<li>Laurent Boué, <i>Deep learning for pedestrians: backpropagation in Transformers</i>, arXiv:2512.23329 —— 残差 / 归一化 / 线性层的前后向推导范式</li>'
             '<li>DeepSeek-V4 技术报告 —— mHC（流形约束超连接，n_hc=4，Birkhoff/Sinkhorn）</li>'
             '<li>本仓库源码：<code>tile_kernels/torch/mhc.py</code>（前向/反向参考实现）、<code>tile_kernels/mhc/norm_fn_kernel.py</code>（TileLang）、<code>norm_fn_cute_kernel.py</code>（CuteDSL）</li></ul>')
    H.append('<p style="color:#666;font-size:12.5px;margin-top:18px">\U0001F916 本文档由源码与 ncu/nsys 实测派生 · 推导经 PyTorch autograd 数值核验 · 图示为手绘 SVG · num_tokens=4096, hidden=8192, n_rms_group=1 @ H100</p>')

    H.append('</main></body></html>')

    html = ''.join(H)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print('wrote', OUT, len(html), 'bytes')


if __name__ == '__main__':
    main()
