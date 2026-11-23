# -*- coding: utf-8 -*-
"""Generate the mHC full BACKWARD-flow figure, matching the reference forward
figure's draw.io style (purple op boxes, green Linear/F, gray dashed arrows,
red residual highway, shape annotations, yellow summary box).

Output: cutedsl_review/mhc_backward_fullflow.svg
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "mhc_backward_fullflow.svg")

PUR_F, PUR_S = '#e1d5e7', '#9673a6'   # op boxes
GRN_F, GRN_S = '#d5e8d4', '#82b366'   # Linear / Layer F
GRY_F, GRY_S = '#f5f5f5', '#999999'   # x_l / x_{l+1} / raw heads
YEL_F, YEL_S = '#fff2cc', '#d6b656'   # summary
HL_F, HL_S = '#fde7e9', '#d6342c'     # pre-norm-fn highlight (our kernel)
RED = '#d6342c'
ARR = '#555555'
INK = '#1a1a1a'
MUT = '#666666'
SANS = '-apple-system,"PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif'
MONO = '"SF Mono",Menlo,Consolas,monospace'


def box(x, y, w, h, fill, st, rx=8, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{st}" stroke-width="{sw}"{d}/>'


def t(x, y, s, size=12, fill=INK, anchor='middle', weight='normal', mono=False):
    fam = MONO if mono else SANS
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" font-family=\'{fam}\'>{s}</text>'


def arrow(x1, y1, x2, y2, color=ARR, sw=1.8, dash=None, mk='ar'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mk})"{d}/>'


def opbox(x, y, w, h, title, lines, fill=PUR_F, st=PUR_S, tcol=INK):
    s = box(x, y, w, h, fill, st)
    s += t(x + w / 2, y + 22, title, 14, tcol, weight='bold')
    yy = y + 42
    for ln, mono in lines:
        s += t(x + w / 2, yy, ln, 11, MUT, mono=mono)
        yy += 16
    return s


def main():
    W, H = 1180, 880
    P = []
    P.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family=\'{SANS}\'>')
    P.append('<defs>'
             f'<marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{ARR}"/></marker>'
             f'<marker id="arr" markerWidth="11" markerHeight="11" refX="8" refY="3.2" orient="auto"><path d="M0,0 L8,3.2 L0,6.4 Z" fill="{RED}"/></marker>'
             '</defs>')
    P.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    P.append(t(W / 2, 34, 'mHC 完整结构 · 反向全流程：梯度沿 pre / post / res 三条路径回传到 x_l', 19, INK, weight='bold'))
    P.append(t(W / 2, 56, 'δ ≡ ∂L/∂x；下游梯度 × 本地偏导（VJP），x_l 汇总三条路径之和（path-sum）', 12.5, MUT))

    # ---- column x positions mirror the forward figure ----
    # left column (paths / F / sum / x), center (residual + H heads), right (Linear/RMSNorm/raw heads)
    # top row y~80
    P.append(opbox(40, 80, 230, 78, '∂L/∂x_l  (累加)', [('= res + pre + 预测 三路之和', False), ('T × n × C', True)], fill=GRY_F, st=GRY_S))
    P.append(opbox(560, 80, 210, 78, 'RMSNorm 反向', [('x_l_grad += r(x̄_grad', False), ('  − (c/G) x̄)', False)], fill=HL_F, st=HL_S))
    P.append(opbox(900, 80, 240, 90, 'Linear φ 反向', [('x̄_grad = z_grad · Wφᵀ', True), ('Wφ_grad = x̄ᵀ · z_grad', True), ('（pre-norm-fn 反向）', False)], fill=HL_F, st=HL_S))
    # raw heads grad (right, mid)
    P.append(opbox(900, 300, 240, 96, 'Raw heads 反向 (concat)', [('z_grad : T×(2n+n²)', True), ('= [H̃pre_g | H̃post_g | H̃res_g]', True), ('激活反向: ·σ′ / ·2σ′ / Sinkhorn', False)], fill=PUR_F, st=PUR_S))
    # H heads grad (center-right column)
    P.append(opbox(660, 188, 170, 70, 'H^pre 反向', [('H̃pre_g = Hpre_g·σ′', True), ('shape: T×n', True)]))
    P.append(opbox(660, 312, 170, 78, 'H^res 反向', [('Hres_g via Sinkhorn', False), ('-bwd (recompute)', False), ('shape: T×n×n', True)]))
    P.append(opbox(660, 444, 170, 70, 'H^post 反向', [('H̃post_g = Hpost_g·2σ′', True), ('shape: T×n', True)]))
    # paths (left/center)
    P.append(opbox(40, 188, 250, 96, 'Pre path 反向', [('u_grad ← Layer F 反向', False), ('x_l_grad += Hpre[t,s]·u_grad[t,c]', True), ('Hpre_g[t,s] = Σ_c u_grad·x_l', True)]))
    P.append(opbox(40, 322, 250, 70, 'Layer F 反向', [('u_grad = J_Fᵀ · v_grad', True), ('in: v_grad → out: u_grad', False)], fill=GRN_F, st=GRN_S))
    P.append(opbox(355, 312, 250, 96, 'Residual path 反向', [('x_l_grad += Σ_r Hres[t,r,s]·δ[t,r,c]', True), ('Hres_g[t,r,s] = Σ_c δ[t,r,c]·x_l[t,s,c]', True), ('per-token n×n 转置矩阵乘', False)]))
    P.append(opbox(40, 450, 250, 96, 'Post path 反向', [('v_grad = Σ_s Hpost[t,s]·δ[t,s,c]', True), ('Hpost_g[t,s] = Σ_c δ[t,s,c]·v[t,c]', True), ('broadcast 的反向 = 沿流求和', False)]))
    # sum + x_{l+1}
    P.append(f'<circle cx="165" cy="610" r="26" fill="#fff" stroke="{ARR}" stroke-width="2"/>' + t(165, 617, '+', 24, INK, weight='bold'))
    P.append(opbox(95, 670, 150, 70, '∂L/∂x_{l+1}', [('= δ_{l+1}', True), ('T × n × C', True)], fill=GRY_F, st=GRY_S))

    # ---------- arrows (reverse of forward) ----------
    a = arrow
    # δ_{l+1} -> + (up)
    P.append(a(165, 668, 165, 638, ARR, 2.2))
    # + splits to post path and residual path (gradient copied)
    P.append(a(165, 584, 165, 548))            # + -> post path
    P.append(t(230, 568, 'δ 拷贝到两支', 10.5, MUT, anchor='start'))
    P.append(a(190, 600, 430, 410, ARR, 1.8))  # + -> residual path
    # red highway: δ_{l+1} -> x_l_grad directly (residual identity term, via Hres)
    P.append(f'<path d="M 245 700 H 700 C 760 700 760 200 760 150 L 270 119" fill="none" stroke="{RED}" stroke-width="2.6" stroke-dasharray="2 0" marker-end="url(#arr)"/>')
    P.append(t(720, 690, '残差高速路：δ 经 Hresᵀ 直达 x_l', 11, RED, anchor='start', weight='bold'))
    # post path -> Layer F (v_grad)
    P.append(a(110, 448, 110, 394, ARR, 1.8))
    P.append(t(150, 420, 'v_grad', 10.5, MUT, anchor='start', mono=True))
    # Layer F -> pre path (u_grad)
    P.append(a(110, 320, 110, 286, ARR, 1.8))
    P.append(t(150, 305, 'u_grad', 10.5, MUT, anchor='start', mono=True))
    # H grads from paths to H boxes
    P.append(a(292, 220, 658, 220, ARR, 1.6))     # pre path -> H^pre
    P.append(t(470, 212, 'Hpre_g', 10, MUT, mono=True))
    P.append(a(607, 350, 658, 350, ARR, 1.6))     # residual path -> H^res
    P.append(t(632, 342, 'Hres_g', 9.5, MUT, mono=True))
    P.append(a(292, 490, 658, 480, ARR, 1.6))     # post path -> H^post
    P.append(t(470, 472, 'Hpost_g', 10, MUT, mono=True))
    # H boxes -> raw heads grad (activation bwd)
    P.append(a(832, 220, 1010, 300, ARR, 1.6))
    P.append(a(832, 350, 898, 350, ARR, 1.6))
    P.append(a(832, 480, 1010, 396, ARR, 1.6))
    # raw heads -> Linear bwd
    P.append(a(1020, 298, 1020, 172, ARR, 1.8))
    P.append(t(1075, 240, 'z_grad', 10.5, MUT, anchor='start', mono=True))
    # Linear -> RMSNorm (x̄_grad)
    P.append(a(898, 120, 772, 120, ARR, 1.8))
    P.append(t(835, 112, 'x̄_grad', 10.5, MUT, mono=True))
    # RMSNorm -> x_l_grad (prediction-path contribution)
    P.append(a(558, 120, 272, 120, ARR, 1.8))
    P.append(t(415, 112, '预测路 x_l_grad', 10.5, MUT, mono=True))
    # residual path & pre path contribute to x_l_grad too (dashed up-left)
    P.append(a(355, 330, 200, 160, ARR, 1.5, dash='5 4'))   # residual -> x_l_grad
    P.append(a(120, 186, 130, 160, ARR, 1.5, dash='5 4'))   # pre path -> x_l_grad
    P.append(t(250, 150, 'res 路', 9.5, MUT, mono=True))

    # Wφ_grad / param grads note
    P.append(opbox(900, 430, 240, 60, '参数梯度', [('Wφ_grad, α_grad, b_grad', True)], fill=GRN_F, st=GRN_S))
    P.append(a(1020, 396, 1020, 428, ARR, 1.4))

    # bottom summary
    P.append(box(40, 770, 1100, 84, YEL_F, YEL_S, rx=8, sw=2))
    P.append(t(W / 2, 800, 'x_l 的梯度 = 三条路径之和：res 路（Σ_r Hresᵀ·δ）+ pre 路（Hpre·u_grad）+ 预测路（RMSNorm∘Linear∘激活 反向）', 13.5, '#7a5b00', weight='bold'))
    P.append(t(W / 2, 826, 'pre = 沿流加权求和的反向；post = 沿流求和（广播的反向）；res = 沿流的 n×n 矩阵乘转置', 12, '#7a5b00'))
    P.append(t(W / 2, 845, '红框 = pre-norm-fn（RMSNorm + Linear），本文 CuteDSL kernel 负责的部分', 11.5, RED, weight='bold'))

    P.append('</svg>')
    svg = ''.join(P)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(svg)
    print('wrote', OUT, len(svg), 'bytes')


if __name__ == '__main__':
    main()
