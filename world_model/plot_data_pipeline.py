"""world_model/plot_data_pipeline.py — GT 데이터 생성 파이프라인 시각화"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm

# macOS 한글 폰트
for fname in ['AppleGothic', 'NanumGothic', 'Malgun Gothic', 'DejaVu Sans']:
    try:
        fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
        plt.rcParams['font.family'] = fname
        break
    except Exception:
        continue

fig, axes = plt.subplots(1, 2, figsize=(20, 14))
fig.patch.set_facecolor('#0f1117')

# ─── 색상 팔레트 ───
C_BOX   = '#1e2130'
C_BOX2  = '#252840'
C_EDGE  = '#4a5080'
C_ARROW = '#7080c0'
C_TITLE = '#c8d0f0'
C_SUB   = '#8890b0'
C_CODE  = '#a0d0a0'
C_HIGH  = '#f0c060'
C_RED   = '#e06060'
C_BLUE  = '#6090e0'
C_GREEN = '#60c080'
C_PURP  = '#c060c0'
C_ORNG  = '#e09040'

# ══════════════════════════════════════════════════════════
# AX 0 : 데이터 생성 파이프라인
# ══════════════════════════════════════════════════════════
ax = axes[0]
ax.set_facecolor('#0f1117')
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis('off')
ax.set_title('① 데이터 수집 파이프라인  (generate_data_fixeddt.py)',
             color=C_TITLE, fontsize=13, fontweight='bold', pad=10)


def box(ax, x, y, w, h, label, sublabel='', color=C_BOX, edgecolor=C_EDGE,
        fontsize=10, lw=1.5, code=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=edgecolor, lw=lw)
    ax.add_patch(rect)
    tc = C_CODE if code else C_TITLE
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha='center', va='center', color=tc, fontsize=fontsize, fontweight='bold')
    if sublabel:
        ax.text(x, y - 0.28, sublabel, ha='center', va='center',
                color=C_SUB, fontsize=8)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, color=C_HIGH, fontsize=8, va='center')


# 1. BilliardsEnv + policy
box(ax, 5, 17.0, 8, 0.9, 'BilliardsEnv  +  SAC policy',
    'obs(16dim) → action(2dim)', color='#1a2035', edgecolor=C_BLUE, lw=2)
arrow(ax, 5, 16.55, 5, 15.85)

# 2. pooltool simulate
box(ax, 5, 15.5, 7, 0.9, 'pooltool.simulate(shot)',
    '물리엔진: 쿠션/충돌/포켓 이벤트 계산', color='#1a2035', edgecolor=C_GREEN, lw=2)
arrow(ax, 5, 15.05, 5, 14.25)

# 3. 두 산출물: events + trajectory
box(ax, 2.5, 13.9, 3.8, 0.8, 'events[]',
    '[(time, type)] 실수 시간', color=C_BOX2, edgecolor=C_PURP)
box(ax, 7.2, 13.9, 3.8, 0.8, 'ball trajectory',
    'cue & tgt 연속 상태', color=C_BOX2, edgecolor=C_PURP)

# 4. fixed DT sampling
arrow(ax, 2.5, 13.50, 5, 12.65, color=C_PURP)
arrow(ax, 7.2, 13.50, 5, 12.65, color=C_PURP)
box(ax, 5, 12.3, 7, 0.8, 'Fixed-DT 샘플링   DT = 0.05 s',
    'timestamps = [0, 0.05, 0.10, …, T·0.05]', color='#1a2035', edgecolor=C_HIGH, lw=2)
arrow(ax, 5, 11.9, 5, 11.1)

# 5. window별 레코드
box(ax, 5, 10.7, 8.5, 0.8,
    'for each window [t_i, t_{i+1})',
    'states[i] = interpolate(cue, tgt)  →  14dim', color=C_BOX2, edgecolor=C_EDGE)
arrow(ax, 5, 10.3, 5, 9.5)

# 6. coll_flags / coll_types
box(ax, 2.8, 9.1, 3.8, 0.8, 'coll_flags[i]',
    'True if event in window', color=C_BOX2, edgecolor=C_RED)
box(ax, 7.2, 9.1, 3.8, 0.8, 'coll_types[i]',
    '-1/0/1/2/3 (first event)', color=C_BOX2, edgecolor=C_RED)
ax.annotate('', xy=(2.8, 9.5), xytext=(5, 9.5),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
ax.annotate('', xy=(7.2, 9.5), xytext=(5, 9.5),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
arrow(ax, 2.8, 8.7, 5, 7.9, color=C_RED)
arrow(ax, 7.2, 8.7, 5, 7.9, color=C_RED)

# 7. .npz 저장
box(ax, 5, 7.55, 8.5, 0.8,
    'save  →  .npz',
    'states(N,220,14) / coll_flags(N,220) / coll_types(N,220,int8)', color='#1a2035',
    edgecolor=C_ORNG, lw=2)
arrow(ax, 5, 7.15, 5, 6.35)

# 8. SSMDataset 로드 + v16 remap
box(ax, 5, 6.0, 8.5, 0.8,
    'SSMDataset  —  v16 remap: ep_t = coll_types + 1',
    '', color='#1a2035', edgecolor=C_GREEN, lw=2)

# remap table
remap_x = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
remap_orig = ['-1', '0', '1', '2', '3', '']
remap_v16  = ['0', '1', '2', '3', '4', '']
remap_name = ['no_coll', 'ball_ball', 'linear', 'circular', 'pocket', '']
remap_col  = [C_SUB, C_BLUE, C_GREEN, C_GREEN, C_ORNG, C_SUB]

for i, (xr, orig, v16, name, col) in enumerate(
        zip(remap_x, remap_orig, remap_v16, remap_name, remap_col)):
    if orig == '':
        continue
    ax.text(xr, 5.35, orig,  ha='center', va='center', color=C_SUB,  fontsize=9)
    ax.text(xr, 5.05, '↓',   ha='center', va='center', color=C_ARROW, fontsize=9)
    ax.text(xr, 4.75, v16,   ha='center', va='center', color=col,    fontsize=10, fontweight='bold')
    ax.text(xr, 4.40, name,  ha='center', va='center', color=col,    fontsize=7.5)

ax.axhline(5.6, xmin=0.05, xmax=0.95, color=C_EDGE, lw=0.7, ls='--')
ax.axhline(4.6, xmin=0.05, xmax=0.95, color=C_EDGE, lw=0.7, ls='--')

arrow(ax, 5, 4.15, 5, 3.35)

# 9. episodes list
box(ax, 5, 3.0, 8.5, 0.8,
    'episodes = [(ep_s, ep_f, ep_t, n_cush)]  →  DataLoader',
    'ep_s(L,14)  ep_f(L-1,)bool  ep_t(L-1,)int  n_cush:int', color='#1a2035',
    edgecolor=C_BLUE, lw=2)
arrow(ax, 5, 2.6, 5, 1.85)

# 10. seq output
box(ax, 5, 1.5, 8.5, 0.8,
    'batch output: seq_s(B,T+1,14)   seq_f(B,T)   seq_t(B,T)',
    '', color='#1a2035', edgecolor=C_ARROW, lw=1.5)


# ══════════════════════════════════════════════════════════
# AX 1 : Train loop GT 조립 & _decode_ar_state
# ══════════════════════════════════════════════════════════
ax2 = axes[1]
ax2.set_facecolor('#0f1117')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 18)
ax2.axis('off')
ax2.set_title('② Train loop  — gt_ar 조립 &  _decode_ar_state',
              color=C_TITLE, fontsize=13, fontweight='bold', pad=10)


def box2(ax, x, y, w, h, label, sublabel='', color=C_BOX,
         edgecolor=C_EDGE, fontsize=10, lw=1.5):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=edgecolor, lw=lw)
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha='center', va='center', color=C_TITLE, fontsize=fontsize, fontweight='bold')
    if sublabel:
        ax.text(x, y - 0.28, sublabel, ha='center', va='center',
                color=C_SUB, fontsize=7.5)


def arr2(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, label='', lx=0.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    if label:
        ax.text((x1+x2)/2 + lx, (y1+y2)/2, label,
                color=C_HIGH, fontsize=8, va='center')


# ─── gt_ar 조립 ───
box2(ax2, 5, 17.0, 8.5, 0.9,
     'DataLoader batch',
     'seq_s(B,T+1,14)   seq_t(B,T)   seq_f(B,T)',
     color='#1a2035', edgecolor=C_BLUE, lw=2)

arr2(ax2, 3.0, 16.55, 2.5, 15.85, color=C_BLUE)
arr2(ax2, 7.0, 16.55, 7.5, 15.85, color=C_ORNG)

box2(ax2, 2.5, 15.5, 4.0, 0.8,
     'seq_s[:, :T]',
     'physical GT  (B,T,14)', color=C_BOX2, edgecolor=C_BLUE)
box2(ax2, 7.5, 15.5, 4.0, 0.8,
     'F.one_hot(seq_t, 5)',
     'type one-hot  (B,T,5)', color=C_BOX2, edgecolor=C_ORNG)

arr2(ax2, 2.5, 15.1, 5, 14.25, color=C_BLUE)
arr2(ax2, 7.5, 15.1, 5, 14.25, color=C_ORNG)

box2(ax2, 5, 13.9, 8.5, 0.8,
     'gt_ar  =  cat([physical, one_hot],  dim=-1)',
     'shape: (B, T, 19)',
     color='#1a2035', edgecolor=C_HIGH, lw=2.5, fontsize=11)

# ─── 모델 forward ───
arr2(ax2, 5, 13.5, 5, 12.7, color=C_ARROW)

box2(ax2, 5, 12.35, 8.5, 0.8,
     'model(seq_s[:,0], T, gt_states=gt_ar, ss_ratio)',
     'rollout: for t in range(T)',
     color='#1a2035', edgecolor=C_GREEN, lw=2)

arr2(ax2, 5, 11.95, 5, 11.15, color=C_GREEN)

# ─── _decode_ar_state ───
box2(ax2, 5, 10.8, 8.5, 0.8,
     '_decode_ar_state(z_t,  gt_ar[:,t],  t,  ss_ratio)',
     '', color='#202535', edgecolor=C_EDGE, lw=1.5, fontsize=10)

# decoded branch
arr2(ax2, 5, 10.4, 5, 9.65, color=C_ARROW)

box2(ax2, 5, 9.3, 8.5, 0.8,
     'decoded = cat([cue_head(z), tgt_head(z), softmax(type_head(z))])',
     'pred physical(14) + pred type prob(5) = 19dim',
     color=C_BOX2, edgecolor=C_PURP)

# ss_ratio 분기
arr2(ax2, 5, 8.9, 5, 8.1, color=C_ARROW)

# 분기 다이아몬드
diam_x, diam_y = 5, 7.75
d = 0.55
diamond = plt.Polygon([[diam_x, diam_y+d], [diam_x+d*1.8, diam_y],
                        [diam_x, diam_y-d], [diam_x-d*1.8, diam_y]],
                      facecolor='#252840', edgecolor=C_HIGH, lw=2)
ax2.add_patch(diamond)
ax2.text(diam_x, diam_y, 'ss_ratio', ha='center', va='center',
         color=C_HIGH, fontsize=9, fontweight='bold')

# 4 branches
# ss=1.0 (left-top)
ax2.annotate('', xy=(1.5, 6.6), xytext=(3.2, 7.75),
             arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.5))
ax2.text(1.7, 7.3, '≥ 1.0', color=C_BLUE, fontsize=8, fontweight='bold')
box2(ax2, 1.5, 6.3, 2.8, 0.6,
     'return gt_ar[:,t]',
     'teacher forcing', color='#1a2035', edgecolor=C_BLUE, lw=1.5, fontsize=9)

# ss=0.0 (right-top)
ax2.annotate('', xy=(8.5, 6.6), xytext=(6.8, 7.75),
             arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.5))
ax2.text(7.5, 7.3, '= 0.0', color=C_GREEN, fontsize=8, fontweight='bold')
box2(ax2, 8.5, 6.3, 2.8, 0.6,
     'return decoded',
     'pure inference', color='#1a2035', edgecolor=C_GREEN, lw=1.5, fontsize=9)

# 0<ss<1 (left-bottom)
ax2.annotate('', xy=(1.5, 4.9), xytext=(3.2, 7.2),
             arrowprops=dict(arrowstyle='->', color=C_ORNG, lw=1.5))
ax2.text(1.1, 6.1, '0 < ss < 1', color=C_ORNG, fontsize=8, fontweight='bold')
box2(ax2, 1.5, 4.5, 2.9, 0.8,
     'per-sample mix',
     'Bernoulli(ss): GT or decoded', color='#1a2035', edgecolor=C_ORNG, lw=1.5, fontsize=8)

# ss<0 BERT (right-bottom)
ax2.annotate('', xy=(8.5, 4.9), xytext=(6.8, 7.2),
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))
ax2.text(7.5, 6.1, 'ss < 0', color=C_RED, fontsize=8, fontweight='bold')
box2(ax2, 8.5, 4.4, 2.9, 1.0,
     'BERT masking',
     'ss_b~U(0,|ss|)\nmask[b,d]~Bern(ss_b)\nwhere(mask, GT, decoded)',
     color='#1a2035', edgecolor=C_RED, lw=1.5, fontsize=8)

# ─── 합류 → ar_state ───
arr2(ax2, 1.5, 4.1, 5, 3.4, color=C_ARROW)
arr2(ax2, 8.5, 3.9, 5, 3.4, color=C_ARROW)

box2(ax2, 5, 3.1, 8.5, 0.7,
     'ar_state  (B, 19)  →  transition(z_t, ar_state)',
     '', color='#1a2035', edgecolor=C_HIGH, lw=2, fontsize=10)

arr2(ax2, 5, 2.75, 5, 2.05, color=C_ARROW)

box2(ax2, 5, 1.7, 8.5, 0.7,
     'z_{t+1} = LayerNorm(z_t + MLP([z_t ; ar_state]))',
     '', color='#202535', edgecolor=C_PURP, lw=1.5, fontsize=10)

arr2(ax2, 5, 1.35, 5, 0.75, color=C_ARROW)

box2(ax2, 5, 0.45, 8.5, 0.6,
     's_hat(B,T+1,14)   type_logit(B,T,5)   →   loss',
     '', color='#1a2035', edgecolor=C_EDGE, fontsize=9)

# ─── 범례 ───
leg_items = [
    mpatches.Patch(color=C_BLUE,  label='physical GT (14dim)'),
    mpatches.Patch(color=C_ORNG,  label='one-hot type (5dim)'),
    mpatches.Patch(color=C_PURP,  label='decoded (pred)'),
    mpatches.Patch(color=C_RED,   label='BERT masking'),
    mpatches.Patch(color=C_GREEN, label='pure inference'),
]
ax2.legend(handles=leg_items, loc='lower right',
           facecolor='#1e2130', edgecolor=C_EDGE,
           labelcolor=C_TITLE, fontsize=8, framealpha=0.9)

plt.tight_layout(pad=1.5)
plt.savefig('world_model/data_pipeline.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved → world_model/data_pipeline.png")
plt.show()
