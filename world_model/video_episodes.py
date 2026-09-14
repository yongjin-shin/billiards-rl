"""
Val 에피소드 normalized score 기반 best/worst 영상 생성.

score = mean_step_err(cm) / (n_gt_collisions + 1)
  - mean_step_err: 이미 스텝 수로 나눔 (curriculum 길이 차이 보정)
  - / (n_coll + 1): 충돌 많은 어려운 에피소드 보정

저장: world_model/videos/best_01_ep{idx}.mp4 / worst_01_ep{idx}.mp4
"""

import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from pathlib import Path

from world_model.ssm_model import SSMWorldModel, LATENT_DIM
from world_model.wm_predictor import TABLE_W, TABLE_H
from world_model.generate_data_fixeddt import DT
from world_model.val_cache import load_val_cache, CACHE_PATH

import argparse

TYPE_NAMES  = ['no_coll', 'ball_ball', 'linear', 'circular', 'pocket']
TYPE_COLORS = ['#303050', '#5588ff', '#44cc88', '#cc8844', '#ff5566']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    default='world_model/results/ssm_v16_5cls/best.pt')
    p.add_argument('--out-dir', default='world_model/videos')
    p.add_argument('--fps',      type=int, default=5)
    p.add_argument('--top-n',   type=int, default=5)
    p.add_argument('--random-n', type=int, default=0,
                   help='best/worst 대신 랜덤 N개 생성 (rand_XX_epYYY.gif)')
    p.add_argument('--seed',    type=int, default=None)
    args = p.parse_args()

    CKPT      = args.ckpt
    OUT_DIR   = Path(args.out_dir)
    FPS       = args.fps
    ROLLOUT_T = 60
    TRAIL_LEN = 8
    TOP_N     = args.top_n

    device = 'cpu'
    OUT_DIR.mkdir(exist_ok=True)
    HAS_FFMPEG = shutil.which('ffmpeg') is not None

    # ── 모델 로드 ─────────────────────────────────────────────────────
    model = SSMWorldModel(LATENT_DIM).to(device)
    ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state'])
    model.eval()
    print(f"Model: {CKPT}  (epoch={ckpt['epoch']}, err={ckpt['mean_err']:.1f}cm)")

    # ── 캐시 로드 ─────────────────────────────────────────────────────
    if not CACHE_PATH.exists():
        print(f"Cache not found. Run:  python world_model/val_cache.py")
        sys.exit(1)
    val_eps = load_val_cache()
    print(f"Loaded {len(val_eps)} episodes from cache")

    # ── 전 에피소드 스코어링 ──────────────────────────────────────────
    print("Scoring all episodes...")
    records = []
    with torch.no_grad():
        for i, (ep_s, ep_f, ep_t, n_cush) in enumerate(val_eps):
            T = min(ROLLOUT_T, len(ep_s) - 1)
            if T < 1:
                continue

            gf  = ep_f[:T]
            et  = ep_t[:T]
            n_coll = int(gf.sum())

            s0 = torch.from_numpy(ep_s[0:1]).float()
            sh, tl = model(s0, T)
            pr = sh[0].numpy()
            pc = (tl[0].argmax(-1).numpy() != 0)

            gt   = ep_s[1:T+1]
            pr_p = pr[1:T+1]
            cue_err = np.sqrt(((pr_p[:,0]-gt[:,0])*TABLE_W*100)**2 +
                              ((pr_p[:,1]-gt[:,1])*TABLE_H*100)**2)
            tgt_err = np.sqrt(((pr_p[:,7]-gt[:,7])*TABLE_W*100)**2 +
                              ((pr_p[:,8]-gt[:,8])*TABLE_H*100)**2)
            step_err = (cue_err + tgt_err) / 2

            mean_err = float(step_err.mean())
            tp = int((gf & pc[:len(gf)]).sum())
            fp = int((~gf & pc[:len(gf)]).sum())
            recall = tp / n_coll if n_coll > 0 else float('nan')
            score  = mean_err / (n_coll + 1)

            records.append(dict(
                idx=i, ep_s=ep_s, ep_f=gf, ep_t=et,
                pr=pr, step_err=step_err,
                T=T, n_coll=n_coll, n_cush=n_cush,
                mean_err=mean_err, tp=tp, fp=fp, recall=recall,
                score=score,
            ))

    records.sort(key=lambda x: x['score'])
    best_5  = records[:TOP_N]
    worst_5 = records[-TOP_N:][::-1]

    print(f"\n{'='*70}")
    print(f"{'idx':>4}  {'score':>6}  {'err':>7}  {'n_coll':>6}  {'n_cush':>6}  {'TP':>3}  {'FP':>3}  {'recall':>6}")
    print(f"--- BEST {TOP_N} ---")
    for r in best_5:
        print(f"{r['idx']:4d}  {r['score']:6.2f}  {r['mean_err']:6.1f}cm  {r['n_coll']:6d}  {r['n_cush']:6d}  "
              f"{r['tp']:3d}  {r['fp']:3d}  {r['recall']:6.2f}")
    print(f"--- WORST {TOP_N} ---")
    for r in worst_5:
        print(f"{r['idx']:4d}  {r['score']:6.2f}  {r['mean_err']:6.1f}cm  {r['n_coll']:6d}  {r['n_cush']:6d}  "
              f"{r['tp']:3d}  {r['fp']:3d}  {r['recall']:6.2f}")
    print(f"{'='*70}")


    # ── 영상 생성 ─────────────────────────────────────────────────────
    def make_video(rec: dict, label: str, rank: int):
        ep_s     = rec['ep_s']
        gf       = rec['ep_f']   # (T,) bool
        et       = rec['ep_t']   # (T,) int
        pr       = rec['pr']     # (T+1, 14)
        step_err = rec['step_err']
        T        = rec['T']
        gt       = ep_s[:T+1]   # (T+1, 14)
    
        # ── 그림 레이아웃 ─────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 7), facecolor='#0d0f1a')
        ax_t = fig.add_axes([0.03, 0.08, 0.40, 0.82])   # 탑뷰 (portrait table)
        ax_e = fig.add_axes([0.52, 0.55, 0.45, 0.35])   # 오차 차트
        ax_c = fig.add_axes([0.52, 0.10, 0.45, 0.35])   # 타입 확률 차트 placeholder
    
        # ── 탑뷰 설정 ─────────────────────────────────────────────────
        ax_t.set_facecolor('#1a3a1a')
        ax_t.set_xlim(-0.04, 1.04)
        ax_t.set_ylim(-0.04, 1.04)
        ax_t.set_aspect('equal')
        ax_t.axis('off')
    
        # 테이블 테두리
        ax_t.add_patch(plt.Rectangle((0,0), 1, 1, fill=False,
                                      edgecolor='#8B6914', lw=4, zorder=2))
        # 포켓 6개 (portrait 테이블: 긴 변=y축, 사이드 포켓은 y=0.5)
        POCKET_XY = [(0,0),(1,0),(0,0.5),(1,0.5),(0,1),(1,1)]
        for px, py in POCKET_XY:
            ax_t.add_patch(plt.Circle((px, py), 0.028, color='#111', zorder=3))
    
        # 배경 flash용 rect
        bg = plt.Rectangle((0,0), 1, 1, color='#1a3a1a', zorder=0)
        ax_t.add_patch(bg)
    
        # trails
        cue_gt_tr, = ax_t.plot([], [], color='#5588ff', lw=2,   alpha=0.7, zorder=4)
        cue_pr_tr, = ax_t.plot([], [], color='#ff6644', lw=2,   alpha=0.7, ls='--', zorder=4)
        tgt_gt_tr, = ax_t.plot([], [], color='#44cc88', lw=2,   alpha=0.7, zorder=4)
        tgt_pr_tr, = ax_t.plot([], [], color='#ffaa44', lw=2,   alpha=0.7, ls='--', zorder=4)
    
        # 현재 위치 (실선=GT, 빈원=Pred)
        cue_gt_d, = ax_t.plot([], [], 'o',  color='#5588ff', ms=12, zorder=6)
        cue_pr_d, = ax_t.plot([], [], 'o',  color='#ff6644', ms=12, zorder=6, mfc='none', mew=2.5)
        tgt_gt_d, = ax_t.plot([], [], 'o',  color='#44cc88', ms=12, zorder=6)
        tgt_pr_d, = ax_t.plot([], [], 'o',  color='#ffaa44', ms=12, zorder=6, mfc='none', mew=2.5)
    
        info_txt = ax_t.text(0.02, 0.98, '', transform=ax_t.transAxes,
                              color='#e0e8ff', fontsize=9, va='top', fontfamily='monospace',
                              zorder=7)
        legend_h = [
            mpatches.Patch(color='#5588ff', label='GT cue'),
            mpatches.Patch(color='#ff6644', label='Pred cue'),
            mpatches.Patch(color='#44cc88', label='GT tgt'),
            mpatches.Patch(color='#ffaa44', label='Pred tgt'),
        ]
        ax_t.legend(handles=legend_h, loc='lower right', fontsize=8,
                    facecolor='#1a1d2e', edgecolor='#2a2d45', labelcolor='#c0c8e8')
    
        # ── 오차 차트 ─────────────────────────────────────────────────
        spine_c = '#2a2d45'
        for ax_ in [ax_e, ax_c]:
            ax_.set_facecolor('#0d0f1a')
            for sp in ax_.spines.values(): sp.set_color(spine_c)
            ax_.tick_params(colors='#6068a0', labelsize=7)
    
        ts = np.arange(1, T+1) * DT
        ax_e.plot(ts, step_err, color='#aaaadd', lw=1.2, alpha=0.8)
        for t_i in np.where(gf)[0]:
            ax_e.axvline((t_i+1)*DT, color=TYPE_COLORS[et[t_i]], alpha=0.6, lw=1)
        ax_e.set_xlim(0, T*DT)
        ax_e.set_ylim(0, max(step_err.max()*1.15, 5))
        ax_e.set_ylabel('err (cm)', color='#8890b0', fontsize=8)
        ax_e.set_title('per-step position error', color='#c8d0f0', fontsize=8.5)
        vline_e = ax_e.axvline(0, color='#ffffff', lw=1.5, alpha=0.8)
    
        # ── 타입별 GT vs Pred ─────────────────────────────────────────
        # collision flags bar (GT=상단, Pred=하단)
        ax_c.set_xlim(0, T*DT)
        ax_c.set_ylim(-0.5, 1.5)
        ax_c.set_yticks([0, 1])
        ax_c.set_yticklabels(['pred', 'GT'], fontsize=7, color='#8890b0')
        ax_c.set_xlabel('time (s)', color='#8890b0', fontsize=8)
        ax_c.set_title('collision detection', color='#c8d0f0', fontsize=8.5)
    
        for t_i in np.where(gf)[0]:
            ax_c.scatter((t_i+1)*DT, 1, marker='x', s=60,
                         color=TYPE_COLORS[et[t_i]], zorder=5, lw=2)
    
        # pred collision markers (drawn after init — static for blit compat)
        pc_pred = rec.get('pc_cache', None)
        if pc_pred is None:
            # reconstruct from pr
            with torch.no_grad():
                s0_ = torch.from_numpy(ep_s[0:1]).float()
                _, tl_ = model(s0_, T)
                pc_pred = (tl_[0].argmax(-1).numpy() != 0)
            rec['pc_cache'] = pc_pred
    
        for t_i in np.where(pc_pred[:T])[0]:
            match = bool(gf[t_i]) if t_i < len(gf) else False
            ax_c.scatter((t_i+1)*DT, 0, marker='^', s=40,
                         color='#44cc88' if match else '#ff5555', alpha=0.85, zorder=4)
    
        vline_c = ax_c.axvline(0, color='#ffffff', lw=1.5, alpha=0.8)
    
        # ── 타이틀 ────────────────────────────────────────────────────
        r_str = f"{rec['recall']:.2f}" if not np.isnan(rec['recall']) else 'n/a'
        fig.suptitle(
            f"[{label.upper()} #{rank+1}]  ep={rec['idx']}  "
            f"score={rec['score']:.2f}  err={rec['mean_err']:.1f}cm  "
            f"n_coll={rec['n_coll']}  TP={rec['tp']}  FP={rec['fp']}  recall={r_str}",
            color='#c8d0f0', fontsize=9.5, fontweight='bold'
        )
    
        # ── 애니메이션 ────────────────────────────────────────────────
        def on_table(x, y): return -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1
    
        def update(frame):
            t = frame + 1   # 1-indexed
            s = max(0, t - TRAIL_LEN)
    
            cue_gt_tr.set_data(gt[s:t+1, 0], gt[s:t+1, 1])
            cue_pr_tr.set_data(pr[s:t+1, 0], pr[s:t+1, 1])
            # trail: 포켓된 구간(테이블 밖) 제외
            tgt_mask_gt = np.array([on_table(gt[i,7], gt[i,8]) for i in range(s, t+1)])
            tgt_mask_pr = np.array([on_table(pr[i,7], pr[i,8]) for i in range(s, t+1)])
            tgt_gt_tr.set_data(gt[s:t+1, 7][tgt_mask_gt], gt[s:t+1, 8][tgt_mask_gt])
            tgt_pr_tr.set_data(pr[s:t+1, 7][tgt_mask_pr], pr[s:t+1, 8][tgt_mask_pr])
    
            cue_gt_d.set_data([gt[t, 0]], [gt[t, 1]])
            cue_pr_d.set_data([pr[t, 0]], [pr[t, 1]])
            if on_table(gt[t, 7], gt[t, 8]):
                tgt_gt_d.set_data([gt[t, 7]], [gt[t, 8]])
            else:
                tgt_gt_d.set_data([], [])
            if on_table(pr[t, 7], pr[t, 8]):
                tgt_pr_d.set_data([pr[t, 7]], [pr[t, 8]])
            else:
                tgt_pr_d.set_data([], [])
    
            cur_t = t * DT
            vline_e.set_xdata([cur_t])
            vline_c.set_xdata([cur_t])
    
            is_coll = bool(gf[t-1]) if t-1 < len(gf) else False
            bg.set_facecolor('#2d4a2d' if is_coll else '#1a3a1a')
    
            coll_type = et[t-1] if (t-1 < len(et) and is_coll) else 0
            gt_str  = TYPE_NAMES[coll_type] if is_coll else 'no_coll'
            err_val = step_err[t-1]
            info_txt.set_text(
                f"t={t:2d}  {cur_t:.2f}s\nerr={err_val:.1f}cm\nGT: {gt_str}"
            )
    
            return (cue_gt_tr, cue_pr_tr, tgt_gt_tr, tgt_pr_tr,
                    cue_gt_d, cue_pr_d, tgt_gt_d, tgt_pr_d,
                    vline_e, vline_c, bg, info_txt)
    
        anim = FuncAnimation(fig, update, frames=T,
                             interval=1000//FPS, blit=True)
    
        ext  = 'mp4' if HAS_FFMPEG else 'gif'
        path = OUT_DIR / f"{label}_{rank+1:02d}_ep{rec['idx']}.{ext}"
        if HAS_FFMPEG:
            anim.save(str(path), writer=FFMpegWriter(fps=FPS, bitrate=1800))
        else:
            anim.save(str(path), writer=PillowWriter(fps=FPS))
        plt.close(fig)
        print(f"  Saved → {path}")


    if args.random_n > 0:
        rng = np.random.default_rng(args.seed)
        rand_recs = rng.choice(records, size=min(args.random_n, len(records)), replace=False).tolist()
        print(f"\nGenerating {len(rand_recs)} random videos ({'mp4' if HAS_FFMPEG else 'gif'})...")
        for rank, rec in enumerate(rand_recs):
            print(f"rand {rank+1}/{len(rand_recs)}  ep={rec['idx']}  score={rec['score']:.2f}  err={rec['mean_err']:.1f}cm", end='  ', flush=True)
            make_video(rec, 'rand', rank)
    else:
        print(f"\nGenerating {TOP_N*2} videos ({'mp4' if HAS_FFMPEG else 'gif'})...")
        for rank, rec in enumerate(best_5):
            print(f"best {rank+1}/{TOP_N}  ep={rec['idx']}", end='  ', flush=True)
            make_video(rec, 'best', rank)
        for rank, rec in enumerate(worst_5):
            print(f"worst {rank+1}/{TOP_N}  ep={rec['idx']}", end='  ', flush=True)
            make_video(rec, 'worst', rank)

    print("Done.")


if __name__ == '__main__':
    main()
