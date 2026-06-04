#!/usr/bin/env python3
"""
generate_report_charts_academic.py
Academic-style ABSA evaluation charts — seaborn ticks theme, minimal annotation
Data source: reports/stage_metrics.json
Output:      reports/charts_academic/*.png
"""

import sys
import os
import json
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

sns.set_theme(style='ticks', palette='muted')

# Set CJK font AFTER sns.set_theme to prevent override
matplotlib.rcParams['font.family'] = [
    'DFKai-SB', 'BiauKai', 'TW-Kai',
    'Microsoft JhengHei', 'Microsoft YaHei',
    'KaiTi', 'SimHei', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Semantic color scheme (colorblind-friendly)
TRACK_A = '#4878D0'   # blue  — rule-based
TRACK_B = '#EE854A'   # orange — LLM zero-shot
ITER_C  = '#4878D0'   # single-track iteration bars
BEST_C  = '#D65F5F'   # best iteration highlight
ABAND   = '#CCCCCC'   # abandoned variant
TP_C    = '#6ACC65'   # true positive
FP_C    = '#D65F5F'   # false positive
FN_C    = '#EE854A'   # false negative

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(SCRIPT_DIR, 'reports', 'stage_metrics.json')
OUT_DIR      = os.path.join(SCRIPT_DIR, 'reports', 'charts_academic')
os.makedirs(OUT_DIR, exist_ok=True)


def _load_metrics():
    with open(METRICS_PATH, encoding='utf-8') as f:
        return json.load(f)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)
    print(f'  Saved: {name}')


def _despine(ax):
    sns.despine(ax=ax)
    ax.grid(axis='x', color='#E0E0E0', linewidth=0.7, zorder=0)


def _despine_y(ax):
    sns.despine(ax=ax)
    ax.grid(axis='y', color='#E0E0E0', linewidth=0.7, zorder=0)


def _status_color(status):
    return {'best': BEST_C, 'regression': FP_C, 'iter': ITER_C, 'abandoned': ABAND}[status]


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1 — Stage 1: Lollipop chart（Cleveland dot plot）
# ═══════════════════════════════════════════════════════════════════════════
def chart1_stage1_iteration(m):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    iters = m['stage1']['iterations']

    for i, it in enumerate(iters):
        if it['status'] == 'abandoned':
            ax.axhline(y=i, xmin=0.0, xmax=0.55, color=ABAND,
                       linewidth=1.0, linestyle=':')
            ax.text(0.594, i, 'abandoned', va='center', ha='left',
                    fontsize=9, color='#999999', style='italic')
        else:
            color  = _status_color(it['status'])
            is_best = it['status'] == 'best'
            ax.plot([0.58, it['f1']], [i, i], color=color,
                    linewidth=1.2, solid_capstyle='round', zorder=2)
            ax.scatter(it['f1'], i, s=90, color=color, zorder=4,
                       marker='D' if is_best else 'o',
                       edgecolors='white', linewidths=0.8)
            ax.text(it['f1'] + 0.004, i, f"{it['f1']:.4f}", va='center',
                    ha='left', fontsize=9, color='#333333')

    best_id = next(it['id'] for it in iters if it['status'] == 'best')
    ax.set_yticks(range(len(iters)))
    ax.set_yticklabels([it['label'] for it in iters], fontsize=9.5)
    ax.set_xlabel('Partial F1', fontsize=11)
    ax.set_xlim(0.58, 0.78)
    ax.set_title('Stage 1 — Aspect Term Extraction 迭代對比', fontsize=12, pad=10)
    ax.tick_params(axis='y', length=0)

    legend_els = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=BEST_C,
               markersize=8, label=f'Best ({best_id})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ITER_C,
               markersize=7, label='Iteration'),
        Line2D([0], [0], linestyle=':', color=ABAND, label='Abandoned'),
    ]
    ax.legend(handles=legend_els, frameon=False, fontsize=9, loc='lower right')
    _despine(ax)
    fig.tight_layout()
    _save(fig, 'chart1_stage1_iteration.png')


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2 — Stage 2: Connected dot plot + UI_UX Recall
# ═══════════════════════════════════════════════════════════════════════════
def chart2_stage2_iteration(m):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    iters = m['stage2']['iterations']

    # ── Left: connected dot plot, F1 progression ───────────────────────────
    labels  = [it['label'] for it in iters]
    f1_vals = [it['f1']    for it in iters]
    x = np.arange(len(iters))

    ax1.plot(x, f1_vals, color='#AAAAAA', linewidth=1.0, zorder=1)
    for xi, it in enumerate(iters):
        color  = _status_color(it['status'])
        marker = 'D' if it['status'] == 'best' else 'o'
        ax1.scatter(xi, it['f1'], s=80, color=color, zorder=3,
                    marker=marker, edgecolors='white', linewidths=0.8)
        ax1.text(xi, it['f1'] + 0.007, f"{it['f1']:.4f}", ha='center',
                 va='bottom', fontsize=8.5, color='#444444')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Explicit Category Macro F1', fontsize=10)
    ax1.set_ylim(0.52, 0.73)
    ax1.set_title('Stage 2 — Category Mapping 迭代', fontsize=11, pad=8)
    ax1.tick_params(axis='x', length=0)
    legend_els = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=BEST_C,
               markersize=8, label='Best'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FP_C,
               markersize=7, label='Regression'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ITER_C,
               markersize=7, label='Iteration'),
    ]
    ax1.legend(handles=legend_els, frameon=False, fontsize=8.5, loc='upper left')
    _despine_y(ax1)

    # ── Right: UI_UX Recall before/after ───────────────────────────────────
    recall_data = m['stage2']['uiux_recall']
    cat_labels  = [r['id']     for r in recall_data]
    recalls     = [r['recall'] for r in recall_data]

    bars = ax2.bar(cat_labels, recalls, color=[ABAND, BEST_C],
                   width=0.4, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, recalls):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.015, f'{val:.2%}',
                 ha='center', va='bottom', fontsize=10, color='#333333')

    ax2.set_ylabel('UI_UX Recall', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Stage 2 — UI_UX Recall 修正', fontsize=11, pad=8)
    ax2.tick_params(axis='x', length=0)
    _despine_y(ax2)

    fig.tight_layout()
    _save(fig, 'chart2_stage2_iteration.png')


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3 — Stage 3: Lollipop + TP/FP/FN grouped bar
# ═══════════════════════════════════════════════════════════════════════════
def chart3_stage3_iteration(m):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    iters = m['stage3']['iterations']

    # ── Left: lollipop F1 ───────────────────────────────────────────────────
    for i, it in enumerate(iters):
        color   = _status_color(it['status'])
        is_best = it['status'] == 'best'
        ax1.plot([0.76, it['f1']], [i, i], color=color, linewidth=1.2,
                 solid_capstyle='round', zorder=2)
        ax1.scatter(it['f1'], i, s=80, color=color, zorder=4,
                    marker='D' if is_best else 'o',
                    edgecolors='white', linewidths=0.8)
        ax1.text(it['f1'] + 0.003, i, f"{it['f1']:.4f}", va='center',
                 ha='left', fontsize=9, color='#333333')

    ax1.set_yticks(range(len(iters)))
    ax1.set_yticklabels([it['label'] for it in iters], fontsize=9.5)
    ax1.set_xlabel('Partial F1', fontsize=10)
    ax1.set_xlim(0.76, 0.91)
    ax1.set_title('Stage 3 — Opinion Extraction 迭代對比', fontsize=11, pad=8)
    ax1.tick_params(axis='y', length=0)
    _despine(ax1)

    # ── Right: TP / FP / FN grouped bar ────────────────────────────────────
    ids     = [it['id'] for it in iters]
    tp_vals = [it['tp'] for it in iters]
    fp_vals = [it['fp'] for it in iters]
    fn_vals = [it['fn'] for it in iters]
    x = np.arange(len(iters))
    w = 0.25

    b1 = ax2.bar(x - w, tp_vals, w, label='TP', color=TP_C, edgecolor='white')
    b2 = ax2.bar(x,     fp_vals, w, label='FP', color=FP_C, edgecolor='white')
    b3 = ax2.bar(x + w, fn_vals, w, label='FN', color=FN_C, edgecolor='white')

    for grp in [b1, b2, b3]:
        for bar in grp:
            h = int(bar.get_height())
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                     str(h), ha='center', va='bottom', fontsize=8.5,
                     color='#444444')

    ax2.set_xticks(x)
    ax2.set_xticklabels(ids, fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Stage 3 — TP / FP / FN 變化', fontsize=11, pad=8)
    ax2.legend(frameon=False, fontsize=9, loc='upper right')
    ax2.tick_params(axis='x', length=0)
    _despine_y(ax2)

    fig.tight_layout()
    _save(fig, 'chart3_stage3_iteration.png')


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4 — Stage 4: P/R grouped bar + F1 line (dual-axis, clean style)
# ═══════════════════════════════════════════════════════════════════════════
def chart4_stage4_pairing(m):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    iters  = m['stage4']['iterations']
    labels = [it['label'] for it in iters]
    prec   = [it['tp'] / (it['tp'] + it['fp']) for it in iters]
    rec    = [it['tp'] / (it['tp'] + it['fn']) for it in iters]
    f1     = [it['f1'] for it in iters]

    status_bar_p = {'iter': '#A8C8F0', 'regression': '#F5B8B8', 'best': '#F5D5A0'}
    status_bar_r = {'iter': ITER_C,    'regression': FP_C,       'best': BEST_C}
    bar_colors_p = [status_bar_p[it['status']] for it in iters]
    bar_colors_r = [status_bar_r[it['status']] for it in iters]

    x = np.arange(len(iters))
    w = 0.32

    bars_p = ax.bar(x - w / 2, prec, w, label='Precision',
                    color=bar_colors_p, edgecolor='white', linewidth=0.8)
    bars_r = ax.bar(x + w / 2, rec,  w, label='Recall',
                    color=bar_colors_r, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars_p, prec):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.010,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8.5, color='#555555')
    for bar, val in zip(bars_r, rec):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.010,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8.5, color='#333333')

    ax.set_ylabel('Precision  /  Recall', fontsize=11)
    ax.set_ylim(0, 0.88)

    ax2 = ax.twinx()
    ax2.plot(x, f1, 'o-', color='#555555', linewidth=1.5,
             markersize=7, markerfacecolor='white', markeredgewidth=1.5, zorder=5)
    for xi, yi in zip(x, f1):
        ax2.text(xi, yi + 0.014, f'{yi:.4f}', ha='center', va='bottom',
                 fontsize=8.5, color='#555555')
    ax2.set_ylabel('Partial F1', fontsize=11, color='#555555')
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax2.set_ylim(0.28, 0.68)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('#CCCCCC')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title('Stage 4 — Aspect-Opinion Pairing：Precision / Recall / F1', fontsize=11, pad=10)
    ax.tick_params(axis='x', length=0)

    legend_els = [
        mpatches.Patch(color='#A8C8F0', label='Precision (light)'),
        mpatches.Patch(color=ITER_C,    label='Recall (solid)'),
        Line2D([0], [0], color='#555555', marker='o', linewidth=1.5,
               markerfacecolor='white', markeredgewidth=1.5, label='Partial F1'),
    ]
    ax.legend(handles=legend_els, frameon=False, fontsize=9, loc='upper left')
    sns.despine(ax=ax, right=False)
    ax.grid(axis='y', color='#E0E0E0', linewidth=0.7, zorder=0)
    fig.tight_layout()
    _save(fig, 'chart4_stage4_pairing.png')


# ═══════════════════════════════════════════════════════════════════════════
# Chart 5 — Pipeline cascade: each Stage standalone vs E2E Quad F1
# ═══════════════════════════════════════════════════════════════════════════
def chart5_pipeline_cascade(m):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    stages   = m['pipeline']['stage_standalone']
    e2e_f1   = m['pipeline']['e2e_quad_f1']

    all_labels = [s['label'] for s in stages] + ['End-to-End\nQuad F1']
    all_vals   = [s['f1']    for s in stages] + [e2e_f1]
    colors     = [TRACK_A] * len(stages) + [BEST_C]

    bars = ax.bar(range(len(all_vals)), all_vals, color=colors,
                  edgecolor='white', linewidth=0.8, width=0.6, zorder=3)

    for bar, val in zip(bars, all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.010,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=9, color='#444444')

    ax.axhline(y=e2e_f1, color=BEST_C, linestyle='--',
               linewidth=1.0, alpha=0.5, zorder=2)
    ax.axvline(x=len(stages) - 0.45, color='#CCCCCC', linewidth=0.8, linestyle=':')

    ax.set_xticks(range(len(all_vals)))
    ax.set_xticklabels(all_labels, fontsize=9.5)
    ax.set_ylabel('F1', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title('Track A — 各 Stage 獨立指標 vs End-to-End Quad Partial F1',
                 fontsize=11, pad=10)
    ax.tick_params(axis='x', length=0)
    ax.legend(handles=[
        mpatches.Patch(color=TRACK_A, label='Stage standalone metric'),
        mpatches.Patch(color=BEST_C,  label='End-to-end Quad F1'),
    ], frameon=False, fontsize=9, loc='upper right')
    _despine_y(ax)
    fig.tight_layout()
    _save(fig, 'chart5_pipeline_cascade.png')


# ═══════════════════════════════════════════════════════════════════════════
# Chart 6 — FN Breakdown: 100% stacked horizontal bar
# ═══════════════════════════════════════════════════════════════════════════
def chart6_fn_breakdown(m):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    tc = m['track_comparison']
    fn_keys = ['both_missed', 'op_found_asp_missed', 'asp_found_op_missed']
    fn_labels = ['Both missed', 'Op found, asp missed', 'Asp found, op missed']
    seg_colors = [FP_C, FN_C, ITER_C]

    tracks = {
        'track_a': 'Track A\n(Rule-Based)',
        'track_b': 'Track B\n(LLM Zero-Shot)',
    }

    for row_i, (track_key, track_name) in enumerate(tracks.items()):
        fn_data = tc[track_key]['fn_breakdown']
        counts  = [fn_data[k] for k in fn_keys]
        total   = sum(counts)
        label   = f'{track_name}\nFN={total}'
        left    = 0.0
        for cnt, color in zip(counts, seg_colors):
            pct = cnt / total
            ax.barh(row_i, pct, left=left, color=color,
                    height=0.45, edgecolor='white', linewidth=1.0)
            if pct > 0.08:
                ax.text(left + pct / 2, row_i,
                        f'{pct:.0%}\n(n={cnt})',
                        ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
            left += pct

    track_row_labels = [
        f'Track A\n(Rule-Based)\nFN={sum(tc["track_a"]["fn_breakdown"][k] for k in fn_keys)}',
        f'Track B\n(LLM Zero-Shot)\nFN={sum(tc["track_b"]["fn_breakdown"][k] for k in fn_keys)}',
    ]
    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels(track_row_labels, fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Proportion of Pair-Partial FN', fontsize=10)
    ax.set_title('FN 來源分解：Track A vs Track B', fontsize=11, pad=10)

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(seg_colors, fn_labels)]
    ax.legend(handles=legend_patches, frameon=False, fontsize=8.5,
              loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    sns.despine(ax=ax, left=True, bottom=False)
    ax.grid(axis='x', color='#E0E0E0', linewidth=0.7, zorder=0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, 'chart6_fn_breakdown.png')


# ═══════════════════════════════════════════════════════════════════════════
# Chart 7 — Track A vs B: 4 core metrics grouped bar
# ═══════════════════════════════════════════════════════════════════════════
def chart7_track_comparison(m):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    tc = m['track_comparison']
    metric_keys    = ['aspect_partial_f1', 'opinion_partial_f1', 'pair_partial_f1', 'quad_partial_f1']
    metric_labels  = ['Aspect\nPartial F1', 'Opinion\nPartial F1', 'Pair\nPartial F1', 'Quad\nPartial F1']

    track_a = [tc['track_a'][k] for k in metric_keys]
    track_b = [tc['track_b'][k] for k in metric_keys]
    x = np.arange(len(metric_keys))
    w = 0.32

    bars_a = ax.bar(x - w / 2, track_a, w, label='Track A  (Rule-Based)',
                    color=TRACK_A, edgecolor='white', linewidth=0.8)
    bars_b = ax.bar(x + w / 2, track_b, w, label='Track B  (LLM Zero-Shot)',
                    color=TRACK_B, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars_a, track_a):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=8.5, color=TRACK_A)
    for bar, val in zip(bars_b, track_b):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=8.5, color=TRACK_B)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_ylim(0, 0.97)
    ax.set_title('Track A vs Track B — ABSA Quadruplet 評估對比', fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax.tick_params(axis='x', length=0)
    _despine_y(ax)
    fig.tight_layout()
    _save(fig, 'chart7_track_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════
# Chart 8 — Per-Category F1: Track A vs Track B (slope/dumbbell chart)
# ═══════════════════════════════════════════════════════════════════════════
def chart8_category_comparison(m):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    tc         = m['track_comparison']
    cat_f1_a   = tc['track_a']['category_f1']
    cat_f1_b   = tc['track_b']['category_f1']
    categories = list(cat_f1_a.keys())
    f1_a = [cat_f1_a[c] for c in categories]
    f1_b = [cat_f1_b[c] for c in categories]

    y_pos = np.arange(len(categories))

    for i, (cat, a, b) in enumerate(zip(categories, f1_a, f1_b)):
        color = TP_C if b > a else FP_C
        ax.plot([a, b], [i, i], color=color, linewidth=1.8,
                solid_capstyle='round', alpha=0.7, zorder=2)
        ax.scatter(a, i, s=70, color=TRACK_A, zorder=4,
                   edgecolors='white', linewidths=0.8)
        ax.scatter(b, i, s=70, color=TRACK_B, zorder=4,
                   marker='D', edgecolors='white', linewidths=0.8)
        close = abs(b - a) < 0.06
        a_dy  =  0.22 if close else 0
        b_dy  = -0.22 if close else 0
        ax.text(a - 0.012, i + a_dy, f'{a:.2f}', va='center', ha='right',
                fontsize=8, color=TRACK_A)
        ax.text(b + 0.012, i + b_dy, f'{b:.2f}', va='center', ha='left',
                fontsize=8, color=TRACK_B)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel('Category F1  (on Pair-TPs)', fontsize=11)
    ax.set_xlim(0.18, 1.14)
    ax.set_title('各面向 Category F1：Track A vs Track B', fontsize=12, pad=10)
    ax.tick_params(axis='y', length=0)

    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TRACK_A,
               markersize=8, label='Track A  (Rule-Based)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=TRACK_B,
               markersize=8, label='Track B  (LLM Zero-Shot)'),
        Line2D([0], [0], color=TP_C, linewidth=2, label='B > A'),
        Line2D([0], [0], color=FP_C, linewidth=2, label='B < A'),
    ]
    ax.legend(handles=legend_els, frameon=False, fontsize=9,
              loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4)
    _despine(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    _save(fig, 'chart8_category_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Loading metrics from: {METRICS_PATH}\n')
    m = _load_metrics()

    print('Generating academic-style charts (8 total)...\n')
    chart1_stage1_iteration(m)
    chart2_stage2_iteration(m)
    chart3_stage3_iteration(m)
    chart4_stage4_pairing(m)
    chart5_pipeline_cascade(m)
    chart6_fn_breakdown(m)
    chart7_track_comparison(m)
    chart8_category_comparison(m)
    print(f'\nDone. Output: {OUT_DIR}')
