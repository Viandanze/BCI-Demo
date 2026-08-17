import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Style ──
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 180
plt.rcParams['savefig.dpi'] = 180
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE   = '#4A90D9'
ORANGE = '#E8833A'
GREY   = '#B0BEC5'
RED    = '#E74C3C'

# ═══════════════════════════════════════════════════════
# Chart 1: Model Accuracy Comparison (Horizontal)
# ═══════════════════════════════════════════════════════
def chart_model_comparison():
    models = [
        'Conformer', 'TCN', 'EEGNet Baseline',
        'EEGNet + HPO',
        'Ensemble-Weighted', 'Ensemble-Voting', 'Ensemble-Stacking'
    ]
    accs = [38.43, 40.35, 45.83, 54.32, 80.00, 79.29, 82.14]
    colors = [BLUE]*4 + [ORANGE]*3

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, accs, color=colors, height=0.6, edgecolor='white', linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=11, fontweight='bold', color='#333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('BCI Model Accuracy Comparison', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlim(0, 95)
    ax.axvline(x=33.33, color=GREY, linestyle='--', linewidth=1, alpha=0.7)
    ax.text(34.5, -0.6, 'Random Baseline 33.3%', ha='left', fontsize=9, color='#999')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BLUE, label='DL Single Model'),
                       Patch(facecolor=ORANGE, label='Ensemble Method')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
              framealpha=0.9, edgecolor='#ddd')

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'model_comparison_en.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════
# Chart 2: Ensemble Strategy Comparison
# ═══════════════════════════════════════════════════════
def chart_ensemble_comparison():
    strategies = ['Voting\n(Soft)', 'Stacking\n(Tangent Space)', 'Weighted\n(Weighted Vote)']
    accs = [79.29, 82.14, 80.00]
    kappas = [0.472, 0.537, 0.477]
    colors = ['#5DADE2', RED, '#F39C12']

    fig, ax1 = plt.subplots(figsize=(8, 5.8))
    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accs, width, color=colors, edgecolor='white',
                     linewidth=0.8, label='Accuracy (%)', zorder=3)
    ax2 = ax1.twinx()
    kappa_pct = [k*100 for k in kappas]
    bars2 = ax2.bar(x + width/2, kappa_pct, width,
                     color=[c+'99' for c in colors], edgecolor='white',
                     linewidth=0.8, label='Kappa (×100)', zorder=3)

    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333')
    for bar, kappa in zip(bars2, kappas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f'{kappa:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#666')

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylabel('Kappa (×100)', fontsize=12, color='#666')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=12)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 65)
    ax1.set_title('Ensemble Strategy Performance Comparison', fontsize=15, fontweight='bold', pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11,
               framealpha=0.9, edgecolor='#ddd')

    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'ensemble_comparison_en.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════
# Chart 3: EEGNet Tuning Strategies
# ═══════════════════════════════════════════════════════
def chart_tuning_strategies():
    strategies = ['Baseline', 'Mixup', 'Time Shift', 'HPO', 'SE Attention']
    accs = [45.83, 49.58, 49.50, 54.32, 45.75]
    kappas = [0.227, 0.157, 0.196, 0.243, 0.250]
    f1s = [0.406, 0.394, 0.456, 0.0, 0.468]

    x = np.arange(len(strategies))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.8))

    acc_colors = [GREY, '#5DADE2', '#5DADE2', RED, '#5DADE2']
    bars1 = ax.bar(x - width, accs, width, color=acc_colors, edgecolor='white',
                    linewidth=0.5, label='Accuracy (%)', zorder=3)
    bars2 = ax.bar(x, [k*100 for k in kappas], width, color='#8E44AD',
                    edgecolor='white', linewidth=0.5, alpha=0.7, label='Kappa (×100)', zorder=3)
    bars3 = ax.bar(x + width, [f*100 for f in f1s], width, color='#27AE60',
                    edgecolor='white', linewidth=0.5, alpha=0.7, label='F1-macro (×100)', zorder=3)

    for bar, acc in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#333')

    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylim(0, 68)
    ax.set_title('EEGNet Tuning Strategy Comparison', fontsize=15, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#ddd')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    ax.annotate('↑ +8.5pp', xy=(3, 54.32), xytext=(3.6, 62),
                fontsize=11, color=RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'tuning_strategies_en.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════
# Chart 4: Single Model vs Ensemble (Radar)
# ═══════════════════════════════════════════════════════
def chart_single_vs_ensemble():
    categories = ['Accuracy', 'Kappa', 'F1-macro', 'Generalization', 'Train Efficiency']
    N = len(categories)

    single_scores = [54.32, 24.3, 45.6, 40, 75]
    ensemble_scores = [82.14, 53.7, 75.0, 70, 25]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    single_scores_r = single_scores + [single_scores[0]]
    ensemble_scores_r = ensemble_scores + [ensemble_scores[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.fill(angles, single_scores_r, color=BLUE, alpha=0.15)
    ax.plot(angles, single_scores_r, color=BLUE, linewidth=2.2, marker='o', markersize=5,
            label='EEGNet (HPO) — Best Single')

    ax.fill(angles, ensemble_scores_r, color=ORANGE, alpha=0.15)
    ax.plot(angles, ensemble_scores_r, color=ORANGE, linewidth=2.2, marker='s', markersize=5,
            label='Stacking Ensemble — Best Ensemble')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9, color='#999')
    ax.set_rlabel_position(30)

    for i, (angle, s, e) in enumerate(zip(angles[:-1], single_scores, ensemble_scores)):
        offset_r = 9
        ax.text(angle, min(s + offset_r, 98), f'{s:.1f}', ha='center', va='center',
                fontsize=9, color=BLUE, fontweight='bold')
        ax.text(angle, min(e + offset_r, 98), f'{e:.1f}', ha='center', va='center',
                fontsize=9, color=ORANGE, fontweight='bold')

    ax.text(np.pi/2, 30, 'Stacking vs Single\nAccuracy +27.8pp\nKappa +29.4pp',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=RED, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', edgecolor=RED, alpha=0.8))

    ax.set_title('Single Model vs Ensemble — Comprehensive Comparison', fontsize=14, fontweight='bold', pad=22)
    ax.legend(loc='lower right', bbox_to_anchor=(1.32, -0.05), fontsize=11,
              framealpha=0.9, edgecolor='#ddd')

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'single_vs_ensemble_en.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ── Run ──
chart_model_comparison()
chart_ensemble_comparison()
chart_tuning_strategies()
chart_single_vs_ensemble()
print('\nAll 4 English charts generated successfully!')
