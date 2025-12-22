#!/usr/bin/env python3
"""
Create Figure 1 for the write-up: showing experimental setup and core result.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with two panels
fig = plt.figure(figsize=(14, 5.5))

# Create a gridspec for custom layout
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.22)

# ============================================================================
# LEFT PANEL: Experimental Setup Schematic
# ============================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Experimental Setup', fontsize=14, fontweight='bold', pad=10)

# Colors
problem_color = '#FFF8E1'
filler_color = '#F3E5F5'
repeat_color = '#E8F5E9'
answer_color = '#FFEBEE'
arrow_color = '#555555'

def draw_prompt_box(ax, x, y, width, height, lines, facecolor, fontsize=9.5,
                    highlight_lines=None, highlight_color=None):
    """Draw a prompt box with multiple lines, optionally highlighting specific lines."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=facecolor, edgecolor='#BDBDBD',
                         linewidth=1.5)
    ax.add_patch(box)

    # Calculate line spacing
    total_text_height = height * 0.8
    line_height = total_text_height / max(len(lines), 1)
    start_y = y + height - (height - total_text_height) / 2 - line_height * 0.5

    for i, line in enumerate(lines):
        line_y = start_y - i * line_height
        is_highlighted = highlight_lines and i in highlight_lines

        if is_highlighted and highlight_color:
            # Draw highlight background
            hl_box = FancyBboxPatch((x + 0.08, line_y - line_height*0.4),
                                    width - 0.16, line_height*0.8,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    facecolor=highlight_color, edgecolor='none', alpha=0.7)
            ax.add_patch(hl_box)

        ax.text(x + 0.15, line_y, line, fontsize=fontsize, fontfamily='monospace',
                va='center', ha='left')

# Row positions (reduced vertical spacing, moved up)
y_baseline = 7.2
y_filler = 4.4
y_repeat = 1.6

# Column positions
x_prompt = 1.0
x_arrow_start = 5.7
x_arrow_end = 6.5
x_answer = 6.7

box_width = 4.5
box_height = 1.8

# The example problem (real case where baseline fails but filler/repeats succeed)
problem_line1 = 'How many 8-digit'
problem_line2 = 'integers become 1/9 as big if'
problem_line3 = 'the first digit is removed?'

# ============== BASELINE ROW ==============
# Label above the box
ax1.text(x_prompt + box_width/2, y_baseline + box_height + 0.2, 'Baseline (No CoT)',
         fontsize=11, fontweight='bold', va='bottom', ha='center')

draw_prompt_box(ax1, x_prompt, y_baseline, box_width, box_height,
                [f'Problem: {problem_line1}', problem_line2, problem_line3],
                problem_color, fontsize=9)

# Arrow
ax1.annotate('', xy=(x_arrow_end, y_baseline + 0.9), xytext=(x_arrow_start, y_baseline + 0.9),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
ax1.text((x_arrow_start + x_arrow_end)/2, y_baseline + 1.25, 'LLM', fontsize=9,
         ha='center', style='italic', color='#666')

# Answer box - WRONG answer for baseline
ans_box = FancyBboxPatch((x_answer, y_baseline + 0.4), 2.3, 1.0,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor='#FFCDD2', edgecolor='#E53935', linewidth=1.5)
ax1.add_patch(ans_box)
ax1.text(x_answer + 1.15, y_baseline + 0.9, 'Answer: 8', fontsize=10,
         fontfamily='monospace', ha='center', va='center', fontweight='bold', color='#C62828')

# ============== FILLER ROW ==============
ax1.text(x_prompt + box_width/2, y_filler + box_height + 0.2, 'Filler Tokens',
         fontsize=11, fontweight='bold', va='bottom', ha='center')

draw_prompt_box(ax1, x_prompt, y_filler, box_width, box_height,
                [f'Problem: {problem_line1}...', '', 'Filler: 1 2 3 ... 300'],
                problem_color, fontsize=9, highlight_lines=[2], highlight_color=filler_color)

ax1.annotate('', xy=(x_arrow_end, y_filler + 0.9), xytext=(x_arrow_start, y_filler + 0.9),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
ax1.text((x_arrow_start + x_arrow_end)/2, y_filler + 1.25, 'LLM', fontsize=9,
         ha='center', style='italic', color='#666')

# Answer box - CORRECT answer for filler
ans_box2 = FancyBboxPatch((x_answer, y_filler + 0.4), 2.3, 1.0,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#C8E6C9', edgecolor='#43A047', linewidth=1.5)
ax1.add_patch(ans_box2)
ax1.text(x_answer + 1.15, y_filler + 0.9, 'Answer: 7', fontsize=10,
         fontfamily='monospace', ha='center', va='center', fontweight='bold', color='#2E7D32')

# ============== REPEAT ROW ==============
ax1.text(x_prompt + box_width/2, y_repeat + box_height + 0.2, 'Repeats',
         fontsize=11, fontweight='bold', va='bottom', ha='center')

draw_prompt_box(ax1, x_prompt, y_repeat, box_width, box_height,
                [f'Problem: {problem_line1}...', 'Problem (repeat 2): ...', 'Problem (repeat 3): ...'],
                problem_color, fontsize=9, highlight_lines=[1, 2], highlight_color=repeat_color)

ax1.annotate('', xy=(x_arrow_end, y_repeat + 0.9), xytext=(x_arrow_start, y_repeat + 0.9),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
ax1.text((x_arrow_start + x_arrow_end)/2, y_repeat + 1.25, 'LLM', fontsize=9,
         ha='center', style='italic', color='#666')

# Answer box - CORRECT answer for repeats
ans_box3 = FancyBboxPatch((x_answer, y_repeat + 0.4), 2.3, 1.0,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#C8E6C9', edgecolor='#43A047', linewidth=1.5)
ax1.add_patch(ans_box3)
ax1.text(x_answer + 1.15, y_repeat + 0.9, 'Answer: 7', fontsize=10,
         fontfamily='monospace', ha='center', va='center', fontweight='bold', color='#2E7D32')

# Legend at top - positioned to avoid overlap with "Baseline" label
filler_patch = mpatches.Patch(facecolor=filler_color, edgecolor='#9C27B0',
                               linewidth=1.5, label='Filler tokens (unrelated text)')
repeat_patch = mpatches.Patch(facecolor=repeat_color, edgecolor='#4CAF50',
                               linewidth=1.5, label='Problem repetitions')
ax1.legend(handles=[filler_patch, repeat_patch], loc='upper center',
           ncol=2, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.1))

# ============================================================================
# RIGHT PANEL: Bar Chart of Core Results
# ============================================================================
ax2 = fig.add_subplot(gs[1])

# Data from Easy-Comp-Math dataset (Opus 4.5)
conditions = ['Baseline\n(No CoT)', 'Filler tokens\n(f=300)', 'Repeats\n(r=5)']
accuracies = [45.2, 51.0, 51.2]
n = 907

# Calculate 95% confidence intervals
errors = [1.96 * np.sqrt(acc/100 * (1-acc/100) / n) * 100 for acc in accuracies]

# Colors matching the left panel
colors = ['#BDBDBD', '#CE93D8', '#81C784']
edge_colors = ['#757575', '#7B1FA2', '#388E3C']

# Create bar chart (no error bars)
bars = ax2.bar(conditions, accuracies, color=colors, edgecolor=edge_colors, linewidth=2.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

# Add horizontal line at baseline for reference
ax2.axhline(y=45.2, color='#9E9E9E', linestyle='--', linewidth=1.5, alpha=0.6)

# Add significance annotations
def add_significance_bracket(ax, x1, x2, y, text, h=1.5):
    """Add a significance bracket between two bars."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', linewidth=1.5)
    ax.text((x1+x2)/2, y+h+0.3, text, ha='center', va='bottom', fontsize=10, fontweight='bold')

# Significance from baseline to filler
add_significance_bracket(ax2, 0, 1, 57, 'p < 0.001', h=1.2)
# Significance from baseline to repeats
add_significance_bracket(ax2, 0, 2, 62, 'p < 0.001', h=1.2)

# Customize axes
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Opus 4.5 on Easy-Comp-Math', fontsize=14, fontweight='bold', pad=10)
ax2.set_ylim(0, 72)
ax2.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
ax2.set_axisbelow(True)

# Remove top and right spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add main figure title
fig.suptitle('Filler tokens and problem repeats improve no-CoT math performance',
             fontsize=14, fontweight='bold', y=1.02)

plt.savefig('figure1.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figure1.pdf', bbox_inches='tight', facecolor='white')
print('Saved figure1.png and figure1.pdf')
plt.close()
