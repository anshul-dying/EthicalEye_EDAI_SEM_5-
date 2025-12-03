"""
Generate shap_example.png diagram for the research paper
Shows SHAP explanation overlay on a mock e-commerce webpage
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Background color (webpage-like)
page_bg = Rectangle((0, 0), 10, 6, facecolor='#F5F5F5', edgecolor='none')
ax.add_patch(page_bg)

# Title
ax.text(5, 5.5, 'Example E-Commerce Product Page', 
        ha='center', fontsize=14, fontweight='bold', color='#2C3E50')

# Product section
product_box = FancyBboxPatch((1, 3.5), 8, 1.2, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='#CCCCCC', 
                            facecolor='white', 
                            linewidth=1)
ax.add_patch(product_box)

# Product title
ax.text(1.2, 4.4, 'Premium Wireless Headphones', 
        ha='left', fontsize=12, fontweight='bold', color='#2C3E50')

# Price
ax.text(1.2, 4.0, '$199.99', 
        ha='left', fontsize=14, fontweight='bold', color='#E74C3C')

# Dark pattern text with SHAP highlights
pattern_text_y = 3.7

# Normal text
ax.text(1.2, pattern_text_y, 'Hurry! Only ', 
        ha='left', fontsize=11, color='#2C3E50')

# Highlighted word 1: "Hurry" (high positive SHAP - red)
highlight1 = Rectangle((1.2, pattern_text_y - 0.15), 0.5, 0.25, 
                      facecolor='#FF6B6B', alpha=0.4, edgecolor='#FF6B6B', linewidth=1.5)
ax.add_patch(highlight1)
ax.text(1.45, pattern_text_y, 'Hurry', 
        ha='center', fontsize=11, fontweight='bold', color='#C92A2A')

# Normal text
ax.text(1.7, pattern_text_y, '! Only ', 
        ha='left', fontsize=11, color='#2C3E50')

# Highlighted word 2: "Only" (high positive SHAP - red)
highlight2 = Rectangle((1.95, pattern_text_y - 0.15), 0.4, 0.25, 
                      facecolor='#FF6B6B', alpha=0.4, edgecolor='#FF6B6B', linewidth=1.5)
ax.add_patch(highlight2)
ax.text(2.15, pattern_text_y, 'Only', 
        ha='center', fontsize=11, fontweight='bold', color='#C92A2A')

# Normal text
ax.text(2.35, pattern_text_y, ' 2 left in stock! ', 
        ha='left', fontsize=11, color='#2C3E50')

# Highlighted word 3: "2 left" (high positive SHAP - red)
highlight3 = Rectangle((2.65, pattern_text_y - 0.15), 0.5, 0.25, 
                      facecolor='#FF6B6B', alpha=0.4, edgecolor='#FF6B6B', linewidth=1.5)
ax.add_patch(highlight3)
ax.text(2.9, pattern_text_y, '2 left', 
        ha='center', fontsize=11, fontweight='bold', color='#C92A2A')

# Normal text
ax.text(3.15, pattern_text_y, 'in stock!', 
        ha='left', fontsize=11, color='#2C3E50')

# SHAP Tooltip (appears on hover)
tooltip_box = FancyBboxPatch((5.5, 2.5), 3.5, 1.2, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='#4A90E2', 
                             facecolor='white', 
                             linewidth=2,
                             linestyle='--')
ax.add_patch(tooltip_box)

ax.text(7.25, 3.4, 'SHAP Explanation', 
        ha='center', fontsize=11, fontweight='bold', color='#2C3E50')
ax.text(7.25, 3.1, 'Pattern: Urgency', 
        ha='center', fontsize=10, color='#E74C3C', fontweight='bold')
ax.text(7.25, 2.9, 'Confidence: 94%', 
        ha='center', fontsize=9, color='#2C3E50')
ax.text(7.25, 2.7, 'Key words: "Hurry", "Only", "2 left"', 
        ha='center', fontsize=9, color='#666666', style='italic')

# Arrow from highlighted text to tooltip
arrow = mpatches.FancyArrowPatch((3.5, 3.6), (5.5, 3.1),
                                 arrowstyle='->', mutation_scale=20,
                                 color='#4A90E2', linewidth=2, linestyle='--')
ax.add_patch(arrow)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#FF6B6B', alpha=0.4, edgecolor='#FF6B6B', 
                   label='High SHAP Value (Positive)'),
    mpatches.Patch(facecolor='white', edgecolor='#4A90E2', 
                   linestyle='--', label='SHAP Tooltip')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
         framealpha=0.9, edgecolor='#CCCCCC')

# Pattern detection badge
badge = FancyBboxPatch((1, 1.8), 2, 0.6, 
                      boxstyle="round,pad=0.1", 
                      edgecolor='#E74C3C', 
                      facecolor='#FFE5E5', 
                      linewidth=2)
ax.add_patch(badge)
ax.text(2, 2.1, '⚠ Dark Pattern Detected', 
        ha='center', fontsize=10, fontweight='bold', color='#C92A2A')

# Additional context text
ax.text(1.2, 1.2, 'This text creates false urgency to rush purchase decisions.', 
        ha='left', fontsize=9, color='#666666', style='italic')

# Footer note
ax.text(5, 0.3, 'SHAP highlights show which words most influenced the "Urgency" classification', 
        ha='center', fontsize=8, color='#999999', style='italic')

plt.tight_layout()
plt.savefig('shap_example.png', dpi=300, bbox_inches='tight', facecolor='white')
print("SHAP example diagram saved as shap_example.png")
plt.close()

