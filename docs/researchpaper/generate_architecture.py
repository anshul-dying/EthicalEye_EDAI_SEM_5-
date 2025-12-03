"""
Generate architecture.png diagram for the research paper
Shows the system architecture of Ethical Eye extension
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Define colors
color_client = '#4A90E2'  # Blue
color_backend = '#50C878'  # Green
color_arrow = '#333333'  # Dark gray
color_text = '#2C3E50'  # Dark blue-gray

# Client Side (Left)
client_box = FancyBboxPatch((0.5, 1), 3.5, 5, 
                           boxstyle="round,pad=0.1", 
                           edgecolor=color_client, 
                           facecolor='#E8F4FD', 
                           linewidth=2)
ax.add_patch(client_box)
ax.text(2.25, 5.7, 'Chrome Extension (Client)', 
        ha='center', va='center', fontsize=14, fontweight='bold', color=color_text)

# Content Script
content_box = FancyBboxPatch((0.7, 4.2), 1.4, 0.6, 
                            boxstyle="round,pad=0.05", 
                            edgecolor=color_client, 
                            facecolor='white', 
                            linewidth=1.5)
ax.add_patch(content_box)
ax.text(1.4, 4.5, 'Content Script', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=color_text)
ax.text(1.4, 4.2, 'DOM Scanning', 
        ha='center', va='center', fontsize=8, color=color_text)

# Background Service
bg_box = FancyBboxPatch((2.1, 4.2), 1.4, 0.6, 
                       boxstyle="round,pad=0.05", 
                       edgecolor=color_client, 
                       facecolor='white', 
                       linewidth=1.5)
ax.add_patch(bg_box)
ax.text(2.8, 4.5, 'Background', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=color_text)
ax.text(2.8, 4.2, 'Service Worker', 
        ha='center', va='center', fontsize=8, color=color_text)

# Popup UI
popup_box = FancyBboxPatch((1.4, 2.5), 1.4, 0.6, 
                          boxstyle="round,pad=0.05", 
                          edgecolor=color_client, 
                          facecolor='white', 
                          linewidth=1.5)
ax.add_patch(popup_box)
ax.text(2.1, 2.8, 'Popup UI', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=color_text)
ax.text(2.1, 2.5, 'Status Display', 
        ha='center', va='center', fontsize=8, color=color_text)

# Backend Side (Right)
backend_box = FancyBboxPatch((6, 1), 3.5, 5, 
                            boxstyle="round,pad=0.1", 
                            edgecolor=color_backend, 
                            facecolor='#E8F8F0', 
                            linewidth=2)
ax.add_patch(backend_box)
ax.text(7.75, 5.7, 'Flask Backend (Server)', 
        ha='center', va='center', fontsize=14, fontweight='bold', color=color_text)

# Flask API
flask_box = FancyBboxPatch((6.2, 4.2), 1.4, 0.6, 
                          boxstyle="round,pad=0.05", 
                          edgecolor=color_backend, 
                          facecolor='white', 
                          linewidth=1.5)
ax.add_patch(flask_box)
ax.text(6.9, 4.5, 'Flask API', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=color_text)
ax.text(6.9, 4.2, 'REST Endpoints', 
        ha='center', va='center', fontsize=8, color=color_text)

# DistilBERT Model
model_box = FancyBboxPatch((7.6, 4.2), 1.4, 0.6, 
                          boxstyle="round,pad=0.05", 
                          edgecolor=color_backend, 
                          facecolor='white', 
                          linewidth=1.5)
ax.add_patch(model_box)
ax.text(8.3, 4.5, 'DistilBERT', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=color_text)
ax.text(8.3, 4.2, 'Classifier', 
        ha='center', va='center', fontsize=8, color=color_text)

# SHAP Explainer
shap_box = FancyBboxPatch((6.9, 2.5), 1.4, 0.6, 
                         boxstyle="round,pad=0.05", 
                         edgecolor=color_backend, 
                         facecolor='white', 
                         linewidth=1.5)
ax.add_patch(shap_box)
ax.text(7.6, 2.8, 'SHAP', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=color_text)
ax.text(7.6, 2.5, 'Explainer', 
        ha='center', va='center', fontsize=8, color=color_text)

# Arrows - Request flow
arrow1 = FancyArrowPatch((4, 4.5), (6, 4.5),
                         arrowstyle='->', mutation_scale=20,
                         color=color_arrow, linewidth=2)
ax.add_patch(arrow1)
ax.text(5, 4.8, 'HTTP POST', ha='center', fontsize=9, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_arrow, alpha=0.8))

# Arrows - Response flow
arrow2 = FancyArrowPatch((6, 2.8), (4, 2.8),
                         arrowstyle='->', mutation_scale=20,
                         color=color_arrow, linewidth=2)
ax.add_patch(arrow2)
ax.text(5, 2.5, 'JSON Response', ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_arrow, alpha=0.8))

# Internal arrows in backend
arrow3 = FancyArrowPatch((7.6, 4.2), (7.6, 3.1),
                        arrowstyle='->', mutation_scale=15,
                        color=color_backend, linewidth=1.5, linestyle='--')
ax.add_patch(arrow3)
ax.text(7.9, 3.65, 'SHAP', ha='left', fontsize=8, color=color_backend, rotation=90)

# Data flow labels
ax.text(2.25, 1.3, 'Text Segments', ha='center', fontsize=9, 
        style='italic', color=color_text)
ax.text(7.75, 1.3, 'Predictions + Explanations', ha='center', fontsize=9,
        style='italic', color=color_text)

# Title
ax.text(5, 6.5, 'Ethical Eye System Architecture', 
        ha='center', va='center', fontsize=16, fontweight='bold', color=color_text)

plt.tight_layout()
plt.savefig('architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved as architecture.png")
plt.close()

