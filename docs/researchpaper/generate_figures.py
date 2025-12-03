"""
Generate architecture.png and shap_example.png
Fixed: Text visibility (zorder), Alignment issues, and Overlapping text
"""

import sys
import os

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
    import numpy as np
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Error importing matplotlib: {e}")
    sys.exit(1)

def draw_text_box(ax, x, y, width, height, title, subtitle, color, bg_color='white'):
    """Helper to draw a standardized component box"""
    # Box (zorder=2)
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.1",
                        edgecolor=color,
                        facecolor=bg_color,
                        linewidth=1.5,
                        zorder=2)
    ax.add_patch(box)
    
    # Text (zorder=10 to ensure it sits ON TOP of everything)
    cx = x + width / 2
    cy = y + height / 2
    ax.text(cx, cy + 0.15, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#2C3E50', zorder=10)
    ax.text(cx, cy - 0.15, subtitle, ha='center', va='center', fontsize=8, color='#555555', zorder=10)

def generate_architecture():
    """Generate architecture.png with fixed z-indexing"""
    print("\nGenerating architecture.png...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(11, 7))
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 7)
        ax.axis('off')

        # Colors
        c_client = '#4A90E2'
        c_backend = '#27AE60'
        c_arrow = '#34495E'

        # --- CONTAINERS (zorder=1) ---
        # Client Container
        client_bg = FancyBboxPatch((0.5, 1), 3.5, 5, boxstyle="round,pad=0.2", 
                                  edgecolor=c_client, facecolor='#F0F7FF', linewidth=2, zorder=1)
        ax.add_patch(client_bg)
        ax.text(2.25, 6.2, 'Chrome Extension (Client)', ha='center', fontsize=13, fontweight='bold', color='#2C3E50', zorder=10)

        # Backend Container
        server_bg = FancyBboxPatch((7, 1), 3.5, 5, boxstyle="round,pad=0.2", 
                                  edgecolor=c_backend, facecolor='#F0FFF4', linewidth=2, zorder=1)
        ax.add_patch(server_bg)
        ax.text(8.75, 6.2, 'Flask Server (Backend)', ha='center', fontsize=13, fontweight='bold', color='#2C3E50', zorder=10)

        # --- COMPONENTS (zorder=2) ---
        draw_text_box(ax, 1.0, 4.2, 2.5, 0.8, "Content Script", "DOM Scraper", c_client)
        draw_text_box(ax, 1.0, 3.0, 2.5, 0.8, "Background Service", "Event Handler", c_client)
        draw_text_box(ax, 1.0, 1.8, 2.5, 0.8, "Popup UI", "React / HTML", c_client)

        draw_text_box(ax, 7.5, 4.2, 2.5, 0.8, "Flask API", "Routes / Endpoints", c_backend)
        draw_text_box(ax, 7.5, 3.0, 2.5, 0.8, "DistilBERT", "Classification Model", c_backend)
        draw_text_box(ax, 7.5, 1.8, 2.5, 0.8, "SHAP Explainer", "Feature Importance", c_backend)

        # --- ARROWS (zorder=3) ---
        # Request Arrow
        a1 = FancyArrowPatch((4.2, 4.6), (6.8, 4.6), arrowstyle='->', mutation_scale=20, color=c_arrow, lw=2, zorder=3)
        ax.add_patch(a1)
        # Label with opaque background
        ax.text(5.5, 4.9, "HTTP POST\n(Text Data)", ha='center', fontsize=8, zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

        # Response Arrow
        a2 = FancyArrowPatch((6.8, 2.2), (4.2, 2.2), arrowstyle='->', mutation_scale=20, color=c_arrow, lw=2, zorder=3)
        ax.add_patch(a2)
        ax.text(5.5, 1.8, "JSON Response\n(Score + SHAP)", ha='center', fontsize=8, zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

        # Internal Arrows (Client)
        ax.arrow(2.25, 4.2, 0, -0.4, head_width=0.1, color=c_client, length_includes_head=True, zorder=3)
        ax.arrow(2.25, 3.0, 0, -0.4, head_width=0.1, color=c_client, length_includes_head=True, zorder=3)

        # Internal Arrows (Backend)
        ax.arrow(8.75, 4.2, 0, -0.4, head_width=0.1, color=c_backend, length_includes_head=True, zorder=3)
        ax.arrow(8.75, 3.0, 0, -0.4, head_width=0.1, color=c_backend, length_includes_head=True, zorder=3)

        plt.tight_layout()
        plt.savefig('architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ architecture.png generated successfully")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def generate_shap_example():
    """Generate shap_example.png with robust sequential text drawing"""
    print("\nGenerating shap_example.png...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 7)
        ax.axis('off')
        
        # Background
        ax.add_patch(Rectangle((0, 0), 12, 7, color='#F8F9FA', zorder=0))

        # Product Card
        card = FancyBboxPatch((1, 3.5), 10, 2.5, boxstyle="round,pad=0.1", fc='white', ec='#DEE2E6', lw=1, zorder=1)
        ax.add_patch(card)

        # Static Text
        ax.text(1.3, 5.5, 'Wireless Headphones', fontsize=16, fontweight='bold', color='#212529', zorder=10)
        ax.text(1.3, 5.0, '$199.99', fontsize=18, fontweight='bold', color='#E74C3C', zorder=10)

        # --- ROBUST HIGHLIGHTING LOGIC ---
        # Instead of overwriting text, we draw it sequentially (word by word)
        # This prevents alignment mismatches and overlapping.
        
        # Text to render
        segments = [
            ("Hurry! ", True),
            ("Only ", True),
            ("2 left ", True),
            ("in stock – ", False),
            ("order soon!", True)
        ]

        # Starting position
        cursor_x = 1.3
        cursor_y = 4.3
        
        # Force a draw to initialize the renderer (Crucial for getting text size)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        trans = ax.transData

        for text_str, is_highlight in segments:
            # 1. Determine style
            color = '#C92A2A' if is_highlight else '#212529'
            weight = 'bold' if is_highlight else 'normal'
            bg_color = '#FF6B6B' if is_highlight else None
            
            # 2. Draw text temporarily to get size
            t = ax.text(cursor_x, cursor_y, text_str, fontsize=14, fontweight=weight, color=color, zorder=10)
            
            # 3. Calculate bounding box
            bbox = t.get_window_extent(renderer=renderer)
            bbox_data = bbox.transformed(trans.inverted())
            width_data = bbox_data.width
            height_data = bbox_data.height
            
            # 4. If highlighted, add the red background rectangle BEHIND the text
            if bg_color:
                rect = Rectangle((cursor_x, cursor_y - 0.1), width_data, height_data + 0.2, 
                                 facecolor=bg_color, alpha=0.3, edgecolor='none', zorder=5)
                ax.add_patch(rect)
            
            # 5. Move cursor for next word
            cursor_x += width_data

        # --- SHAP EXPLANATION UI ---
        # Tooltip box
        tip_x, tip_y = 6.5, 3.8
        tooltip = FancyBboxPatch((tip_x, tip_y), 4.5, 1.5, boxstyle="round,pad=0.2", 
                                fc='white', ec='#4A90E2', lw=2, zorder=15)
        ax.add_patch(tooltip)

        # Tooltip text
        ax.text(tip_x + 2.25, tip_y + 1.1, "SHAP Model Explanation", ha='center', fontweight='bold', color='#4A90E2', zorder=20)
        ax.text(tip_x + 2.25, tip_y + 0.7, "Pattern: Urgency Scarcity", ha='center', fontsize=11, color='#C92A2A', zorder=20)
        ax.text(tip_x + 2.25, tip_y + 0.3, "Contributing words highlighted in red", ha='center', fontsize=9, style='italic', color='#555', zorder=20)

        # Connection Arrow (Curved)
        arrow = FancyArrowPatch((4.5, 4.3), (6.4, 4.5), arrowstyle='->', 
                               connectionstyle="arc3,rad=-0.2", color='#4A90E2', lw=2, zorder=15)
        ax.add_patch(arrow)

        plt.tight_layout()
        plt.savefig('shap_example.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
        plt.close()
        print("✓ shap_example.png generated successfully")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("Generating Fixed Figures...")
    
    s1 = generate_architecture()
    s2 = generate_shap_example()
    
    if s1 and s2:
        print("\n✓ Success! Check the directory for new PNG files.")
    else:
        print("\n✗ Failed.")