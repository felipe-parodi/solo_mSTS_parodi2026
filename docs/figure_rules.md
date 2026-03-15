# Figure Aesthetics Guidelines

This document outlines the style guidelines for generating figures for the FreeMat LFP analysis project, aiming for clarity, consistency, and publication-readiness (e.g., Nature standards).

## General Principles

*   **Clarity:** Figures should be easy to understand at a glance. Avoid clutter.
*   **Consistency:** Use consistent fonts, colors, line styles, and layouts across all figures.
*   **Information:** Ensure axes are clearly labeled with units. Titles should be informative but concise.
*   **Simplicity:** Minimize non-data ink. Avoid unnecessary borders, backgrounds, or grid lines unless they aid interpretation.

## Specific Rules

1.  **Fonts:**
    *   Use a sans-serif font (e.g., Arial, Helvetica) for all text elements.
    *   **Axis Labels & Titles:** Use a larger font size (e.g., 12-14pt) for readability.
    *   **Tick Labels:** Use a slightly smaller font size (e.g., 10-12pt).
    *   **Legends:** Use a similar size to tick labels.

2.  **Axes:**
    *   **Minimalism:** Typically, only show the bottom (x) and left (y) axes lines (despine top and right).
    *   **Ticks:** Ticks should point outwards. Use a reasonable number of ticks – not too dense, not too sparse.
    *   **Labels:** Clearly label *all* axes with the quantity and its units (e.g., "Time (s)", "Frequency (Hz)", "Power (dB)").

3.  **Colors:**
    *   **Scheme:** Define a consistent color palette for different conditions, channels, or groups. Consider using palettes from libraries like `matplotlib.colormaps` (e.g., 'viridis', 'plasma', 'tab10', 'tab20') or established scientific palettes.
    *   **Color Blindness:** Choose palettes that are distinguishable by individuals with common forms of color blindness.
    *   **Background:** Use a white background.

4.  **Lines & Markers:**
    *   **Line Widths:** Use appropriate line widths (e.g., 1-2pt) for visibility.
    *   **Markers:** Use markers only if necessary to distinguish overlapping lines or indicate specific data points. Keep them simple (e.g., circles, squares).

5.  **Layout:**
    *   **Padding:** Ensure adequate padding around plot elements (labels, titles) so they don't overlap.
    *   **Multi-panel Figures:** Align axes across panels where appropriate. Use clear labels (A, B, C...) for subplots.

6.  **Saving:**
    *   **Formats:** Save figures in vector formats (SVG, PDF, EPS) for scalability and lossless quality, and high-resolution raster formats (PNG, TIFF at >=300 DPI) for submissions or presentations.
    *   **Transparency:** Use transparent backgrounds when saving if figures will be overlaid.

7.  **Multi-Panel Composite Sizing:**
    *   **Box aspect control:** Use `ax.set_box_aspect(ratio)` on every panel to enforce
        consistent sizing. Use `1.0` for square heatmaps/matrices, `0.75` for line plots,
        scatter plots, and histograms.
    *   **Uniform column count:** Use the same number of subgridspec columns in every row,
        even if the last column is left empty. This ensures identical cell widths across rows.
    *   **Title alignment:** When panels in a row have different aspect ratios, titles sit at
        different heights. Use a post-draw helper (`_align_row_titles`) to compute the highest
        axes-top per row and shift all titles to that level.
    *   **Square heatmaps:** Use `aspect="equal"` inside `imshow()` for matrix heatmaps.
        Combined with `set_box_aspect(1.0)`, this ensures all matrices render at the same size.
    *   **Tick density:** Skip every other tick label when axes have >8 labels (e.g., 12
        syllable names). Use `MaxNLocator(nbins=4-5)` for numeric axes.

*(This is a living document - update as needed)*
