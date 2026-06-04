"""
Generate an SVG flowchart (記述子スクリーニングフローチャート):
  Stage 1: Full descriptor pool (34 descriptors × 8 level-categories)
  Stage 2: Class-specific LOO screening (一つ抜き交差検証, 穷举法)
  Stage 3: 14 retained descriptors grouped by 4 physical categories
  Stage 4: Final R² for 4 classes × 4 Arrhenius parameters

Output: analysis_figures/descriptor_screening_flowchart.svg
"""
import os

# ============================================================================
# Color palette (by physical category)
# ============================================================================
CAT_COLOR = {
    'Electronic': '#e74c3c',
    'Steric':     '#3498db',
    'Bonding':    '#f39c12',
    'Solvation':  '#2ecc71',
    'Rejected':   '#95a5a6',
}

# ============================================================================
# Stage 1: Full pool (grouped by Level × Category)
# ============================================================================
POOL = [
    # (level_label,             category,     [descriptors], accepted?)
    ('L1 empirical',            'Electronic', ['σ_Hammett'],                                            True),
    ('L1 geometry',             'Steric',     ['B1', 'B5', 'L', '%V_bur', 'mol_vol'],                   True),
    ('L1 xTB',                  'Electronic', ['q(C)', 'HOMO', 'LUMO', 'η', 'fukui f⁻', 'q(Li)'],      True),  # LUMO, q(Li) partially used
    ('L1 xTB',                  'Bonding',    ['d(Li-C)', 'BDE', 'Wiberg'],                            True),
    ('L1 xTB',                  'Solvation',  ['Gsolv', 'dipole'],                                     True),
    ('L2 HF/def2-SVP',          'Electronic', ['Mulliken', 'Löwdin', 'HOMO', 'LUMO', 'gap'],           False),
    ('L3 M06-2X/def2-SVP',      'Electronic', ['Mulliken', 'Löwdin', 'HOMO', 'LUMO', 'gap'],           False),
    ('L4 ADCH',                 'Electronic', ['ADCH-Li', 'ADCH-C'],                                   False),
    ('L4 QTAIM',                'Bonding',    ['ρ(BCP)', 'G', 'V', 'H', '∇²ρ', 'ellip.', '|V|/G'],    False),
    ('L5 M06-2X+PCM',           'Solvation',  ['Mull+PCM'],                                            False),
]

# ============================================================================
# Stage 3: 14 retained descriptors
# ============================================================================
RETAINED = [
    ('Electronic', ['q(C_ipso)', 'HOMO', 'η (gap)', 'fukui f⁻', 'σ_Hammett', 'BDE(Li-C)']),
    ('Steric',     ['Sterimol B1', 'Sterimol B5', 'Sterimol L', '%V_bur', 'mol_volume']),
    ('Bonding',    ['d(Li-C)']),
    ('Solvation',  ['Gsolv(THF)', 'dipole']),
]

# ============================================================================
# Stage 4: R² results
# ============================================================================
R2_TABLE = {
    # class: {param: r²}
    'p-ArLi':     {'Ea_f': None,  'Ea_d': 0.642, 'lnA_f': 0.845, 'lnA_d': 0.982},
    'm-ArLi':     {'Ea_f': 0.981, 'Ea_d': 0.985, 'lnA_f': 0.998, 'lnA_d': 0.998},
    'o-ArLi':     {'Ea_f': 0.892, 'Ea_d': 0.987, 'lnA_f': 0.971, 'lnA_d': 0.944},
    'oxiranylLi': {'Ea_f': 0.983, 'Ea_d': 0.998, 'lnA_f': 0.974, 'lnA_d': 0.995},
}
PARAMS = ['Ea_f', 'Ea_d', 'lnA_f', 'lnA_d']
CLASSES = ['p-ArLi', 'm-ArLi', 'o-ArLi', 'oxiranylLi']


def r2_to_color(r2):
    """Map R² value to a blue-to-red gradient color (low to high)."""
    if r2 is None:
        return '#d5d8dc'  # gray for N/A
    # 0.6 → light pink, 1.0 → dark red
    r2_norm = max(0, min(1, (r2 - 0.6) / 0.4))
    # interpolate from light red to dark red
    r = int(255 - 70 * r2_norm)
    g = int(230 - 180 * r2_norm)
    b = int(220 - 180 * r2_norm)
    return f'#{r:02x}{g:02x}{b:02x}'


# ============================================================================
# SVG generation
# ============================================================================
W, H = 1400, 1400  # canvas
svg = []
svg.append(f'<?xml version="1.0" encoding="UTF-8"?>')
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'font-family="Arial, Helvetica, sans-serif">')

# Background
svg.append(f'<rect width="{W}" height="{H}" fill="#fafafa"/>')

# Arrow marker definition
svg.append('''<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
  </marker>
</defs>''')

# ----------------------------------------------------------------------------
# TITLE
# ----------------------------------------------------------------------------
svg.append(f'<text x="{W/2}" y="38" text-anchor="middle" '
           f'font-size="24" font-weight="bold" fill="#222">'
           f'Exhaustive descriptor screening workflow '
           f'(網羅的記述子スクリーニングワークフロー)</text>')
svg.append(f'<text x="{W/2}" y="66" text-anchor="middle" '
           f'font-size="16" fill="#555">'
           f'From 34 candidates across 5 computational levels → 14 selected '
           f'→ class-specific Arrhenius fitting (R²)</text>')

# Category legend (top-right)
legend_x, legend_y = W - 300, 90
svg.append(f'<text x="{legend_x}" y="{legend_y}" font-size="13" '
           f'font-weight="bold" fill="#333">Physical category (物理類別):</text>')
for k, (cat, col) in enumerate(list(CAT_COLOR.items())[:4]):
    y = legend_y + 18 + k * 18
    svg.append(f'<rect x="{legend_x}" y="{y-12}" width="14" height="14" '
               f'fill="{col}" stroke="#333" stroke-width="0.5"/>')
    svg.append(f'<text x="{legend_x+22}" y="{y}" font-size="12" '
               f'fill="#333">{cat}</text>')

# ----------------------------------------------------------------------------
# STAGE 1 — FULL POOL (top, y=100 to 520)
# ----------------------------------------------------------------------------
stage1_y0 = 110
stage1_title_y = stage1_y0 + 20
svg.append(f'<text x="40" y="{stage1_title_y}" font-size="18" '
           f'font-weight="bold" fill="#222">'
           f'① Full descriptor pool (記述子プール)  — 34 descriptors</text>')

# Outer rounded rect
pool_x, pool_y = 40, stage1_y0 + 35
pool_w, pool_h = 900, 380
svg.append(f'<rect x="{pool_x}" y="{pool_y}" width="{pool_w}" height="{pool_h}" '
           f'fill="white" stroke="#777" stroke-width="1.5" rx="8"/>')

# Rows
row_h = 38
for i, (lvl, cat, descs, accepted) in enumerate(POOL):
    y = pool_y + 12 + i * row_h
    color = CAT_COLOR[cat] if accepted else CAT_COLOR['Rejected']
    status = '✓' if accepted else '✗'

    # Left: level label
    svg.append(f'<text x="{pool_x+14}" y="{y+20}" font-size="11.5" '
               f'font-weight="bold" fill="#333">{lvl}</text>')

    # Descriptor pills
    x_cursor = pool_x + 160
    for d in descs:
        # Pill width based on text length
        pw = 8 + len(d) * 7.5
        opacity = '1' if accepted else '0.5'
        svg.append(f'<rect x="{x_cursor}" y="{y+5}" width="{pw}" height="26" '
                   f'rx="13" fill="{color}" fill-opacity="{opacity}" '
                   f'stroke="#333" stroke-width="0.4"/>')
        text_color = 'white' if accepted else '#333'
        svg.append(f'<text x="{x_cursor + pw/2}" y="{y+22}" text-anchor="middle" '
                   f'font-size="11" fill="{text_color}" font-weight="500">{d}</text>')
        x_cursor += pw + 5

    # Right: accept/reject marker
    status_color = '#27ae60' if accepted else '#c0392b'
    svg.append(f'<text x="{pool_x+pool_w-30}" y="{y+22}" font-size="20" '
               f'font-weight="bold" fill="{status_color}">{status}</text>')

# Divider line separating L1 (accepted) from L2+
sep_y = pool_y + 12 + 5 * row_h
svg.append(f'<line x1="{pool_x+10}" y1="{sep_y}" x2="{pool_x+pool_w-10}" y2="{sep_y}" '
           f'stroke="#777" stroke-width="0.8" stroke-dasharray="5,3"/>')
svg.append(f'<text x="{pool_x+pool_w-12}" y="{sep_y-4}" text-anchor="end" font-size="10" '
           f'fill="#27ae60">— L1 retained (採用) —</text>')
svg.append(f'<text x="{pool_x+pool_w-12}" y="{sep_y+16}" text-anchor="end" font-size="10" '
           f'fill="#c0392b">— L2-L5 rejected (却下) —</text>')

# ----------------------------------------------------------------------------
# STAGE 2 — SCREENING PROCESS (right of pool, y=100 to 520)
# ----------------------------------------------------------------------------
screen_x = 980
screen_y = pool_y
screen_w = 380

# Arrow from pool to screen
svg.append(f'<line x1="{pool_x+pool_w+5}" y1="{pool_y+pool_h/2}" '
           f'x2="{screen_x-10}" y2="{pool_y+pool_h/2}" '
           f'stroke="#333" stroke-width="2.5" marker-end="url(#arrow)"/>')

# Screening box
svg.append(f'<rect x="{screen_x}" y="{screen_y}" width="{screen_w}" height="{pool_h}" '
           f'fill="#fef5e7" stroke="#f39c12" stroke-width="2" rx="8"/>')
svg.append(f'<text x="{screen_x + screen_w/2}" y="{screen_y + 32}" text-anchor="middle" '
           f'font-size="16" font-weight="bold" fill="#c67e00">② LOO screening</text>')
svg.append(f'<text x="{screen_x + screen_w/2}" y="{screen_y + 54}" text-anchor="middle" '
           f'font-size="13" fill="#c67e00">(一つ抜き交差検証)</text>')

# Screening details
details = [
    '',
    '• For each class × Arrhenius param:',
    '     enumerate all 1–3 descriptor combos',
    '     (全組合せを網羅的に探索)',
    '',
    '• Keep combos with:',
    '     LOO-R² > 0.6',
    '     No severe collinearity (共線性なし)',
    '',
    '• Count descriptors hit at least once',
    '     across all (class, param) cells',
    '     (命中した記述子のみ保持)',
    '',
    '4 classes × 4 params = 16 models',
    '   ↓',
    f'14 unique descriptors retained',
]
for k, line in enumerate(details):
    yy = screen_y + 88 + k * 18
    fw = 'bold' if '→' in line or '14 unique' in line or '16 models' in line else 'normal'
    fsz = 12.5 if fw == 'bold' else 12
    svg.append(f'<text x="{screen_x + 22}" y="{yy}" font-size="{fsz}" '
               f'fill="#333" font-weight="{fw}">{line}</text>')

# ----------------------------------------------------------------------------
# STAGE 3 — 14 RETAINED DESCRIPTORS (middle, y=540 to 760)
# ----------------------------------------------------------------------------
stage3_y0 = 540
svg.append(f'<text x="40" y="{stage3_y0+20}" font-size="18" '
           f'font-weight="bold" fill="#222">'
           f'③ 14 retained descriptors grouped by physical category '
           f'(14の採用記述子 — 物理類別別)</text>')

# Outer container
cont_x, cont_y = 40, stage3_y0 + 40
cont_w, cont_h = W - 80, 180
svg.append(f'<rect x="{cont_x}" y="{cont_y}" width="{cont_w}" height="{cont_h}" '
           f'fill="white" stroke="#777" stroke-width="1.5" rx="8"/>')

# 4 category columns
col_widths = [460, 380, 150, 280]   # weighted by count
col_x = cont_x + 10
for (cat, items), cw in zip(RETAINED, col_widths):
    color = CAT_COLOR[cat]

    # Column header
    svg.append(f'<rect x="{col_x}" y="{cont_y+12}" width="{cw-10}" height="34" '
               f'fill="{color}" rx="4"/>')
    svg.append(f'<text x="{col_x + (cw-10)/2}" y="{cont_y+34}" text-anchor="middle" '
               f'font-size="15" font-weight="bold" fill="white">'
               f'{cat}  ({len(items)})</text>')

    # Items as pills
    item_y = cont_y + 56
    ix = col_x + 8
    for d in items:
        pw = 10 + len(d) * 7.5
        if ix + pw > col_x + cw - 10:
            ix = col_x + 8
            item_y += 36
        svg.append(f'<rect x="{ix}" y="{item_y}" width="{pw}" height="28" '
                   f'rx="14" fill="white" stroke="{color}" stroke-width="1.8"/>')
        svg.append(f'<text x="{ix + pw/2}" y="{item_y+18}" text-anchor="middle" '
                   f'font-size="12" fill="#222" font-weight="600">{d}</text>')
        ix += pw + 6

    col_x += cw

# Arrows from Stage 2 to Stage 3 (vertical)
svg.append(f'<line x1="{screen_x + screen_w/2}" y1="{screen_y + pool_h + 5}" '
           f'x2="{screen_x + screen_w/2}" y2="{cont_y - 5}" '
           f'stroke="#333" stroke-width="2.5" marker-end="url(#arrow)"/>')

# ----------------------------------------------------------------------------
# STAGE 4 — CLASS × PARAM R² RESULTS (bottom)
# ----------------------------------------------------------------------------
stage4_y0 = cont_y + cont_h + 45
svg.append(f'<text x="40" y="{stage4_y0+20}" font-size="18" '
           f'font-weight="bold" fill="#222">'
           f'④ Class-specific Arrhenius fitting results (LOO-R²) '
           f'— 4 classes × 4 parameters</text>')

# R² table
tbl_x = 300
tbl_y = stage4_y0 + 45
cell_w = 170
cell_h = 68
hdr_h = 42
cls_col_w = 200

# Header row (params)
svg.append(f'<rect x="{tbl_x}" y="{tbl_y}" width="{cls_col_w}" height="{hdr_h}" '
           f'fill="#34495e"/>')
svg.append(f'<text x="{tbl_x + cls_col_w/2}" y="{tbl_y + 27}" text-anchor="middle" '
           f'font-size="14" font-weight="bold" fill="white">Class / パラメータ</text>')
for j, p in enumerate(PARAMS):
    x = tbl_x + cls_col_w + j * cell_w
    svg.append(f'<rect x="{x}" y="{tbl_y}" width="{cell_w}" height="{hdr_h}" '
               f'fill="#34495e" stroke="white" stroke-width="1.5"/>')
    svg.append(f'<text x="{x + cell_w/2}" y="{tbl_y + 27}" text-anchor="middle" '
               f'font-size="16" font-weight="bold" fill="white">{p}</text>')

# Data rows
for i, cls in enumerate(CLASSES):
    y = tbl_y + hdr_h + i * cell_h
    # Class label
    svg.append(f'<rect x="{tbl_x}" y="{y}" width="{cls_col_w}" height="{cell_h}" '
               f'fill="#ecf0f1" stroke="white" stroke-width="1.5"/>')
    svg.append(f'<text x="{tbl_x + cls_col_w/2}" y="{y + cell_h/2 + 6}" '
               f'text-anchor="middle" font-size="16" font-weight="bold" '
               f'fill="#222">{cls}</text>')

    # R² cells
    for j, p in enumerate(PARAMS):
        x = tbl_x + cls_col_w + j * cell_w
        r2 = R2_TABLE[cls][p]
        bg = r2_to_color(r2)
        svg.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                   f'fill="{bg}" stroke="white" stroke-width="1.5"/>')
        if r2 is None:
            txt = 'class\nmean'
            svg.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 - 2}" '
                       f'text-anchor="middle" font-size="12" font-style="italic" '
                       f'fill="#555">(not predictable)</text>')
            svg.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 14}" '
                       f'text-anchor="middle" font-size="11" font-style="italic" '
                       f'fill="#555">(予測不能)</text>')
        else:
            text_color = 'white' if r2 > 0.9 else '#222'
            svg.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 7}" '
                       f'text-anchor="middle" font-size="22" font-weight="bold" '
                       f'fill="{text_color}">{r2:.3f}</text>')

# Border around entire table
tbl_total_h = hdr_h + len(CLASSES) * cell_h
tbl_total_w = cls_col_w + len(PARAMS) * cell_w
svg.append(f'<rect x="{tbl_x}" y="{tbl_y}" width="{tbl_total_w}" height="{tbl_total_h}" '
           f'fill="none" stroke="#333" stroke-width="2"/>')

# Arrow from Stage 3 to Stage 4
svg.append(f'<line x1="{W/2}" y1="{cont_y + cont_h + 5}" '
           f'x2="{W/2}" y2="{tbl_y - 5}" '
           f'stroke="#333" stroke-width="2.5" marker-end="url(#arrow)"/>')

# Final conclusion box below table
concl_y = tbl_y + tbl_total_h + 40
concl_x = 180
concl_w = W - 360
concl_h = 90
svg.append(f'<rect x="{concl_x}" y="{concl_y}" width="{concl_w}" height="{concl_h}" '
           f'fill="#fdf2e9" stroke="#e67e22" stroke-width="2" rx="8"/>')
svg.append(f'<text x="{W/2}" y="{concl_y + 30}" text-anchor="middle" '
           f'font-size="16" font-weight="bold" fill="#c0392b">'
           f'Conclusion (結論): 13 / 16 models achieve LOO-R² &gt; 0.90</text>')
svg.append(f'<text x="{W/2}" y="{concl_y + 58}" text-anchor="middle" '
           f'font-size="14" fill="#333">'
           f'E_a ← Electronic + Steric (電子 + 立体) ；  '
           f'ln A ← Solvation + Polarity (溶媒和 + 極性)</text>')
svg.append(f'<text x="{W/2}" y="{concl_y + 78}" text-anchor="middle" '
           f'font-size="13" fill="#555" font-style="italic">'
           f'→ 81 % accuracy on flash/flow/batch reactor recommendation '
           f'(フラッシュ/フロー/バッチ反応器推薦)</text>')

svg.append('</svg>')

os.makedirs('analysis_figures', exist_ok=True)
out_path = 'analysis_figures/descriptor_screening_flowchart.svg'
with open(out_path, 'w') as f:
    f.write('\n'.join(svg))

print(f'✓ {out_path}')
print(f'  尺寸: {W} × {H}')
print(f'  (SVG 可无损缩放, 适合 PPT/海报/论文插图)')
