"""
按四类画 Arrhenius plot: ln(k_d) vs 1000/T
每类一张大图, 每个子图 = 一个化合物
子图标注: 分子结构 + Ea/lnA + R²
字体大, 便于阅读。
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
import os
import math

R_GAS = 8.314e-3  # kJ/(mol·K)

# ===== 加载数据 =====
arr = pd.read_csv('global_arrhenius.csv')
desc = pd.read_csv('clean_organolithium_unified_descriptors.csv')
unique = desc.drop_duplicates(subset='intermediate_smiles_canonical')
merged = arr.merge(unique[['intermediate_smiles_canonical', 'intermediate_class', 'intermediate']],
                   left_on='smi', right_on='intermediate_smiles_canonical', how='left',
                   suffixes=('', '_desc'))

tier = pd.read_csv('model_comparison_L1_L2.csv')
tier['tier_clean'] = tier['tier'].apply(
    lambda x: 'Tier A' if 'Tier A' in x else ('Tier B' if 'Tier B' in x else 'Tier C'))
excluded_smi = set(tier[tier['tier_clean'].isin(['Tier A', 'Tier B'])]['smi'])
modeling = merged[~merged['smi'].isin(excluded_smi)].copy()

halflives = pd.read_csv('phase_a_halflives.csv')

def classify(smi):
    smi = str(smi)
    if any(p in smi for p in ['CO1', 'C1CO1', 'OC1', 'C1OC1']):
        return 'oxiranylLi'
    if '[Li]c1ccccc1' in smi and smi != '[Li]c1ccccc1':
        return 'o-ArLi'
    if '[Li]c1ccc(' in smi:
        return 'p-ArLi'
    if '[Li]c1cccc(' in smi:
        return 'm-ArLi'
    if '[Li]c1' in smi:
        return 'hetero-ArLi'
    return 'other'

modeling['cls'] = modeling['smi'].apply(classify)

def short_label(name):
    name = str(name).strip()
    replacements = {
        'monolithiated ': '', ' anion': '', ' intermediate': '',
        'lithium ': '', 'lithiated ': '',
        "4,4'-dibromobiphenyl": "4,4'-Br2-biphenyl",
        '2-(2-naphthyl)ethylene oxide': '2-naphthyl-oxirane',
        'ethylene oxide': 'oxirane',
        "2-bromo-2'-lithiobiphenyl": "2-Br-biphenyl-2'",
        'methyl ': 'Me-', 'ethyl ': 'Et-',
        'isopropyl ': 'iPr-', 'tert-butyl ': 'tBu-',
        'triphenylsilyl': 'SiPh3',
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    return name

def get_mol_image(smi, size=(300, 240)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.06
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def draw_class_arrhenius(cls_code, cls_title, cls_color, output_path, ncols=3):
    sub = modeling[modeling['cls'] == cls_code].sort_values('intermediate_desc').reset_index(drop=True)
    n = len(sub)
    if n == 0:
        print(f'(skip {cls_code}: empty)')
        return

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 5.4*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(cls_title, fontsize=26, fontweight='bold',
                 color=cls_color, y=0.998)

    for i, row in sub.iterrows():
        ax = axes[i // ncols, i % ncols]
        smi = row['smi']
        name = short_label(row['intermediate_desc'])

        # 从 phase_a_halflives 提取 per-T k_d 数据点 (去除 NaN)
        hdata = halflives[halflives['intermediate_smiles'] == smi].dropna(subset=['k_d', 'T_K'])
        hdata = hdata[hdata['k_d'] > 0].copy()

        Ea_d = row['Ea_d']
        lnA_d = row['lnA_d']
        r2_global = row.get('r2_global', np.nan)

        # 确定 x-range: 优先用数据点范围, 否则默认 -78 to +25°C
        if len(hdata) >= 1:
            T_min = max(hdata['T_K'].min() - 8, 180)
            T_max = hdata['T_K'].max() + 8
        else:
            T_min, T_max = 195, 300

        # 拟合线
        T_fit = np.linspace(T_min, T_max, 100)
        lnk_fit = lnA_d - Ea_d / (R_GAS * T_fit)
        ax.plot(1000/T_fit, lnk_fit, '-', color=cls_color, lw=2.8, alpha=0.9,
                label='global fit', zorder=3)

        # 数据点
        if len(hdata) >= 1:
            ax.scatter(1000/hdata['T_K'], np.log(hdata['k_d']),
                       s=160, color=cls_color, edgecolor='black',
                       linewidth=1.8, zorder=5, label='data')

        n_data = len(hdata)
        insufficient = n_data < 3
        r2_text = (f'R² = {r2_global:.3f}'
                   if not np.isnan(r2_global) else 'R² = N/A')

        # 标题 (化合物名)
        ax.set_title(name, fontsize=16, fontweight='bold', pad=10)

        # 轴标签与刻度
        ax.set_xlabel('1000 / T  (K⁻¹)', fontsize=15)
        ax.set_ylabel(r'ln $k_d$', fontsize=16)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(alpha=0.3, linestyle='--')

        # 注释: Ea, lnA, R² (左下角, 字号更大)
        text = (f'$E_a$ = {Ea_d:.1f} kJ/mol\n'
                f'ln A = {lnA_d:.2f}\n'
                f'{r2_text}\n'
                f'n = {n_data}')
        ax.text(0.04, 0.04, text,
                transform=ax.transAxes,
                fontsize=14, verticalalignment='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white', edgecolor=cls_color,
                          linewidth=1.5, alpha=0.95))

        # 分子结构 inset (右上角)
        mol_img = get_mol_image(smi, size=(320, 240))
        if mol_img is not None:
            imagebox = OffsetImage(mol_img, zoom=0.48)
            ab = AnnotationBbox(imagebox, (0.76, 0.82),
                                xycoords='axes fraction', frameon=False,
                                box_alignment=(0.5, 0.5))
            ax.add_artist(ab)

        # 温度点不足时红框警告
        if insufficient:
            for spine in ax.spines.values():
                spine.set_edgecolor('#cb181d')
                spine.set_linewidth(2.5)

    # 隐藏多余子图
    for j in range(n, nrows*ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'✓ {cls_code}: {n} compounds → {output_path}')

# ===== 输出 =====
os.makedirs('analysis_figures', exist_ok=True)

class_config = [
    ('p-ArLi',     'p-ArLi: para-substituted aryllithiums',
     '#1f77b4', 'analysis_figures/arrhenius_p_ArLi.png', 3),
    ('m-ArLi',     'm-ArLi: meta-substituted aryllithiums',
     '#2ca02c', 'analysis_figures/arrhenius_m_ArLi.png', 3),
    ('o-ArLi',     'o-ArLi: ortho-substituted aryllithiums (chelation / benzyne)',
     '#d62728', 'analysis_figures/arrhenius_o_ArLi.png', 3),
    ('oxiranylLi', 'oxiranylLi: epoxide lithiums (ring-opening)',
     '#ff7f0e', 'analysis_figures/arrhenius_oxiranylLi.png', 3),
]

for code, title, color, path, ncols in class_config:
    draw_class_arrhenius(code, title, color, path, ncols=ncols)

print('\n=== 完成 ===')
