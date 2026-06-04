"""
按四类画 yield vs tR 曲线, 每个子图 = 一个化合物
- 原始数据点 (clean_organolithium_unified.csv)
- 全局 5-参数 Arrhenius 模型拟合曲线 (yield = y_max * (1-exp(-k_f*tR)) * exp(-k_d*tR))
- 按温度颜色编码 (cold 蓝 → hot 红)
- 分子结构 inset + Ea/lnA/R²
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import cm
from matplotlib.colors import Normalize
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

raw = pd.read_csv('clean_organolithium_unified.csv')

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

def global_yield_model(tR, T_K, Ea_f, lnA_f, Ea_d, lnA_d, y_max):
    """全局 5-参数模型: yield = y_max * (1-exp(-k_f*tR)) * exp(-k_d*tR)"""
    k_f = np.exp(lnA_f - Ea_f/(R_GAS * T_K))
    k_d = np.exp(lnA_d - Ea_d/(R_GAS * T_K))
    return y_max * (1 - np.exp(-k_f*tR)) * np.exp(-k_d*tR)

def draw_class_yield(cls_code, cls_title, cls_color, output_path, ncols=3):
    sub = modeling[modeling['cls'] == cls_code].sort_values('intermediate_desc').reset_index(drop=True)
    n = len(sub)
    if n == 0:
        return

    nrows = math.ceil(n / ncols)
    # 每子图额外留出顶部空间放分子结构
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2*ncols, 7.1*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(cls_title, fontsize=26, fontweight='bold', color='black', y=0.998)

    for i, row in sub.iterrows():
        ax = axes[i // ncols, i % ncols]
        smi = row['smi']
        name = short_label(row['intermediate_desc'])

        # 全局模型参数
        Ea_f = row['Ea_f']; lnA_f = row['lnA_f']
        Ea_d = row['Ea_d']; lnA_d = row['lnA_d']
        y_max = row['y_max']
        r2_global = row.get('r2_global', np.nan)

        # 原始数据点
        rd = raw[raw['intermediate_smiles_canonical'] == smi][['tR1_s', 'T1_C', 'yield_pct']].dropna()
        rd = rd[(rd['tR1_s'] > 0) & (rd['yield_pct'].notna())]

        # 温度颜色映射 (-78 冷蓝 → +25 热红)
        norm = Normalize(vmin=-78, vmax=25)
        cmap = cm.get_cmap('coolwarm')

        # 按 T 分组画数据点
        temps_in_data = sorted(rd['T1_C'].unique())
        for T in temps_in_data:
            d_T = rd[rd['T1_C'] == T]
            color = cmap(norm(T))
            ax.scatter(d_T['tR1_s'], d_T['yield_pct'],
                       s=85, color=color, edgecolor='black',
                       linewidth=1.1, alpha=0.85, zorder=5)

        # 在每个 T 画全局模型拟合曲线
        if len(rd) > 0:
            tR_min = max(rd['tR1_s'].min() * 0.5, 1e-4)
            tR_max = rd['tR1_s'].max() * 2
        else:
            tR_min, tR_max = 1e-3, 100
        tR_fit = np.logspace(np.log10(tR_min), np.log10(tR_max), 200)

        for T in temps_in_data:
            T_K = T + 273.15
            y_fit = global_yield_model(tR_fit, T_K, Ea_f, lnA_f, Ea_d, lnA_d, y_max)
            color = cmap(norm(T))
            ax.plot(tR_fit, y_fit, '-', color=color, lw=2.3, alpha=0.9,
                    label=f'T = {T:+.0f}°C')

        # 轴
        ax.set_xscale('log')
        ax.set_xlabel(r'$t_R$  (s)', fontsize=15)
        ax.set_ylabel('Yield  (%)', fontsize=15)
        ax.set_ylim(-5, 105)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(alpha=0.3, linestyle='--')

        # 子图编号 (替代化合物名, 避免与分子结构重叠)
        label = f'({chr(ord("a") + i)})'
        ax.text(-0.12, 1.26, label,
                transform=ax.transAxes,
                fontsize=22, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                color='black')

        # 参数标注 (左下角) — 显示全部 5 个全局拟合参数
        r2_text = f'R² = {r2_global:.3f}' if not np.isnan(r2_global) else 'R² = N/A'
        text = (f'Formation (生成, $k_f$):\n'
                f'  $E_{{a,f}}$ = {Ea_f:.1f} kJ/mol,  ln $A_f$ = {lnA_f:.2f}\n'
                f'Decomposition (分解, $k_d$):\n'
                f'  $E_{{a,d}}$ = {Ea_d:.1f} kJ/mol,  ln $A_d$ = {lnA_d:.2f}\n'
                f'$y_{{max}}$ = {y_max:.1f} %    {r2_text}')
        ax.text(0.03, 0.03, text,
                transform=ax.transAxes,
                fontsize=11.5, verticalalignment='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white', edgecolor='black',
                          linewidth=1.2, alpha=0.95))

        # 图例 (温度)
        if len(temps_in_data) >= 1:
            ax.legend(loc='upper right', fontsize=10.5, framealpha=0.85,
                      ncol=1 if len(temps_in_data) <= 5 else 2)

        # 分子结构 inset (350 px 宽, 上移, 避免遮挡子图顶部)
        mol_img = get_mol_image(smi, size=(350, 245))
        if mol_img is not None:
            imagebox = OffsetImage(mol_img, zoom=0.42)
            ab = AnnotationBbox(imagebox, (0.5, 1.22),
                                xycoords='axes fraction', frameon=False,
                                box_alignment=(0.5, 0.5))
            ax.add_artist(ab)

        # 数据不足警告 (用黑色加粗边框代替红色)
        if len(temps_in_data) < 3:
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2.5)
                spine.set_linestyle('--')

    for j in range(n, nrows*ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=170, bbox_inches='tight')
    plt.close()
    print(f'✓ {cls_code}: {n} compounds → {output_path}')

# ===== 输出 =====
os.makedirs('analysis_figures', exist_ok=True)

class_config = [
    ('p-ArLi',     'p-ArLi: para-substituted aryllithiums',
     '#1f77b4', 'analysis_figures/yield_curves_p_ArLi.png', 3),
    ('m-ArLi',     'm-ArLi: meta-substituted aryllithiums',
     '#2ca02c', 'analysis_figures/yield_curves_m_ArLi.png', 3),
    ('o-ArLi',     'o-ArLi: ortho-substituted aryllithiums (chelation / benzyne)',
     '#d62728', 'analysis_figures/yield_curves_o_ArLi.png', 3),
    ('oxiranylLi', 'oxiranylLi: epoxide lithiums (ring-opening)',
     '#ff7f0e', 'analysis_figures/yield_curves_oxiranylLi.png', 3),
]

for code, title, color, path, ncols in class_config:
    draw_class_yield(code, title, color, path, ncols=ncols)

print('\n=== 完成 ===')
