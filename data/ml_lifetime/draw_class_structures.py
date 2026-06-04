"""
按四类 (p-ArLi, m-ArLi, o-ArLi, oxiranylLi) 画出 Tier C 预测化合物的分子结构图。
每类一张图，保存到 analysis_figures/。
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os
import math

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

# ===== 短标签清理 =====
def short_label(name):
    """把长的中间体名字缩短为可读标签"""
    name = str(name).strip()
    replacements = {
        'monolithiated ': '',
        ' anion': '',
        ' intermediate': '',
        'lithium ': '',
        'lithiated ': '',
        "4,4'-dibromobiphenyl": "4,4'-Br2-biphenyl",
        '2-(2-naphthyl)ethylene oxide': '2-naphthyl-oxirane',
        'ethylene oxide': 'oxirane',
        "2-bromo-2'-lithiobiphenyl": "2-Br-biphenyl-2'",
        'methyl ': 'Me-',
        'ethyl ': 'Et-',
        'isopropyl ': 'iPr-',
        'tert-butyl ': 'tBu-',
        'triphenylsilyl': 'SiPh3',
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    return name

# ===== 渲染函数 =====
def draw_class_figure(cls_name, compounds, output_path, ncols=3, figsize_per=None):
    n = len(compounds)
    nrows = math.ceil(n / ncols)
    if figsize_per is None:
        figsize_per = (3.5, 3.5)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = axes.flatten()

    fig.suptitle(f'{cls_name}  (n={n})', fontsize=18, fontweight='bold', y=0.995)

    for i, (_, row) in enumerate(compounds.iterrows()):
        ax = axes[i]
        smi = row['smi']
        label = short_label(row['intermediate_desc'])

        # 生成分子图 (RDKit → PNG → matplotlib)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            ax.text(0.5, 0.5, f'(invalid SMILES)\n{smi}', ha='center', va='center')
            ax.set_title(label, fontsize=10)
            ax.axis('off')
            continue
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2
        opts.padding = 0.05
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(BytesIO(drawer.GetDrawingText()))
        ax.imshow(img)
        ax.set_title(label, fontsize=11, pad=6)
        ax.axis('off')

    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'✓ {cls_name}: {n} compounds → {output_path}')

# ===== 输出目录 =====
os.makedirs('analysis_figures', exist_ok=True)

# ===== 四类分别画图 =====
class_config = [
    ('p-ArLi',     'p-ArLi:  para-substituted aryllithiums',           'analysis_figures/class_p_ArLi.png',     3),
    ('m-ArLi',     'm-ArLi:  meta-substituted aryllithiums',           'analysis_figures/class_m_ArLi.png',     3),
    ('o-ArLi',     'o-ArLi:  ortho-substituted aryllithiums (chelation / benzyne)', 'analysis_figures/class_o_ArLi.png',     3),
    ('oxiranylLi', 'oxiranylLi:  epoxide lithiums (ring-opening)',     'analysis_figures/class_oxiranylLi.png', 3),
]

for cls_code, title, path, ncols in class_config:
    sub = modeling[modeling['cls'] == cls_code].sort_values('intermediate_desc')
    if len(sub) == 0:
        print(f'(skip {cls_code}: empty)')
        continue
    draw_class_figure(title, sub, path, ncols=ncols)

print('\n=== 完成 ===')
