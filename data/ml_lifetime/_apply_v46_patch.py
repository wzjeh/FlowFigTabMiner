"""Insert v4.6 patch cells (markdown + code) after cell 36 (7c.10) in the notebook."""
import json
import uuid
from pathlib import Path

NB = Path(__file__).parent / "organolithium_stability_tutorial.ipynb"

MARKDOWN_CELL = """### 7c.10b  v4.6 / HYBRID v3 patch — 自适应规则选择

**为什么需要这个 patch**: 2026-05-10 用 4-溴氟苯实验数据回查 m-ArLi Ea_d (LOO-R²=0.290) 时发现 v4.5 筛选两个可改进点:

1. **排序用 in-sample R² 而非 LOO** — m-ArLi Ea_d 的 fukui+B1+ΔS_dim (in-sample 0.97 / LOO 0.29) 被选, 但 LOO 排序应该选 Es+BDE+ΔS (LOO 0.45) 等
2. **|r|<0.7 共线性约束太严** — 真最优 Gsolv+vol+ΔVbur (LOO=**0.9999**) 因 |r|=0.94 (m-ArLi 化合物的 Gsolv 与分子 vol 强相关) 被排除, 但 OLS+ΔVbur 仍能精确恢复 Ea_d

**v4.6 修复**: 对每个 (class × param), 在 4 种规则变体下分别筛选, 取 LOO-R² 最高者:

| 规则 | \\|r\\| 阈值 | 排序 | Agg 约束 |
|---|---|---|---|
| R0 (复现 v4.5) | 0.7 / oxLi 0.99 | in-sample R² | ArLi 含 / oxLi 排 |
| R1 严约束+LOO | 0.7 / oxLi 0.99 | LOO-R² | 同 v4.5 |
| R2 放宽共线 | 0.95 | LOO-R² | 同 v4.5 |
| R3 完全自由 | 0.95 | LOO-R² | 不强制 |

**结果**: 5 个模型显著改善, 0 退化 (详见 `MODEL_CHANGELOG.md` v4.6 段落).

本 cell 加载 `class_fitted_models_v46.csv` **覆盖 `class_best_HYBRID` dict** 中的 16 个条目, 并重新 OLS 拟合系数. 后续 7c.11/7c.12/Step 8/9/10 均使用 `class_best_HYBRID`, 因此自动获得 v4.6 数字, 无需改任何下游 cell."""

CODE_CELL = """# ===== 7c.10b  v4.6 / HYBRID v3 patch =====
# 加载 class_fitted_models_v46.csv 覆盖 class_best_HYBRID dict.
# 不改 7c.10 cell, 不覆盖 class_fitted_models_HYBRID.csv (保留 v4.5 历史).

import pandas as pd
from sklearn.linear_model import LinearRegression

v46 = pd.read_csv('analysis_figures/class_fitted_models_v46.csv')
print(f"Loaded {len(v46)} v4.6 model specs from class_fitted_models_v46.csv")

# Map v4.6 descriptor names -> notebook modeling_set_v3 column names
# (v4.6 builder uses 'pVbur' for buried_vol_Li; notebook uses '%Vbur')
NAME_MAP = {'pVbur': '%Vbur'}

def map_name(n):
    return NAME_MAP.get(n, n)

# --- Track changes vs v4.5 ---
v45_dict = {(c, p): info for c, params in class_best_HYBRID.items()
            for p, info in (params or {}).items() if info}

n_changed = n_unchanged = n_skipped = 0
patch_log = []

for _, row in v46.iterrows():
    cls, param = row['class'], row['param']
    descs = [map_name(d) for d in row['descriptors'].split('+')]

    cd = modeling_set_v3[modeling_set_v3['class'] == cls]
    sub = cd[descs + [param]].dropna()
    if len(sub) < len(descs) + 2:
        print(f"  [skip] {cls} {param}: insufficient data after dropna ({len(sub)})")
        n_skipped += 1
        continue
    X, y = sub[descs].values, sub[param].values

    # Re-fit OLS to get coefs (v46 CSV has them but recompute for safety)
    reg = LinearRegression().fit(X, y)

    # LOO-R² re-evaluation
    n = len(y); pred = np.zeros(n)
    for k in range(n):
        msk = np.arange(n) != k
        pred[k] = LinearRegression().fit(X[msk], y[msk]).predict(X[k:k+1])[0]
    sr = ((y - pred) ** 2).sum(); st = ((y - y.mean()) ** 2).sum()
    r2_loo = 1 - sr / st if st > 1e-10 else float('-inf')

    # max intra |r|
    if X.shape[1] >= 2:
        R = np.corrcoef(X.T)
        max_off = max(abs(R[i, j]) for i in range(X.shape[1])
                      for j in range(i + 1, X.shape[1]))
    else:
        max_off = 0.0

    new_info = {'cols': descs, 'r2': r2_loo, 'max_intra_r': max_off}

    old_info = v45_dict.get((cls, param))
    old_descs = old_info['cols'] if old_info else None
    old_r2 = old_info['r2'] if old_info else None
    if old_descs == descs:
        n_unchanged += 1
        change = ''
    else:
        n_changed += 1
        change = f"  {'+'.join(old_descs) if old_descs else 'NA'} ({old_r2:.3f}) → {'+'.join(descs)} ({r2_loo:.3f}, {row['rule_used']})"
        patch_log.append((cls, param, change))

    class_best_HYBRID[cls][param] = new_info

class_best = class_best_HYBRID  # keep alias for downstream

print()
print(f"v4.6 patch summary:")
print(f"  changed   : {n_changed:2d} / 16")
print(f"  unchanged : {n_unchanged:2d} / 16")
print(f"  skipped   : {n_skipped:2d} / 16")
if patch_log:
    print(f"\\nChanges (formula → formula, rule):")
    for cls, param, line in patch_log:
        print(f"  {cls:11s} {param:5s}{line}")

# Final stats
all_r2 = [info['r2'] for params in class_best_HYBRID.values()
          for info in params.values() if info]
print(f"\\nv4.6 LOO-R²: median={np.median(all_r2):.3f}  mean={np.mean(all_r2):.3f}")
print(f"  R²≥0.95: {sum(1 for r in all_r2 if r>=0.95)}/16  ≥0.90: {sum(1 for r in all_r2 if r>=0.90)}/16  ≥0.80: {sum(1 for r in all_r2 if r>=0.80)}/16")"""


def main():
    nb = json.loads(NB.read_text())
    cells = nb['cells']

    # Find cell 36 (7c.10 code) by id
    target_id = '6c8d676b'
    idx = next((i for i, c in enumerate(cells) if c.get('id') == target_id), None)
    assert idx is not None, "cell 7c.10 not found"
    print(f"Found 7c.10 code cell at index {idx}")

    # Idempotency: if a cell with text 'v4.6 patch' already follows, skip
    if idx + 1 < len(cells):
        nxt = ''.join(cells[idx + 1]['source']) if isinstance(cells[idx + 1]['source'], list) else cells[idx + 1]['source']
        if 'v4.6 / HYBRID v3 patch' in nxt:
            print("v4.6 patch markdown cell already present — nothing to do.")
            return

    new_md = {
        'cell_type': 'markdown',
        'id': uuid.uuid4().hex[:8],
        'metadata': {},
        'source': MARKDOWN_CELL.splitlines(keepends=True),
    }
    new_code = {
        'cell_type': 'code',
        'id': uuid.uuid4().hex[:8],
        'metadata': {},
        'source': CODE_CELL.splitlines(keepends=True),
        'execution_count': None,
        'outputs': [],
    }

    cells.insert(idx + 1, new_md)
    cells.insert(idx + 2, new_code)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Inserted 2 cells at index {idx+1}, {idx+2}.")
    print(f"  md cell id:   {new_md['id']}")
    print(f"  code cell id: {new_code['id']}")
    print(f"New total cells: {len(cells)}")


if __name__ == "__main__":
    main()
