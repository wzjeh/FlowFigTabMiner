import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import matplotlib.patheffects as patheffects

# 既存のデータ
x0 = np.array([0.002, 0.0069, 0.11, 1.6, 6.3])
y0 = np.array([0, -20, -40, -60, -78])
z0 = np.array([
    [np.nan, np.nan, 51, 0, 0],
    [88, 88, 87, 47, 23],
    [87, 89, 88, 89, 77],
    [73, 88, 93, 93, 92],
    [63, 87, 91, 95, 95]
])

# nanを含むデータポイントを除外
x, y = np.meshgrid(x0, y0)
points = np.array([x[~np.isnan(z0)], y[~np.isnan(z0)]]).T
values = z0[~np.isnan(z0)]

# x0を対数化
x1 = np.log10(x0)

# 元データのポイントを作成
X, Y = np.meshgrid(x1, y0)

# 等高線の表示範囲と密度を決定
xi = np.linspace(x1.min(), x1.max(), 500)
yi = np.linspace(y0.min(), y0.max(), 500)
XI, YI = np.meshgrid(xi, yi)

# RBF（外挿可）で補間：既知点（NaN除外）を学習データに使用
mask = ~np.isnan(z0)
x_train = X[mask]
y_train = Y[mask]
z_train = z0[mask]

# RBF を構築する前に、距離計量の歪みを避けるため x,y を同程度のスケールに正規化
sx = np.std(x_train)
sy = np.std(y_train)
# 0除算を避けるための保険
sx = sx if sx > 0 else 1.0
sy = sy if sy > 0 else 1.0

x_train_s = x_train / sx
y_train_s = y_train / sy

# RBF を構築（より非振動的な 'linear' を既定、必要に応じて 'multiquadric' へ）
# 滑らかさ制御のため small smooth を入れる（0 にすると過学習＆縞の原因になりやすい）
rbf = Rbf(x_train_s, y_train_s, z_train, function='linear', smooth=0.2)

# グリッド上でも同じスケーリングで評価
XI_s = XI / sx
YI_s = YI / sy
ZI = rbf(XI_s, YI_s)

# 外挿や局所振動で生じた範囲外を可視化前にクリップ
ZI = np.clip(ZI, 0, 100)
print("ZI min/max after clip (RBF):", np.nanmin(ZI), np.nanmax(ZI))
# カスタム関数を定義
def format_func(value, tick_number):
    return f"$10^{{{value:.1f}}}$"

# 使用するフォントを一括指定
my_font='Helvetica'
my_fontsize=12
plt.rcParams["font.family"] = my_font   
plt.rcParams["font.size"] = my_fontsize  
plt.rcParams['mathtext.fontset'] = 'dejavusans'

fig, ax = plt.subplots()

# x軸のラベルをカスタム形式に設定
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))


# 0から100までの間隔を指定して、等高線を表示
levels = np.linspace(0, 100, 21)  # 例: 0, 5, 10, ..., 95, 100の場合21
norm = Normalize(vmin=0, vmax=100)
cntr = plt.contourf(XI, YI, ZI, levels=levels, cmap='bwr', norm=norm, extend='both')
 
# zlabelを設定 (変更してください)
my_zlabel = "yield(%)"

# カラーバーの設定
cbar = plt.colorbar(cntr, ax=ax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(my_zlabel, rotation=0, va='bottom', ha='left', labelpad=15)
cbar_label = cbar.ax.yaxis.get_label()
cbar_label.set_verticalalignment('bottom')
cbar_label.set_horizontalalignment('right')
cbar_label.set_position((0, 1.05))
cbar.set_ticks(np.arange(0, 101, 10))

# 元のz0の値がnanでない部分のみをプロット
mask = ~np.isnan(z0)
plt.scatter(X[mask], Y[mask], s=30, facecolor='white', edgecolors='black', zorder=5, clip_on=False)  # 元データのポイントも表示

# オフセットを設定
offset = 0.05
offset2 = 1
# 白抜きの黒いテキストを黒丸の隣に表示
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if not np.isnan(z0[i, j]):
           ha = 'left' if X[i, j] < X.mean() else 'right'
           va = 'top' if Y[i, j] > Y.mean() else 'bottom'
           ax.text(X[i, j] + (offset if ha == 'left' else -offset),
                     Y[i, j] + (offset2 if va == 'bottom' else -offset2),
                     f'{z0[i, j]:.0f}', 
                     verticalalignment=va, horizontalalignment=ha, color='black', fontsize=14, fontname = 'Helvetica', clip_on=False
                     ,path_effects=[patheffects.withStroke(linewidth=3, foreground='white', capstyle="round")])

plt.xlabel('$t_{1}$(s)')
plt.ylabel('$T$(°C)')
#plt.title('Contour plot with cubic interpolation')

# 軸ラベルの位置調整
y_label = ax.yaxis.get_label()
y_label.set_rotation(0)
y_label.set_verticalalignment('bottom')
y_label.set_horizontalalignment('left')
y_label.set_position((0, 1.05))  # ここで位置を調整します
ax.yaxis.labelpad = -10  # 必要に応じてパディングを調整

# 解像度を指定して保存
plt.savefig("figure1.tif", format="tif", dpi=300)
