import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tools.config as config

INPUT_DIR = '../../../input'
df_sorted = pd.read_excel(os.path.join(INPUT_DIR, 'BIODIVERSITY_GBF2_conservation_performance.xlsx'))

x = df_sorted['AREA_COVERAGE_PERCENT'].values
y = df_sorted['PRIORITY_RANK_CUMSUM_CONTRIBUTION'].values

# 要标注的x轴点
x_labels = [10, 20, 30, 40, 50]
for x_label in x_labels:
    y_label = np.interp(x_label, x, y)
    plt.scatter(x_label, y_label, color='red', s=50, zorder=5)
    plt.text(x_label + 2, y_label - 1, f'Area: {x_label}%, biodiversity: {y_label:.2f}%',
             fontsize=10, fontname='Arial', color='black')

# 主曲线
plt.plot(x, y, color='black', linewidth=1.5, label='Cumulative Biodiversity Contribution')

# 图形美化
plt.xlabel('Cumulative Area', fontname='Arial', fontsize=10)
plt.ylabel('Cumulative Biodiversity Contribution', fontname='Arial', fontsize=10)
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'], fontname='Arial')
plt.yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'], fontname='Arial')
plt.tick_params(axis='both', direction='out', length=5, width=1)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
    spine.set_color('black')
plt.grid(False)
plt.tight_layout()
plt.legend(frameon=False, fontsize=10, prop={'family': 'Arial'})

base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
out_dir = f"{base_dir}/3_Paper_figure"
output_path = os.path.join(out_dir, "06_biodiversity_contribution_curve")

plt.savefig(f"{output_path}.png", dpi=300)
plt.show()