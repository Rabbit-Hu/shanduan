import numpy as np
import pandas as pd
import seaborn as sns; sns.set_theme()
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mtx_shanduan = np.array([[5.4264e+05, 3.1950e+03, 6.0800e+02, 1.0060e+03, 3.2000e+01, 1.6000e+01, 2.5300e+02, 3.1000e+01], [8.1240e+03, 7.5004e+04, 4.8830e+03, 4.2400e+02, 4.2000e+01, 3.2000e+01, 6.3000e+01, 5.3500e+02], [4.4900e+02, 5.4830e+03, 2.7693e+04, 5.9000e+01, 3.5000e+01, 2.4900e+02, 5.4000e+01, 6.8000e+02], [5.8900e+02, 6.9000e+01, 1.4000e+01, 4.4080e+03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00], [1.4000e+01, 8.8000e+01, 2.5000e+01, 0.0000e+00, 1.1830e+03, 2.3500e+02, 1.0000e+00, 3.0000e+00],[9.0000e+00, 5.5000e+01, 1.2200e+02, 0.0000e+00, 5.7600e+02, 7.2700e+02, 1.0000e+00, 2.0000e+00], [3.0000e+02, 6.8000e+01, 1.5000e+01, 2.0000e+00, 0.0000e+00, 0.0000e+00, 6.9760e+03, 3.1000e+01], [6.0000e+00, 3.5400e+02, 1.0200e+02, 1.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 3.0800e+02]])
mtx_shanduan = mtx_shanduan / mtx_shanduan.sum() * 100

mtx_jiayan = np.array([[5.3713e+05, 1.0659e+04, 4.7880e+03, 2.2290e+03, 2.2500e+02, 1.5500e+02, 7.0100e+02, 2.4700e+02], [1.4491e+04, 7.3240e+04, 2.6754e+04, 3.4930e+03, 2.2800e+02, 3.0200e+02, 3.3800e+02, 1.2070e+03], [3.3650e+03, 1.3800e+02, 1.5940e+03, 3.0000e+00, 4.0000e+00, 1.8000e+01, 1.5000e+01, 2.2000e+01], [2.3000e+01, 7.0000e+00, 2.0000e+00, 1.3400e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], [1.5200e+02, 1.7800e+02, 7.7000e+01, 1.7000e+01, 1.3210e+03, 4.9200e+02, 3.0000e+00, 4.0000e+00], [4.2000e+01, 4.8000e+01, 1.3100e+02, 2.0000e+00, 8.9000e+01, 2.9100e+02, 1.0000e+00, 2.0000e+00], [3.4300e+02, 2.3000e+01, 1.2000e+01, 9.0000e+00, 0.0000e+00, 0.0000e+00, 6.2920e+03, 3.0000e+01], [9.0000e+00, 2.5000e+01, 1.0600e+02, 1.3000e+01, 1.0000e+00, 1.0000e+00, 0.0000e+00, 7.9000e+01]])
mtx_jiayan = mtx_jiayan / mtx_jiayan.sum() * 100

fig, axs = plt.subplots(figsize=(17, 6), nrows=1,ncols=2)

sns.heatmap(mtx_jiayan, annot=True, fmt=".2f", cmap="crest", norm=LogNorm(), ax=axs[0], square=True)
axs[0].set_title('Jiayan', fontsize=16)
ticks = ["none",] + list("，。、？！：；")
axs[0].set_xticks([x + 0.5 for x in range(8)])
axs[0].set_xticklabels(ticks, font='SimHei', fontsize=16)
axs[0].set_yticks([x + 0.5 for x in range(8)])
axs[0].set_yticklabels(ticks, font='SimHei', fontsize=16, rotation=360, horizontalalignment='right')
axs[0].set(xlabel='ground truth', ylabel='prediction')

sns.heatmap(mtx_shanduan, annot=True, fmt=".2f", cmap="crest", norm=LogNorm(), ax=axs[1], square=True)
axs[1].set_title('Shanduan (ours)', fontsize=16)
ticks = ["none",] + list("，。、？！：；")
axs[1].set_xticks([x + 0.5 for x in range(8)])
axs[1].set_xticklabels(ticks, font='SimHei', fontsize=16)
axs[1].set_yticks([x + 0.5 for x in range(8)])
axs[1].set_yticklabels(ticks, font='SimHei', fontsize=16, rotation=360, horizontalalignment='right')
axs[1].set(xlabel='ground truth', ylabel='prediction')

axs[0].add_patch(Rectangle((1, 1), 2, 2, fill=False, edgecolor='orange', lw=3))
axs[1].add_patch(Rectangle((1, 1), 2, 2, fill=False, edgecolor='orange', lw=3))

fig.savefig('punctuation_heatmaps.png', dpi=400, bbox_inches='tight')

######################################################################################################

mtx_shanduan = np.array([[0.792, 0.009], [0.011, 0.188]]) * 100
mtx_jiayan = np.array([[0.777, 0.027], [0.027, 0.169]]) * 100

fig, axs = plt.subplots(figsize=(8, 3), nrows=1, ncols=2)

sns.heatmap(mtx_jiayan, annot=True, fmt=".1f", cmap="crest", ax=axs[0], square=True)
axs[0].set_title('Jiayan', fontsize=16)
axs[0].set(xlabel='ground truth', ylabel='prediction')

sns.heatmap(mtx_shanduan, annot=True, fmt=".1f", cmap="crest", ax=axs[1], square=True)
axs[1].set_title('Shanduan (ours)', fontsize=16)
axs[1].set(xlabel='ground truth', ylabel='prediction')



fig.savefig('segmentation_heatmaps.png', dpi=400, bbox_inches='tight')