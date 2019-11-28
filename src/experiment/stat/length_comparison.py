import matplotlib.pyplot as plt
import numpy as np
import os


plt.rc('font', family='Times New Roman')
fig, axs = plt.subplots(1, 4, figsize=(19.5, 5.25))
fig.subplots_adjust(hspace=0, wspace=0)

root_folder = os.path.abspath('../../../resource/prediction_result/best_result')
"""
ch_rnn_1y_micro = 0.86, 0.869, 0.884, 0.867, 0.857, 0.872, 0.861
fh_rnn_1y_micro = 0.873, 0.876, 0.889, 0.872, 0.863, 0.877, 0.865
rnn_1y_micro = 0.850, 0.856, 0.869, 0.852, 0.843, 0.857, 0.844
lr_1y_micro = 0.828, 0.828, 0.828, 0.828, 0.828, 0.828, 0.828
hp_1y_micro = 0.73033, 0.73642, 0.73839, 0.73838, 0.73886, 0.73923, 0.73927

ch_rnn_3m_micro = 0.877, 0.875, 0.882, 0.872, 0.857, 0.882, 0.883
fh_rnn_3m_micro = 0.880, 0.885, 0.889, 0.873, 0.861, 0.881, 0.882
rnn_3m_micro = 0.855, 0.861, 0.866, 0.854, 0.842, 0.859, 0.864
lr_3m_micro = 0.798, 0.798, 0.798, 0.798, 0.798, 0.798, 0.798
hp_3m_micro = 0.67904, 0.68450, 0.68762, 0.68738, 0.68807, 0.68874, 0.68883
"""

ch_rnn_1y_macro = 0.838, 0.843, 0.791, 0.803, 0.824, 0.859, 0.852, 0.844, 0.860
fh_rnn_1y_macro = 0.838, 0.854, 0.813, 0.834, 0.842, 0.866, 0.863, 0.867, 0.870
rnn_1y_macro = 0.821, 0.827, 0.806, 0.810, 0.820, 0.825, 0.827, 0.840, 0.836
cnn_1y_macro = 0.753, 0.762, 0.774, 0.785, 0.795, 0.793, 0.799, 0.801, 0.803
lr_1y_macro = 0.736, 0.736, 0.736, 0.736, 0.736, 0.736, 0.736, 0.736, 0.736
hp_1y_macro = 0.706, 0.710, 0.714, 0.715, 0.716, 0.717, 0.717, 0.717, 0.718

ch_rnn_3m_macro = 0.826, 0.834, 0.819, 0.817, 0.811, 0.844, 0.863, 0.875, 0.883
fh_rnn_3m_macro = 0.845, 0.855, 0.848, 0.837, 0.850, 0.864, 0.870, 0.864, 0.868
rnn_3m_macro = 0.808, 0.796, 0.785, 0.787, 0.819, 0.834, 0.842, 0.838, 0.845
cnn_3m_macro = 0.796, 0.803, 0.799, 0.818, 0.824, 0.831, 0.827, 0.836, 0.841
lr_3m_macro = 0.715, 0.715, 0.715, 0.715, 0.715, 0.715, 0.715, 0.715, 0.715
hp_3m_macro = 0.715, 0.718, 0.719, 0.719, 0.719, 0.719, 0.720, 0.720, 0.721

x_axis = np.arange(4, 13, 1)
"""
l1 = axs[0][0].plot(x_axis, lr_3m_micro, '-', marker='o',  markersize=6, ms=4, color='green', label='Micro-LR',
                    linewidth=1)
l2 = axs[0][0].plot(x_axis, rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='blue', label='Micro-RNN',
                    linewidth=1)
l3 = axs[0][0].plot(x_axis, fh_rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='red', label='Micro-FH-RNN',
                    linewidth=1)
l4 = axs[0][0].plot(x_axis, ch_rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='orange', label='Micro-CH-RNN',
                    linewidth=1)
l5 = axs[0][0].plot(x_axis, hp_3m_micro, '-', marker='o',  markersize=6, ms=4, color='purple', label='Micro-HP',
                    linewidth=1)
"""
l6 = axs[0].plot(x_axis, lr_3m_macro, ':', marker='o',  markersize=6, ms=4, color='green', label='Macro-LR',
                 linewidth=1)
l7 = axs[0].plot(x_axis, rnn_3m_macro, ':', marker='x',  markersize=6, ms=4, color='blue', label='Macro-RNN',
                 linewidth=1)
l8 = axs[0].plot(x_axis, fh_rnn_3m_macro, ':', marker='d',  markersize=6, ms=4, color='red', label='Macro-FH-RNN',
                 linewidth=1)
l9 = axs[0].plot(x_axis, cnn_3m_macro, ':', marker='P',  markersize=6, ms=4, color='black', label='Macro-CNN',
                 linewidth=1)
l10 = axs[0].plot(x_axis, ch_rnn_3m_macro, ':', marker='^',  markersize=6, ms=4, color='orange', label='Macro-CH-RNN',
                  linewidth=1)
l11 = axs[0].plot(x_axis, hp_3m_macro, ':', marker='*',  markersize=6, ms=4, color='purple',
                  label='Macro-HP', linewidth=1)

"""
axs[0][1].plot(x_axis, lr_1y_micro, '-', marker='o',  markersize=6, ms=4, color='green', label='Micro-LR', linewidth=1)
axs[0][1].plot(x_axis, rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='blue', label='Micro-RNN', linewidth=1)
axs[0][1].plot(x_axis, fh_rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='red', label='Micro-FH-RNN',
               linewidth=1)
axs[0][1].plot(x_axis, ch_rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='orange', label='Micro-CH-RNN',
               linewidth=1)
axs[0][1].plot(x_axis, hp_1y_micro, '-', marker='o',  markersize=6, ms=4, color='purple', label='Micro-HP',
               linewidth=1)
"""
axs[1].plot(x_axis, lr_1y_macro, ':', marker='o',  markersize=6, ms=4, color='green', label='Macro-LR', linewidth=1)
axs[1].plot(x_axis, rnn_1y_macro, ':', marker='x',  markersize=6, ms=4, color='blue', label='Macro-RNN', linewidth=1)
axs[1].plot(x_axis, fh_rnn_1y_macro, ':', marker='d',  markersize=6, ms=4, color='red', label='Macro-FH-RNN',
            linewidth=1)
axs[1].plot(x_axis, cnn_1y_macro, ':', marker='P',  markersize=6, ms=4, color='black', label='Macro-CNN',
            linewidth=1)
axs[1].plot(x_axis, ch_rnn_1y_macro, ':', marker='^',  markersize=6, ms=4, color='orange', label='Macro-CH-RNN',
            linewidth=1)
axs[1].plot(x_axis, hp_1y_macro, ':', marker='*',  markersize=6, ms=4, color='purple', label='Macro-HP',
            linewidth=1)


axs[0].set_yticks(np.arange(0.65, 0.9, 0.1))
axs[0].set_ylim(0.65, 0.91)
axs[1].set_ylim(0.65, 0.91)
axs[2].set_ylim(0.65, 0.91)
axs[3].set_ylim(0.65, 0.91)

plt.setp(axs[1].get_yticklabels(), visible=False)
plt.setp(axs[1].get_yaxis(), visible=False)

axs[0].set_title('3M Performance', fontsize=18)
axs[1].set_title('1Y Performance', fontsize=18)

axs[0].set_xlabel('Length of the Observational Window (s)\n(a)', fontsize=18, fontweight='bold')
axs[1].set_xlabel('Length of the Observational Window (s)\n(b)', fontsize=18, fontweight='bold')
axs[0].set_ylabel('Avg.  AUC', fontsize=18, fontweight='bold')
axs[1].set_ylabel('Avg.  AUC', fontsize=18, fontweight='bold')

plt.setp(axs[2].get_yticklabels(), visible=False)
plt.setp(axs[2].get_yaxis(), visible=False)
plt.setp(axs[3].get_yticklabels(), visible=False)
plt.setp(axs[3].get_yaxis(), visible=False)
axs[2].set_title('3M Performance', fontsize=18)
axs[3].set_title('1Y Performance', fontsize=18)

axs[2].set_xlabel('Data Fraction (k)\n(c)', fontsize=18, fontweight='bold')
axs[3].set_xlabel('Data Fraction (k)\n(d)', fontsize=18, fontweight='bold')
axs[2].set_ylabel('Avg.  AUC', fontsize=18, fontweight='bold')
axs[3].set_ylabel('Avg.  AUC', fontsize=18, fontweight='bold')

legend = fig.legend([l6, l7, l8,  l9, l10, l11],
                    # [l1, l10, l5, l2, l6, l3, l7, l4, l8, l9],
                    labels=['LR', 'RNN', 'FH-RNN', 'CNN', 'CH-RNN', 'HP'],
                    borderaxespad=0,
                    ncol=6,
                    fontsize=18,
                    loc='center',
                    bbox_to_anchor=[0.5, 1.03],
                    )
"""
ch_rnn_1y_micro = 0.829, 0.84, 0.848, 0.852, 0.859, 0.866, 0.869, 0.874, 0.891
fh_rnn_1y_micro = 0.842, 0.853, 0.858, 0.862, 0.865, 0.869, 0.872, 0.876, 0.893
rnn_1y_micro = 0.822, 0.831, 0.835, 0.841, 0.845, 0.849, 0.851, 0.865, 0.878
lr_1y_micro = 0.757, 0.771, 0.777, 0.786, 0.791, 0.794, 0.798, 0.802, 0.805
hp_1y_micro = 0.73381, 0.73970, 0.73965, 0.73744, 0.73929, 0.73882, 0.73736, 0.73832, 0.73802

ch_rnn_3m_micro = 0.769, 0.814, 0.841, 0.853, 0.866, 0.879, 0.889, 0.895, 0.909
fh_rnn_3m_micro = 0.786, 0.827, 0.853, 0.864, 0.869, 0.877, 0.885, 0.890, 0.905
rnn_3m_micro = 0.765, 0.81, 0.831, 0.846, 0.853, 0.858, 0.865, 0.874, 0.886
lr_3m_micro = 0.744, 0.759, 0.765, 0.772, 0.779, 0.782, 0.789, 0.789, 0.794
hp_3m_micro = 0.67989, 0.69089, 0.69262, 0.68464, 0.68515, 0.68842, 0.68991, 0.68665, 0.68701
"""

ch_rnn_1y_macro = 0.750, 0.774, 0.790, 0.809, 0.822, 0.837, 0.843, 0.853, 0.859
fh_rnn_1y_macro = 0.775, 0.797, 0.812, 0.827, 0.835, 0.845, 0.851, 0.859, 0.863
rnn_1y_macro = 0.745, 0.764, 0.774, 0.786, 0.787, 0.794, 0.809, 0.817, 0.825
cnn_1y_macro = 0.702, 0.715, 0.739, 0.744, 0.752, 0.763, 0.770, 0.783, 0.790
lr_1y_macro = 0.667, 0.681, 0.685, 0.698, 0.701, 0.705, 0.712, 0.716, 0.720
hp_1y_macro = 0.707350, 0.713161, 0.711897, 0.718766, 0.720039, 0.71973, 0.714978, 0.716618, 0.715431

ch_rnn_3m_macro = 0.695, 0.733, 0.770, 0.793, 0.817, 0.837, 0.855, 0.878, 0.898
fh_rnn_3m_macro = 0.703, 0.749, 0.775, 0.799, 0.819, 0.832, 0.841, 0.854, 0.874
rnn_3m_macro = 0.674, 0.724, 0.737, 0.764, 0.776, 0.784, 0.798, 0.813, 0.834
cnn_3m_macro = 0.680, 0.719, 0.744, 0.762, 0.777, 0.798, 0.805, 0.818, 0.825
lr_3m_macro = 0.656, 0.675, 0.68, 0.685, 0.69, 0.697, 0.703, 0.710, 0.715
hp_3m_macro = 0.730792, 0.729634, 0.736729, 0.731376, 0.737795, 0.736569, 0.737778, 0.737474, 0.736

x_axis = np.arange(0.2, 1.01, 0.1)

"""
axs[1][0].plot(x_axis, lr_3m_micro, '-', marker='o',  markersize=6, ms=4, color='green', label='Micro-LR', linewidth=1)
axs[1][0].plot(x_axis, rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='blue', label='Micro-RNN', linewidth=1)
axs[1][0].plot(x_axis, fh_rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='red', label='Micro-FH-RNN',
linewidth=1)
axs[1][0].plot(x_axis, ch_rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='orange', label='Micro-CH-RNN',
               linewidth=1)
axs[1][0].plot(x_axis, hp_3m_micro, '-', marker='o',  markersize=6, ms=4, color='purple', label='Micro-HP',
               linewidth=1)
"""
axs[2].plot(x_axis, lr_3m_macro, ':', marker='o',  markersize=6, ms=4, color='green', label='Macro-LR', linewidth=1)
axs[2].plot(x_axis, rnn_3m_macro, ':', marker='x',  markersize=6, ms=4, color='blue', label='Macro-RNN', linewidth=1)
axs[2].plot(x_axis, fh_rnn_3m_macro, ':', marker='d',  markersize=6, ms=4, color='red', label='Macro-FH-RNN',
            linewidth=1)
axs[2].plot(x_axis, ch_rnn_3m_macro, ':', marker='^',  markersize=6, ms=4, color='orange', label='Macro-CH-RNN',
            linewidth=1)
axs[2].plot(x_axis, cnn_3m_macro, ':', marker='P',  markersize=6, ms=4, color='black', label='Macro-CNN',
            linewidth=1)
axs[2].plot(x_axis, hp_3m_macro, ':', marker='*',  markersize=6, ms=4, color='purple', label='Micro-HP',
            linewidth=1)
"""
axs[1][1].plot(x_axis, lr_1y_micro, '-', marker='o',  markersize=6, ms=4, color='green', label='Micro-LR', linewidth=1)
axs[1][1].plot(x_axis, rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='blue', label='Micro-RNN', linewidth=1)
axs[1][1].plot(x_axis, fh_rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='red', label='Micro-FH-RNN',
               linewidth=1)
axs[1][1].plot(x_axis, ch_rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='orange', label='Micro-CH-RNN',
               linewidth=1)
axs[1][1].plot(x_axis, hp_1y_micro, '-', marker='o',  markersize=6, ms=4, color='purple', label='Macro-HP',
               linewidth=1)
"""
axs[3].plot(x_axis, lr_1y_macro, ':', marker='o',  markersize=6, ms=4, color='green', label='Macro-LR', linewidth=1)
axs[3].plot(x_axis, rnn_1y_macro, ':', marker='x',  markersize=6, ms=4, color='blue', label='Macro-RNN', linewidth=1)
axs[3].plot(x_axis, fh_rnn_1y_macro, ':', marker='d',  markersize=6, ms=4, color='red', label='Macro-FH-RNN',
            linewidth=1)
axs[3].plot(x_axis, ch_rnn_1y_macro, ':', marker='^',  markersize=6, ms=4, color='orange', label='Macro-CH-RNN',
            linewidth=1)
axs[3].plot(x_axis, cnn_1y_macro, ':', marker='P',  markersize=6, ms=4, color='black', label='Macro-CNN',
            linewidth=1)
axs[3].plot(x_axis, hp_1y_macro, ':', marker='*',  markersize=6, ms=4, color='purple', label='Macro-HP',
            linewidth=1)

plt.show()
fig.savefig(os.path.join(root_folder, 'length_comparison'), bbox_inches='tight')
