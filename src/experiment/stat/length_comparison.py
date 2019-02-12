import matplotlib.pyplot as plt
import numpy as np
import os

root_folder = os.path.abspath('../../../resource/prediction_result/best_result')
ch_rnn_1y_micro = 0.902, 0.891, 0.876, 0.89, 0.891, 0.895, 0.886, 0.891
fh_rnn_1y_micro = 0.905, 0.896, 0.88, 0.895, 0.896, 0.9, 0.888, 0.893
rnn_1y_micro = 0.89, 0.88, 0.866, 0.88, 0.881, 0.885, 0.875, 0.878
lr_1y_micro = 0.84, 0.83, 0.823, 0.83, 0.826, 0.833, 0.828, 0.805

ch_rnn_3m_micro = 0.901, 0.9, 0.875, 0.895, 0.874, 0.896, 0.891, 0.909
fh_rnn_3m_micro = 0.908, 0.902, 0.878, 0.897, 0.869, 0.899, 0.882, 0.907
rnn_3m_micro = 0.895, 0.894	, 0.866, 0.886, 0.861, 0.887, 0.875, 0.886
lr_3m_micro = 0.831, 0.824, 0.812, 0.815, 0.799, 0.798, 0.798, 0.794

ch_rnn_1y_macro = 0.848, 0.834, 0.822, 0.832, 0.834, 0.839, 0.833, 0.851
fh_rnn_1y_macro = 0.852	, 0.84, 0.834, 0.836, 0.844, 0.847, 0.839, 0.854
rnn_1y_macro = 0.831, 0.819	, 0.814, 0.816, 0.822, 0.827, 0.821	, 0.807
lr_1y_macro = 0.732, 0.718, 0.709, 0.721, 0.711, 0.728, 0.72, 0.708

ch_rnn_3m_macro = 0.827, 0.823, 0.815, 0.842, 0.829, 0.822, 0.824, 0.859
fh_rnn_3m_macro = 0.825, 0.823, 0.823, 0.836, 0.805, 0.825, 0.815, 0.826
rnn_3m_macro = 0.822, 0.822	, 0.814, 0.835, 0.785, 0.799, 0.798, 0.795
lr_3m_macro = 0.736, 0.732, 0.727, 0.733, 0.711, 0.7, 0.703, 0.687

x_axis = np.arange(3, 11, 1)

plt.rc('font', family='Times New Roman')
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(hspace=0, wspace=0)
l1 = axs[0].plot(x_axis, lr_3m_micro, '-', marker='o',  markersize=6, ms=4, color='green', label='Micro-LR',
                 linewidth=1)
l2 = axs[0].plot(x_axis, rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='blue', label='Micro-RNN',
                 linewidth=1)
l3 = axs[0].plot(x_axis, fh_rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='red', label='Micro-FH-RNN',
                 linewidth=1)
l4 = axs[0].plot(x_axis, ch_rnn_3m_micro, '-', marker='o',  markersize=6, ms=4, color='orange', label='Micro-CH-RNN',
                 linewidth=1)
l5 = axs[0].plot(x_axis, lr_3m_macro, ':', marker='^',  markersize=6, ms=4, color='green', label='Macro-LR',
                 linewidth=1)
l6 = axs[0].plot(x_axis, rnn_3m_macro, ':', marker='^',  markersize=6, ms=4, color='blue', label='Macro-RNN',
                 linewidth=1)
l7 = axs[0].plot(x_axis, fh_rnn_3m_macro, ':', marker='^',  markersize=6, ms=4, color='red', label='Macro-FH-RNN',
                 linewidth=1)
l8 = axs[0].plot(x_axis, ch_rnn_3m_macro, ':', marker='^',  markersize=6, ms=4, color='orange', label='Macro-CH-RNN',
                 linewidth=1)

axs[1].plot(x_axis, lr_1y_micro, '-', marker='o',  markersize=6, ms=4, color='green', label='Micro-LR', linewidth=1)
axs[1].plot(x_axis, rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='blue', label='Micro-RNN', linewidth=1)
axs[1].plot(x_axis, fh_rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='red', label='Micro-FH-RNN',
            linewidth=1)
axs[1].plot(x_axis, ch_rnn_1y_micro, '-', marker='o',  markersize=6, ms=4, color='orange', label='Micro-CH-RNN',
            linewidth=1)
axs[1].plot(x_axis, lr_1y_macro, ':', marker='^',  markersize=6, ms=4, color='green', label='Macro-LR', linewidth=1)
axs[1].plot(x_axis, rnn_1y_macro, ':', marker='^',  markersize=6, ms=4, color='blue', label='Macro-RNN', linewidth=1)
axs[1].plot(x_axis, fh_rnn_1y_macro, ':', marker='^',  markersize=6, ms=4, color='red', label='Macro-FH-RNN',
            linewidth=1)
axs[1].plot(x_axis, ch_rnn_1y_macro, ':', marker='^',  markersize=6, ms=4, color='orange', label='Macro-CH-RNN',
            linewidth=1)

axs[0].set_yticks(np.arange(0.7, 0.90, 0.1))
axs[0].set_ylim(0.68, 0.92)
axs[1].set_yticks(np.arange(0.7, 0.90, 0.1))
axs[1].set_ylim(0.68, 0.92)

plt.setp(axs[1].get_yticklabels(), visible=False)
plt.setp(axs[1].get_yaxis(), visible=False)

axs[0].set_title('3M Average Performance')
axs[1].set_title('1Y Average Performance')

axs[0].set_xlabel('Trajectory Length', fontsize=12, fontweight='bold')
axs[1].set_xlabel('Trajectory Length', fontsize=12, fontweight='bold')
axs[0].set_ylabel('AUC', fontsize=12, fontweight='bold')

legend = fig.legend([l1, l2, l3, l4, l5, l6, l7, l8],
                    labels=['Micro-AUC-LR', 'Micro-AUC-RNN', 'Micro-AUC-CH-RNN', 'Micro-AUC-CH-RNN',
                            'Macro-AUC-LR', 'Macro-AUC-RNN', 'Macro-AUC-FH-RNN', 'Macro-AUC-CH-RNN'],
                    borderaxespad=0,
                    ncol=1,
                    fontsize=10,
                    loc='right',
                    bbox_to_anchor=[1.17, 0.5],
                    )
plt.show()
fig.savefig(os.path.join(root_folder, 'length_comparison'), bbox_inches='tight', bbox_extra_artists=(legend,))
