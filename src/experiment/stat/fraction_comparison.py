import matplotlib.pyplot as plt
import numpy as np
import os

root_folder = os.path.abspath('../../../resource/prediction_result/best_result')
ch_rnn_1y_micro = 0.829, 0.84, 0.848, 0.852, 0.859, 0.866, 0.869, 0.874, 0.891
fh_rnn_1y_micro = 0.842, 0.853, 0.858, 0.862, 0.865, 0.869, 0.872, 0.876, 0.893
rnn_1y_micro = 0.822, 0.831, 0.835, 0.841, 0.845, 0.849, 0.851, 0.865, 0.878
lr_1y_micro = 0.757, 0.771, 0.777, 0.786, 0.791, 0.794, 0.798, 0.802, 0.805

ch_rnn_3m_micro = 0.769, 0.814, 0.841, 0.853, 0.866, 0.879, 0.889, 0.895, 0.909
fh_rnn_3m_micro = 0.786, 0.827, 0.853, 0.864, 0.869, 0.877, 0.885, 0.890, 0.905
rnn_3m_micro = 0.765, 0.81, 0.831, 0.846, 0.853, 0.858, 0.865, 0.874, 0.886
lr_3m_micro = 0.744, 0.759, 0.765, 0.772, 0.779, 0.782, 0.789, 0.789, 0.794

ch_rnn_1y_macro = 0.739, 0.761, 0.778, 0.786, 0.799, 0.813, 0.82, 0.829, 0.851
fh_rnn_1y_macro = 0.761, 0.782, 0.798, 0.803, 0.809, 0.819, 0.826, 0.833, 0.854
rnn_1y_macro = 0.731, 0.749, 0.759, 0.771, 0.781, 0.788, 0.794, 0.802, 0.807
lr_1y_macro = 0.657, 0.672, 0.675, 0.689, 0.692, 0.695, 0.7, 0.703, 0.708

ch_rnn_3m_macro = 0.695, 0.733, 0.76, 0.773, 0.793, 0.817, 0.825, 0.838, 0.865
fh_rnn_3m_macro = 0.703, 0.749, 0.775, 0.789, 0.799, 0.812, 0.821, 0.834, 0.846
rnn_3m_macro = 0.674, 0.724, 0.737, 0.764, 0.776, 0.784, 0.798, 0.813, 0.815
lr_3m_macro = 0.636, 0.655, 0.66, 0.665, 0.67, 0.677, 0.683, 0.68, 0.687

x_axis = np.arange(0.2, 1.01, 0.1)

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

axs[0].set_yticks(np.arange(0.65, 0.96, 0.1))
axs[0].set_ylim(0.62, 0.96)
axs[1].set_yticks(np.arange(0.65, 0.96, 0.1))
axs[1].set_ylim(0.62, 0.96)

plt.setp(axs[1].get_yticklabels(), visible=False)
plt.setp(axs[1].get_yaxis(), visible=False)

axs[0].set_title('3M Average Performance')
axs[1].set_title('1Y Average Performance')

axs[0].set_xlabel('Data Fraction', fontsize=12, fontweight='bold')
axs[1].set_xlabel('Data Fraction', fontsize=12, fontweight='bold')
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
fig.savefig(os.path.join(root_folder, 'fraction_comparison'), bbox_inches='tight', bbox_extra_artists=(legend,))
