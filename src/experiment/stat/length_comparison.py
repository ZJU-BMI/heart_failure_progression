import matplotlib.pyplot as plt
import numpy as np
import os

root_folder = os.path.abspath('../../../resource/prediction_result/best_result')
ch_rnn_1y = [0.8496, 0.8542, 0.851, 0.8507, 0.8391, 0.8667, 0.8466, 0.8561]
fh_rnn_1y = [0.8458, 0.8547, 0.8641, 0.8579, 0.8619, 0.8583, 0.8556, 0.866]
rnn_1y = [0.8435, 0.8134, 0.8243, 0.8184, 0.8205, 0.8178, 0.8169, 0.8301]
lr_1y = [0.7185, 0.7421, 0.7307, 0.6908, 0.7315, 0.7339, 0.7218, 0.7308]

ch_rnn_3m = [0.8446, 0.8737, 0.8736, 0.8825, 0.8579, 0.8586, 0.8679, 0.8909]
fh_rnn_3m = [0.8482, 0.8704, 0.8817, 0.8791, 0.8438, 0.871, 0.8259, 0.8612]
rnn_3m = [0.8387, 0.8307, 0.8348, 0.8372, 0.8051, 0.8291, 0.7895, 0.8185]
lr_3m = [0.7423, 0.7467, 0.74, 0.6826, 0.7176, 0.7225, 0.7538, 0.7255]

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
x_axis = np.arange(3, 11, 1)


plt.rc('font', family='Times New Roman')
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
fig.subplots_adjust(hspace=0, wspace=0)
l1 = axs[0].plot(x_axis, lr_3m, 'o-', ms=4, color='green', label='LR', linewidth=1)
l2 = axs[0].plot(x_axis, rnn_3m, 'o-', ms=4, color='blue', label='RNN', linewidth=1)
l3 = axs[0].plot(x_axis, fh_rnn_3m, 'o-', ms=4, color='red', label='FH-RNN', linewidth=1)
l4 = axs[0].plot(x_axis, ch_rnn_3m, 'o-', ms=4, color='orange', label='CH-RNN', linewidth=1)

axs[1].plot(x_axis, lr_1y, 'o-', ms=4, color='green', label='LR', linewidth=1)
axs[1].plot(x_axis, rnn_1y, 'o-', ms=4, color='blue', label='RNN', linewidth=1)
axs[1].plot(x_axis, fh_rnn_1y, 'o-', ms=4, color='red', label='FH-RNN', linewidth=1)
axs[1].plot(x_axis, ch_rnn_1y, 'o-', ms=4, color='orange', label='CH-RNN', linewidth=1)

axs[0].set_yticks(np.arange(0.7, 0.901, 0.1))
plt.setp(axs[1].get_yticklabels(), visible=False)
plt.setp(axs[1].get_yaxis(), visible=False)
axs[0].set_title('3M Avg. Performance')
axs[1].set_title('1Y Avg. Performance')
axs[0].set_ylim(0.67, 0.901)
axs[0].set_ylim(0.67, 0.901)
axs[1].set_ylim(0.67, 0.901)
axs[0].set_xlabel('Visit Count', fontsize=10, fontweight='bold')
axs[1].set_xlabel('Visit Count', fontsize=10, fontweight='bold')
axs[0].set_ylabel('AUC', fontsize=10, fontweight='bold')

legend = fig.legend([l1, l2, l3, l4],
                    labels=['LR', 'RNN', 'FH-RNN', 'CH-RNN'],
                    borderaxespad=0,
                    ncol=4,
                    fontsize=10,
                    loc='center',
                    bbox_to_anchor=[0.52, 1.05],
                    )

plt.show()
fig.savefig(os.path.join(root_folder, 'length_comparison'), bbox_inches='tight', bbox_extra_artists=(legend,))