import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from textwrap import wrap
from time import time
import os


def main():
    root_path = os.path.abspath('../../../resource/prediction_result/tsne/')
    event_list = ['一年死亡']
    eng_event_dict = {'一年死亡': '1Y-Death',
                      '一年再血管化手术': '1Y-Revascularization',
                      '三月死亡': '3M-Death',
                      '三月再血管化手术': '3M-Revascularization'}
    plt.rc('font', family='Times New Roman')
    (fig, subplots) = plt.subplots(1, 3, figsize=(8, 3))

    representation_dict = read_data(root_path, event_list)

    for i, event in enumerate(event_list):
        if i == 0:
            index = [0, 0]
        elif i == 1:
            index = [0, 3]
        elif i == 2:
            index = [1, 0]
        elif i == 3:
            index = [1, 3]
        else:
            raise ValueError('')
        model = 'CH-RNN'
        label = representation_dict[model][event][1]
        representation = representation_dict[model][event][0]
        ax = subplots[index[1]]
        tsne_analysis(model, event, label, representation, ax, eng_event_dict)
    for i, event in enumerate(event_list):
        if i == 0:
            index = [0, 1]
        elif i == 1:
            index = [0, 4]
        elif i == 2:
            index = [1, 1]
        elif i == 3:
            index = [1, 4]
        else:
            raise ValueError('')
        model = 'FH-RNN'
        label = representation_dict[model][event][1]
        representation = representation_dict[model][event][0]
        ax = subplots[index[1]]
        tsne_analysis(model, event, label, representation, ax, eng_event_dict)
    for i, event in enumerate(event_list):
        if i == 0:
            index = [0, 2]
        elif i == 1:
            index = [0, 5]
        elif i == 2:
            index = [1, 2]
        elif i == 3:
            index = [1, 5]
        else:
            raise ValueError('')
        model = 'RNN'
        label = representation_dict[model][event][1]
        representation = representation_dict[model][event][0]
        ax = subplots[index[1]]
        tsne_analysis(model, event, label, representation, ax, eng_event_dict)
    plt.show()


def tsne_analysis(model, event, label, representation, ax, eng_event_dict):
    tsne = manifold.TSNE(n_components=2, init='random', perplexity=5)
    red = label == 1
    green = label == 0
    t0 = time()
    embedding = tsne.fit_transform(representation)
    t1 = time()
    print("event {}, in {} sec".format(eng_event_dict[event], t1 - t0))
    ax.set_title('{} {}'.format(model, eng_event_dict[event]), fontsize=10)
    ax.scatter(embedding[green, 0], embedding[green, 1], c="g", s=1)
    ax.scatter(embedding[red, 0], embedding[red, 1], c="r", s=1.5)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_yaxis(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_xaxis(), visible=False)
    return embedding


def read_data(file_path, event_list):
    representation_dict = {'FH-RNN': dict(), 'CH-RNN': dict(), 'RNN': dict()}
    for event in event_list:
        representation_dict['FH-RNN'][event] = \
            [np.load(os.path.join(file_path, 'hidden_state_fused_hawkes_rnn_gru_{}.npy'.format(event))),
             np.load(os.path.join(file_path, 'label_fused_hawkes_rnn_gru_{}.npy'.format(event)))]
        representation_dict['CH-RNN'][event] = \
            [np.load(os.path.join(file_path, 'hidden_state_concat_hawkes_rnn_gru_{}.npy'.format(event))),
             np.load(os.path.join(file_path, 'label_concat_hawkes_rnn_gru_{}.npy'.format(event)))]
        representation_dict['RNN'][event] = \
            [np.load(os.path.join(file_path, 'hidden_state_vanilla_rnn_gru_{}.npy'.format(event))),
             np.load(os.path.join(file_path, 'label_vanilla_rnn_gru_{}.npy'.format(event)))]
    return representation_dict


if __name__ == '__main__':
    main()
