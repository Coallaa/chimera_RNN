import logging

import torch
from matplotlib import pyplot as plt
import numpy as np
import pickle


class Output:
    def __init__(self, path, finish_train, input_file, predict_file, group_info, compare_round):
        self.path = path
        # self.label = ['before_finish', 'just_finish', 'after_' + str(compare_round)]
        # self.start_plot = [finish_train, finish_train, finish_train + compare_round]
        self.label = ['just_finish', 'after_' + str(compare_round)]
        self.start_plot = [finish_train, finish_train + compare_round]
        self.group_info = group_info
        self.num_nodes = None
        self.cos_out = None
        self.sin_out = None
        self.cos_tar = None
        self.sin_tar = None
        self.total_l2 = None
        self.diff = None
        self.out = None
        self.tar = None
        self.load_data(input_file, predict_file)
        self.statistics = []
        logging.info('Output has been initialized.')

    def load_data(self, input_file, predict_file):
        # load network output
        data = pickle.load(open(self.path + predict_file, 'rb'))
        valid_len = data.shape[0]
        self.num_nodes = data.shape[1]
        self.cos_out = np.cos(data)
        self.sin_out = np.sin(data)
        self.out = data

        # load input as target
        target = pickle.load(open(input_file, 'rb'))
        target = np.transpose(target[:, :valid_len])
        self.cos_tar = np.cos(target)
        self.sin_tar = np.sin(target)
        self.tar = target
        logging.info('Data are loaded from error.pkl')

    def cal_l2_error(self):
        self.diff = np.multiply(self.cos_out - self.cos_tar, self.cos_out - self.cos_tar) + \
               np.multiply(self.sin_out - self.sin_tar, self.sin_out - self.sin_tar)
        self.total_l2 = np.sqrt(np.sum(self.diff, axis=1))
        logging.info('l2 error is calculated.')

    def plot_figure_l2_error(self, name, start_plot, total_l2, window):
        if name == 'before_finish':
            start_plot = start_plot - window
        plt.scatter(np.array([i for i in range(start_plot, start_plot + window)]),
                    total_l2[start_plot: start_plot + window], s=1)
        plt.xlabel('round number')
        plt.ylabel('l2 trigonometric error')
        plt.savefig(self.path + 'l2_' + name + '.jpg')
        plt.clf()

    def plot_figure_tar_out(self, name, start_plot, n, cos_tar, sin_tar, cos_out, sin_out, window):
        if name == 'before_finish':
            start_plot = start_plot - window
        labels = []
        for i in range(n):
            labels.append(('cos(theta_' + str(i) + '_hat)', 'cos(theta_' + str(i) + ')'))
        for i in range(n):
            labels.append(('sin(theta_' + str(i) + '_hat)', 'sin(theta_' + str(i) + ')'))

        s = 2
        c = '#FF8C00'
        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(2.1*n, 8), dpi=400, tight_layout=True)
        x_input = np.array([i for i in range(start_plot, start_plot + window)])

        for i in range(n):
            axes[i][0].scatter(x_input, cos_tar[start_plot: start_plot + window, i], label=labels[i][1], s=s, c=c)
            axes[i][0].scatter(x_input, cos_out[start_plot: start_plot + window, i], label=labels[i][0], s=s, c='blue')
            axes[i][0].legend(loc='upper left', fontsize=10)

        for i in range(n):
            j = i + 6
            axes[i][1].scatter(x_input, sin_tar[start_plot: start_plot + window, i], label=labels[j][1], s=s, c=c)
            axes[i][1].scatter(x_input, sin_out[start_plot: start_plot + window, i], label=labels[j][0], s=s, c='blue')
            axes[i][1].legend(loc='upper left', fontsize=10)

        plt.xlabel('round number')
        plt.ylabel('network output origin')
        plt.savefig(self.path + 'tar_out_' + name + '.jpg')
        plt.clf()

    def plot_angle_tar_out(self, name, start_plot, n, window):
        if name == 'before_finish':
            start_plot = start_plot - window
        labels = []
        for i in range(n):
            labels.append(('theta_' + str(i) + '_hat', 'theta_' + str(i)))

        s = 2
        c = '#FF8C00'
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(1.05*n, 8), dpi=400, tight_layout=True)
        x_input = np.array([i for i in range(start_plot, start_plot + window)])

        for i in range(n):
            axes[i].scatter(x_input, self.out[start_plot: start_plot + window, i], label=labels[i][1], s=s, c=c)
            axes[i].scatter(x_input, self.tar[start_plot: start_plot + window, i], label=labels[i][0], s=s, c='blue')
            axes[i].legend(loc='upper left', fontsize=10)

        plt.xlabel('round number')
        plt.ylabel('network output angle')
        plt.savefig(self.path + 'angle_tar_out_' + name + '.jpg')
        plt.clf()

    def plot_figures(self, tar_out_window, l2_error_window):
        # self.plot_long(self.start_plot[1], self.num_nodes)
        for (n, s) in zip(self.label, self.start_plot):
            self.plot_figure_tar_out(n, s, self.num_nodes, self.cos_tar, self.sin_tar, self.cos_out, self.sin_out, tar_out_window)
            self.plot_angle_tar_out(n, s, self.num_nodes, tar_out_window)
            logging.info('Plot tar_out {} figure'.format(n))
            # self.plot_l2(n, s, l2_error_window)
            self.plot_figure_l2_error(n, s, self.total_l2, l2_error_window)
            logging.info('Plot l2 error {} figure'.format(n))

    def cal_statistic(self, statistic_window):
        self.statistics = []
        for s in self.start_plot:
            self.statistics.append(np.mean(self.total_l2[s:s + statistic_window]))
            self.statistics.append(np.var(self.total_l2[s:s + statistic_window]))

        start = 0
        stop = 0
        for i in self.group_info:
            stop += i
            for s in self.start_plot:
                tmp_l2 = np.sqrt(np.sum(self.diff[:, start:stop], axis=1))
                self.statistics.append(np.mean(tmp_l2[s:s + statistic_window]))
                self.statistics.append(np.var(tmp_l2[s:s + statistic_window]))
            start = stop

        np.savetxt(self.path + 'result.txt', self.statistics)
        logging.info('Statistic results are calculated and stored.')

    def plot_l2(self, name, start_plot, window):
        if name == 'before_finish':
            start_plot = start_plot - window

        plt.figure(figsize=(8*(len(self.group_info)+1), 6))
        self.statistics = []
        plt.subplot(int(str(1)+str(len(self.group_info)+1)+str(1)))
        plt.scatter(np.array([i for i in range(start_plot, start_plot + window)]), self.total_l2[start_plot: start_plot + window], label='total', s=1)
        plt.legend(loc='upper left', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        start = 0
        stop = 0
        counter = 0
        for i in self.group_info:
            stop += i
            counter += 1
            tmp_l2 = np.sqrt(np.sum(self.diff[:, start:stop], axis=1))
            plt.subplot(int(str(1)+str(len(self.group_info)+1)+str(counter+1)))
            plt.scatter(np.array([i for i in range(start_plot, start_plot + window)]), tmp_l2[start_plot: start_plot + window], label='group'+str(counter-1), s=1)
            plt.legend(loc='upper left', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            start = stop

        plt.xlabel('round number', fontsize=16)
        plt.ylabel('l2 trigonometric error', fontsize=16)
        plt.savefig(self.path + 'grouping_l2_' + name + '.jpg')

    def plot_long(self, start_plot, n, window=100000):
        labels = []
        for i in range(n):
            labels.append(('theta_' + str(i) + '_hat', 'theta_' + str(i)))

        s = 2
        c = '#FF8C00'
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(200, 8), dpi=100, tight_layout=True)
        x_input = np.array([i for i in range(start_plot, start_plot + window)])

        for i in range(n):
            axes[i].scatter(x_input, self.out[start_plot: start_plot + window, i], label=labels[i][1], s=s, c='blue')
            axes[i].scatter(x_input, self.tar[start_plot: start_plot + window, i], label=labels[i][0], s=s, c=c)
            axes[i].legend(loc='upper left', fontsize=10)

        plt.xlabel('round number')
        plt.ylabel('network output angle')
        plt.savefig(self.path + 'long_angle_tar_out_1' + '.jpg')
        plt.clf()

        start_plot = start_plot + window
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(200, 8), dpi=100, tight_layout=True)
        x_input = np.array([i for i in range(start_plot, start_plot + window)])

        for i in range(n):
            axes[i].scatter(x_input, self.out[start_plot: start_plot + window, i], label=labels[i][1], s=s, c='blue')
            axes[i].scatter(x_input, self.tar[start_plot: start_plot + window, i], label=labels[i][0], s=s, c=c)
            axes[i].legend(loc='upper left', fontsize=10)

        plt.xlabel('round number')
        plt.ylabel('network output angle')
        plt.savefig(self.path + 'long_angle_tar_out_2' + '.jpg')
        plt.clf()

        start_plot = start_plot + window
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(200, 8), dpi=100, tight_layout=True)
        x_input = np.array([i for i in range(start_plot, start_plot + window)])

        for i in range(n):
            axes[i].scatter(x_input, self.out[start_plot: start_plot + window, i], label=labels[i][1], s=s, c='blue')
            axes[i].scatter(x_input, self.tar[start_plot: start_plot + window, i], label=labels[i][0], s=s, c=c)
            axes[i].legend(loc='upper left', fontsize=10)

        plt.xlabel('round number')
        plt.ylabel('network output angle')
        plt.savefig(self.path + 'long_angle_tar_out_3' + '.jpg')
        plt.clf()
