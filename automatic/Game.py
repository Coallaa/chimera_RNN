import torch
import logging
import pickle
import time
import tqdm


class Game:
    def __init__(self, path, model):
        # storage_param is a dictionary which determines how to store the output. 'd': (10, 1) means that
        # first 10 values of row 1 will be recorded for each step
        self.model = model
        self.path = path
        self.total_round = None
        self.store_counter = 0
        self.prediction = torch.zeros(self.model.num_nodes * 2).type(torch.float64).to(self.model.device)
        self.error = torch.zeros(self.model.num_nodes * 2).type(torch.float64).to(self.model.device)
        self.store_dic = {}
        self.target = None
        self.dt = None

    def storage_init(self):
        device = self.model.device
        m = 2 * self.model.num_nodes
        self.store_dic['predict'] = torch.zeros((int(self.total_round) + 1, self.model.num_nodes)).type(torch.float64).to(device)
        self.store_dic['error'] = torch.zeros((int(self.total_round) + 1, m)).type(torch.float64).to(device)

    def data_load(self, file):
        data = torch.from_numpy(pickle.load(open(file, 'rb'))).to(self.model.device)

        nodes = self.model.num_nodes
        self.target = torch.zeros((2 * nodes, len(data[0]))).type(torch.float64).to(self.model.device)
        self.target[0:nodes] = torch.cos(data)
        self.target[nodes:2 * nodes] = torch.sin(data)
        logging.info('Inputs are loaded from target.pkl.')

    def run(self, pre_train_round, train_round, post_train_round):
        self.total_round = pre_train_round + train_round + post_train_round
        self.storage_init()

        # pre_training
        logging.info('Start {} rounds pre-training.'.format(pre_train_round))
        self.run_without_train(0, pre_train_round)
        # training
        logging.info('Start {} rounds training.'.format(train_round))
        self.run_train(pre_train_round, pre_train_round + train_round)
        # post_training
        logging.info('Start {} rounds post-training.'.format(post_train_round))
        self.run_without_train(pre_train_round + train_round, pre_train_round + train_round + post_train_round)

    def run_without_train(self, start, stop):
        # use some smart method, t_step should adaptive (at least can be modified manually), or try tqdm.
        t = time.perf_counter()
        bar = tqdm.tqdm(range(start, stop))
        for i in bar:
            self.prediction = self.model.forward(self.prediction)
            self.error = self.prediction - self.target[:, i]
            self.store()

        logging.info('{}'.format(tqdm.tqdm.__str__(bar)))

    def run_train(self, start, stop):
        # use some smart method, t_step should adaptive (at least can be modified manually), or try tqdm.
        t = time.perf_counter()
        bar = tqdm.tqdm(range(start, stop))
        for i in bar:
            self.prediction = self.model.forward(self.prediction)
            self.error = self.prediction - self.target[:, i]
            self.model.train(self.error)
            self.store()

        logging.info('{}'.format(tqdm.tqdm.__str__(bar)))

    def store(self):
        self.store_dic['predict'][self.store_counter] = torch.arctan2(self.prediction[self.model.num_nodes:],
                                                                      self.prediction[:self.model.num_nodes])
        self.store_dic['error'][self.store_counter] = self.error
        self.store_counter += 1

    def save_storage_file(self):
        for key in self.store_dic.keys():
            if self.model.device.type != 'cpu':
                self.store_dic[key] = self.store_dic[key].cpu().numpy()
            else:
                self.store_dic[key] = self.store_dic[key].numpy()
            pickle.dump(self.store_dic[key], open(self.path + key + '.pkl', 'wb'))

        logging.info('{} are stored as pkl files.'.format(self.store_dic.keys()))
