import argparse
from ChimeraModel import ChimeraModel
# from ChimeraModel1 import ChimeraModel1
from Game import Game
# from Game2 import Game2
from Output import Output
import time
import logging
import sys
import os
from shutil import copyfile


def create_output_folder(dic):
    if os.path.exists(dic):
        logging.info("Already exist {}.".format(dic))
    else:
        os.mkdir(dic)


def other_params_parser(args):
    device, rounds, num_neuron, group_info, dt, sparse_param, lam, G, Q, compare_round, statistic_window, \
        tar_out_window, l2_error_window = load_origin_params(args.origin)
    job_name = 'job'
    if args.device:
        job_name = job_name + '_device_' + args.device
        device = args.device
    if args.rounds:
        job_name = job_name + '_rounds_' + args.rounds
        rounds = list(map(int, args.rounds.split(',')))
    if args.num_neuron:
        job_name = job_name + '_num_neuron_' + args.num_neuron
        num_neuron = int(args.num_neuron)
    if args.group_info:
        job_name = job_name + '_group_info' + args.group_info
        group_info = list(map(int, args.group_info.split(',')))
    if args.dt:
        job_name = job_name + '_dt_' + args.dt
        dt = float(args.dt)
    if args.sparse_param:
        job_name = job_name + '_sparse_param_' + args.sparse_param
        sparse_param = float(args.sparse_param)
    if args.lam:
        job_name = job_name + '_lam_' + args.lam
        lam = float(args.lam)
    if args.G:
        job_name = job_name + '_G_' + args.G
        G = float(args.G)
    if args.Q:
        job_name = job_name + '_Q_' + args.Q
        Q = float(args.Q)
    if args.compare_round:
        job_name = job_name + '_compare_round_' + compare_round
        compare_round = int(args.compare_round)
    if args.statistic_window:
        job_name = job_name + '_statistic_window_' + args.statistic_window
        statistic_window = int(args.statistic_window)
    if args.tar_out_window:
        job_name = job_name + '_tar_out_window_' + args.tar_out_window
        tar_out_window = int(args.tar_out_window)
    if args.l2_error_window:
        job_name = job_name + '_l2_error_window_' + args.l2_error_window
        l2_error_window = int(args.l2_error_window)

    try:
        if tar_out_window + compare_round > rounds[2] or l2_error_window + compare_round > rounds[2]:
            raise Exception("Unreasonable window size and post rounds.")
    except Exception as err:
        logging.error('An exception happened: ' + str(err))
        sys.exit(1)

    return device, rounds, num_neuron, group_info, dt, sparse_param, lam, G, Q, compare_round, statistic_window, \
        tar_out_window, l2_error_window


def load_origin_params(file_name):
    file = open(file_name, 'r')
    params = file.readlines()
    idx = 0
    while params[idx].startswith("#"):
        idx += 1
    device = list(map(str, params[idx].split('\n')))[0]
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    rounds = list(map(int, params[idx].split(',')))
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    num_neuron = int(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    group_info = list(map(int, params[idx].split(',')))
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    dt = float(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    sparse_param = float(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    lam = float(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    G = float(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    Q = float(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    compare_round = int(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    statistic_window = int(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    tar_out_window = int(params[idx])
    idx += 1
    while params[idx].startswith("#"):
        idx += 1
    l2_error_window = int(params[idx])
    file.close()
    return device, rounds, num_neuron, group_info, dt, sparse_param, lam, G, Q, compare_round, statistic_window, \
        tar_out_window, l2_error_window


def run(args):
    device, rounds, num_neuron, group_info, dt, sparse_param, lam, G, Q, compare_round, statistic_window, \
        tar_out_window, l2_error_window = other_params_parser(args)
    logging.info('The parameters has been loaded: device={}, rounds={}, num_neuron={}, group_info={}, dt={}, sparse_pa'
                 'ram={}, lam={}, G={}, Q={}, compare_round={}, statistic_window={}, tar_out_window={}, l2_error_window'
                 '={}'.format(device, rounds, num_neuron, group_info, dt, sparse_param, lam, G, Q, compare_round,
                              statistic_window, tar_out_window, l2_error_window))
    job_name = args.dic.split('/')[-2]
    model = ChimeraModel(dt=dt, sparse_param=sparse_param, num_neuron=num_neuron, num_nodes=sum(group_info), lam=lam,
                         G=G, Q=Q, device=device, use_reload_seed=False)
    game = Game(args.dic, model)

    logging.info('Model and Game have been initialized.')

    game.data_load(args.input)
    game.run(rounds[0], rounds[1], rounds[2])
    game.save_storage_file()
    logging.info('Finish running.')

    output = Output(path=args.dic, finish_train=rounds[0] + rounds[1] + 1, input_file=args.input,
                    predict_file='predict.pkl', group_info=group_info, compare_round=compare_round)
    output.cal_l2_error()
    output.plot_figures(tar_out_window, l2_error_window)
    output.cal_statistic(statistic_window)
    return job_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default='../input/target.pkl', type=str, help='input pickle file path')
    parser.add_argument('--origin', default='../input/origin.txt', type=str, help='load origin parameters')
    parser.add_argument('--dic', default='../test_case/', type=str, help='dictionary to store the output files')
    parser.add_argument('--device', default=None, type=str, help='device')
    parser.add_argument('--rounds', default=None, type=str, help='pre/during/post train rounds')
    parser.add_argument('--num_neuron', default=None, type=str, help='number of neurons')
    parser.add_argument('--group_info', default=None, type=str, help='grouping information')
    parser.add_argument('--dt', default=None, type=str, help='time step')
    parser.add_argument('--sparse_param', default=None, type=str, help='sparse parameter')
    parser.add_argument('--lam', default=None, type=str, help='lambda')
    parser.add_argument('--G', default=None, type=str, help='G')
    parser.add_argument('--Q', default=None, type=str, help='Q')
    parser.add_argument('--compare_round', default=None, type=str, help='compare result after x rounds post train')
    parser.add_argument('--statistic_window', default=None, type=str, help='statistic window')
    parser.add_argument('--tar_out_window', default=None, type=str, help='tar_out window')
    parser.add_argument('--l2_error_window', default=None, type=str, help='l2_error window')

    args = parser.parse_args()
    # start job
    f = open('../std_output/run_info.txt', 'a')
    f.write('Start {}\n'.format(args.dic))
    create_output_folder(args.dic)

    # add logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # log print
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # log file
    fh = logging.FileHandler(args.dic+"test.log", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    start = time.perf_counter()
    job_name = run(args)
    finish = time.perf_counter()

    logging.info('All process finished. Runtime: {}s'.format(finish - start))
    copyfile(args.dic+"predict.pkl", "../std_output/" + job_name + ".pkl")
    copyfile(args.dic+"test.log", "../std_output/" + job_name + ".log")
    f_job = open('../std_output/jobs.txt', 'a')
    f_job.write(job_name+'\n')
    f_job.close()
    f.write('Finish {}\n'.format(args.dic))
    f.close()

