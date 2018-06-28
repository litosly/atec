#####################################
# Main File
# Author: Justin
# Date modified: 2018-06-24
#####################################


import os
import sys
import argparse
import numpy as np
import yaml
from hyperopt import hp, fmin, tpe
import hyperopt
import copy

# from dataloader import get_test_dataloader
from solver import Solver


def load_config(config):
    # try:
    # 	params = yaml.load(config.params_path)
    # 	complete_config = {**config, **params}
    # 	return complete_config
    # except:
    # 	print("Failed to load model hyper-parameters.")
    # 	sys.exit(1)
    params = yaml.load(config.params_path)
    complete_config = {**config, **params}
    return complete_config

def create_dirs(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)


def objective_wrapper(config):
    def objective(hypes):
        x = copy.deepcopy(config)
        x.hidden_size, x.lr = hypes
        x.hidden_size = int(x.hidden_size)
        return Solver(x).train()
    return objective


def main(config):
    # set up
    # config = load_config(config)
    folders = [config.model_dir, config.log_dir]
    create_dirs(folders)

    # # model stuff
    # solver = Solver(config)
    # solver.train()
    space = [hp.quniform("x", 5, 100, 1), hp.uniform("y", 0.0001, 0.1)]
    obj = objective_wrapper(config)
    best = fmin(obj, space, algo=tpe.suggest, max_evals=10)
    print("Best Parameter: ", best)
    print(hyperopt.space_eval(space, best))

####################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data related 
    parser.add_argument('--params_path', type=str, default="../data", help='model parameters file')
    parser.add_argument('--data_dir', type=str, default="../src", help='data directory')
    parser.add_argument('--model_dir', type=str, default="../models", help='model directory')
    parser.add_argument('--log_dir', type=str, default="../logs", help='log directory')
    # parser.add_argument('--train_file', type=str, default="train.txt", help='training data path')
    # parser.add_argument('--valid_file', type=str, default="valid.txt", help='validation data path')
    # parser.add_argument('--test_file', type=str, default="test.txt", help='test data path')

    # # training specs 
    parser.add_argument('--input_size', type=int, default=5, help='interval between training model saves')
    parser.add_argument('--hidden_size', type=int, default=10, help='interval between training model saves')
    parser.add_argument('--output_size', type=int, default=5, help='interval between training model saves')

    parser.add_argument('--batch_size', type=int, default=3, help='number of workers for parallelizationp')
    parser.add_argument('--lr', type=float, default=0.01, help='training learning rate')
    parser.add_argument('--num_epoch', type=int, default=10, help='number of trianing epochs')
    parser.add_argument('--log_step', type=int, default=10, help='interval between training status logs')
    parser.add_argument('--num_worker', type=int, default=4, help='number of workers for parallelizationp')
    parser.add_argument('--save_step', type=int, default=10, help='interval between training model saves')
    print('-------------Good Boy--------------')
    args = parser.parse_args()
    print(args)
    print('-------------Good Girl--------------')
    main(args)
    # print(yaml.load(config.params_path))
