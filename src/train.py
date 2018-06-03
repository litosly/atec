import os 
import argparse 
import numpy as np 

import torch 
from torch.optim import Adam 
from torch.optim.lr_scheduler import MultiStepLR

from model import AtecModel
from dataloader import get_dataloader



def main(config):
	# load data 
	train_dataloader = get_dataloader(config, mode="train")

	# define model, optimizer, scheduler, logger 
	model = AtecModel(config)
	optimizer = Adam(model.parameters, lr=config.lr)
	scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
	 
	# training iterations 
	total_steps = len(train_dataloader)

	for epoch in range(config.num_epoch):
		scheduler.step()
		for i, data in enumerate(train_dataloader):
			pass 




if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# data related 
	parser.add_argument('--data_dir', type=str, default="", help='data directory')
	parser.add_argument('--train_file', type=str, default="train.txt", help='training data path')
	parser.add_argument('--valid_file', type=str, default="valid.txt", help='validation data path')
	parser.add_argument('--test_file', type=str, default="test.txt", help='test data path')

	# training specs 
	parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
	parser.add_argument('--num_epoch', type=int, default=100, help='number of trianing epochs')
	parser.add_argument('--lr', type=float, default=0.01, help='training learning rate')
	parser.add_argument('--log_step', type=int, default=10, help='interval between training status logs')
	parser.add_argument('--save_step', type=int, default=10, help='interval between training model saves')
	parser.add_argument('--num_worker', type=int, default=4, help='number of workers for parallelizationp')

	args = parser.parse_args()
	print(args)

	main(args)