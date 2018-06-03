import os 
import argparse 
import numpy as np 

from model import AtecModel
from dataloader import get_dataloader


def evaluate(model, config, mode="test"):
	# load data 
	dataloader = get_dataloader(config, mode=mode)

	# evaluate 
	for i, data in enumerate(dataloader):
			pass 




if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# data related 
	parser.add_argument('--data_dir', type=str, default="", help='data directory')
	parser.add_argument('--train_file', type=str, default="train.txt", help='training data path')
	parser.add_argument('--valid_file', type=str, default="valid.txt", help='validation data path')
	parser.add_argument('--test_file', type=str, default="test.txt", help='test data path')

	parser.add_argument('--model_file', type=str, default="", help='model path')

	args = parser.parse_args()
	print(args)

	# load model 
	model =  None 
	
	# evaluate 
	evaluate(model, args, mode="train")
	evaluate(model, args, mode="valid")
	evaluate(model, args, mode="test")

