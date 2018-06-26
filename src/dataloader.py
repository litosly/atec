#####################################
# Load data/embedding 
# Author: Justin
# Date modified: 2018-06-24
#####################################


import os 
import re 
import csv
import argparse 
import numpy as np 

import torch 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

####################################################################################################


class AtecDataset(Dataset):
	def __init__(self, config, mode="train"):
		super(AtecDataset, self).__init__()
		self.config = config 
		self.mode = mode 
		self.load_embedded_data()

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.data[index], self.labels[index]

	def load_embedded_data(self):
		# for test 
		self.data = [(np.random.rand(4, 5), np.random.rand(5, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(6, 5), np.random.rand(7, 5)),
					 (np.random.rand(8, 5), np.random.rand(9, 5))]
		self.labels =  [0, 1, 1]
		
		# 
		# self.data = None  # list of tuples 
		# self.labels = None # list of ints 


def my_collate(batch):
	sentences = [torch.FloatTensor(sent) for item in batch for sent in item[0]]   # item[0] is sentence tuple
	labels = torch.LongTensor([item[1] for item in batch]) 	# item[1] is label

	lengths = torch.LongTensor([len(sent) for sent in sentences])
	sorted_lengths, indices = torch.topk(lengths, k=len(lengths))
	sentences = [sentences[idx] for idx in indices]

	data = pad_sequence(sentences, batch_first=True)
	data = pack_padded_sequence(data, sorted_lengths, batch_first=True)
	return data, labels, indices, lengths


def get_dataloader(config, mode="train", full=False):
	atec_dataset = AtecDataset(config, mode)
	if not full:
		atec_dataloader = DataLoader(dataset = atec_dataset, 
								batch_size = config.batch_size,
								shuffle = True, 
								collate_fn = my_collate,
								num_workers = config.num_worker)
	else:
		atec_dataloader = DataLoader(dataset = atec_dataset, 
								batch_size = len(atec_dataset),
								shuffle = True, 
								collate_fn = my_collate,
								num_workers = config.num_worker)
	return atec_dataloader



