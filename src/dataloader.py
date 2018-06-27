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
import pandas as pd

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

	def read_sentence(self, num_files = 1,sentence_i=1, data_path = "../data/"):
		data = []
		if sentence_i == 1:
			path = data_path + 'Embedded Data/'
			for i in range(num_files):
				filename = 'embedded' + str(i+1) +'.npy'
				temp = np.load(path+filename)
				data.extend(temp)
		else:
			path = data_path + 'Embedded Data 2/'
			for i in range(num_files):
				filename = 'embedded' + str(i+41) +'.npy'
				temp = np.load(path+filename)
				data.extend(temp)
		return data

	def sparse2array(self, sparse_sentence):
		return [i.toarray() for i in sparse_sentence]

	def read_sentence_sparse(self, num_files = 1,sentence_i=1, data_path = "../data/"):
		data = []
		if sentence_i == 1:
			path = data_path + 'sparse1/'
			for i in range(num_files):
				filename = 'sparsed' + str(i+1) +'.npy'
				temp = np.load(path+filename)
				data.extend(temp)
		else:
			path = data_path + 'sparse2/'
			for i in range(num_files):
				filename = 'sparsed' + str(i+41) +'.npy'
				temp = np.load(path+filename)
				data.extend(temp)
		data = [[vector.toarray()[0] for vector in sentence] for sentence in data]
		return data

	def load_embedded_data(self):
		self.num_of_data_files = 10
		self.data_path = "../../atec_data/"
		self.num_of_sentences = self.num_of_data_files*1000 #from how ivan's data is structured
		# self.sentences1 = self.read_sentence(num_files = self.num_of_data_files,sentence_i=1,data_path = self.data_path)
		# self.sentences2 = self.read_sentence(num_files = self.num_of_data_files,sentence_i=2,data_path = self.data_path)
		
		
		
		self.sentences1 = self.read_sentence_sparse(num_files = self.num_of_data_files, sentence_i=1,data_path = self.data_path)
		self.sentences2 = self.read_sentence_sparse(num_files = self.num_of_data_files, sentence_i=2,data_path = self.data_path)
		
		# shape: (1000*num_files) X num_of_vectors (each vector is of size 4096)
		# print("Number of sentences used for training: ", self.num_of_sentences)
		
		#each element in the input sequence should be like the following format
		#x_1 = (sentences1[0],sentences2[0])

		self.data = [(self.sentences1[i], self.sentences2[i]) for i in range(self.num_of_sentences)]
		

		# print("--------- loading label ------------")
		self.labels = list(pd.read_csv('../data/processed_data.csv')['label'])[:self.num_of_sentences]
		# print("Number of labels loaded: ", len(self.labels))
		


		# for test 
		# self.data = [(np.random.rand(4, 5), np.random.rand(5, 5)),
		# 			 (np.random.rand(6, 5), np.random.rand(7, 5)),
		# 			 (np.random.rand(6, 5), np.random.rand(7, 5)),
		# 			]
		# self.labels =  [0, 1, 1]
		
		
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



