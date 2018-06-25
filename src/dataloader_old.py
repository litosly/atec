import os 
import re 
import csv
import argparse 
import numpy as np 

import torch 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class Lang(object):
	""" creates word-index mappings 
	"""
	def __init__(self, name):
		self.name = name 
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2   

	def addSentence(self, sentence):
		# need to split according to Chinese sentences or specific input format
		for word in sentence.split():
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1 
			self.index2word[self.n_words] = word 
			self.n_words += 1 
		else:
			self.word2count[word] += 1 


def unicodeToAscii(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s 

def readLangs(path):
	print("Reading data...")

	with open(data_path, 'rb') as f:
	    reader = csv.reader(f)
	    data_list = list(reader)
	# each item (sent1, sent2, label)
	item_list = [[w for w in item[0].split('\t')] for item in data_list]
	# TODO, apply filters here to remove unwanted lexiacal info 
	# word-index mappings 
	lang = Lang(path)
	return item_list, lang 

def prepareData(path):
	items, lang = readLangs(path)
	for item in items:
		lang.addSentence(item[0])
		lang.addSentence(item[1])
	print("Counted words: ", lang.n_words)
	return items, lang 

def indexFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
	index = indexFromSentence(lang, sentence)
	index.append(EOS_token)
	return torch.tensor(index, dtype=torch.long)


class AtecDataset(Dataset):
	def __init__(self, config, mode="train"):
		super(AtecDataset, self).__init__()
		self.config = config 
		# self.build_dic()
		self.load_embedded_data()

	def __len__(self):
		return len(self.items)

	def __getitem__(self, index):
		return self.items[index]

	# def build_dic(self):
	# 	items, lang = prepareData(config.data_path)
	# 	self.items = items 
	# 	self.lang = lang 
	
	def load_embedded_data(self):


def get_dataloader(config, mode="train"):
	atec_dataset = AtecDataset(config, mode)
	atrc_dataloader = DataLoader(dataset = atec_dataset, 
							batch_size = config.batch_size,
							shuffle = True, 
							num_worker = config.num_worker)
	return atec_dataloader


####################################################################################################

class TestDataset(Dataset):
	def __init__(self, config, mode="train"):
		super(TestDataset, self).__init__() 
		self.config = config 
		self. build_dataset()

	def build_dataset(self):
		self.lengths = np.random.randint(20, size=10) + 1
		self.data = [np.random.rand(l, self.config.input_size) for l in self.lengths]
		self.labels = [np.random.randint(2, size=l) for l in self.lengths]

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.data[index], self.labels[index]


def pad_collate(batch):
	# print(batch)
	batch.sort(key=lambda x: len(x[0]), reverse=True)
	lengths = torch.LongTensor([len(item[1]) for item in batch])
	# print(lengths)
	data = pad_sequence([torch.FloatTensor(item[0]) for item in batch], batch_first=True)
	labels = pad_sequence([torch.LongTensor(item[1]) for item in batch], batch_first=True)
	data = pack_padded_sequence(data, lengths, batch_first=True)
	return data, labels


def get_test_dataloader(config, mode="train"):
	dataset = TestDataset(config, mode)
	dataloader = DataLoader(dataset = dataset, 
							batch_size = config.batch_size,
							shuffle = True,
							collate_fn=pad_collate) 
	return dataloader






####################################################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_size', type=int, default=5, help='interval between training model saves')
	parser.add_argument('--batch_size', type=int, default=3, help='number of workers for parallelizationp')

	args = parser.parse_args()
	# print(args)

	dl = get_test_dataloader(args)
	for i, (data, labels) in enumerate(dl):
		# pass 
		print("data", data)
		# print("labels", labels)
		break 














