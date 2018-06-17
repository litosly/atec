import os 
import re 
import csv
import numpy as np 

import torch 
from torch.utils.data import Dataset, DataLoader


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


class AtecDataset(Dataset):
	def __init__(self, config, mode="train"):
		self.config = config 
		self.build_dic()

	def __len__(self):
		return len(self.items)

	def __getitem__(self, index):
		return self.items[index]

	def build_dic(self):
		items, lang = prepareData(config.data_path)
		self.items = items 
		self.lang = lang 


def get_dataloader(config, mode="train"):
	atec_dataset = AtecDataset(config, mode)
	atrc_dataloader = DataLoader(dataset = atec_dataset, 
							batch_size = config.batch_size,
							shuffle = True, 
							num_worker = config.num_worker)
	return atec_dataloader
