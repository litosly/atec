import os 
import numpy as np 

import torch 
from torch.utils.data import Dataset, DataLoader


class AtecDataset(Dataset):
	def __init__(self, config, mode="train"):
		pass 

	def __len__(self):
		return None 

	def __getitem__(self, index):
		return None 


def get_dataloader(config, mode="train"):
	atec_dataset = AtecDataset(config, mode)
	atrc_dataloader = DataLoader(dataset = atec_dataset, 
							batch_size = config.batch_size,
							shuffle = True, 
							num_worker = config.num_worker)
	return atec_dataloader
