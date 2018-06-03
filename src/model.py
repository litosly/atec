import os 
import numpy as np 

import torch 
import torch.nn as nn 


class BaseModel(object):
	def __init__(self):
		pass 

	def forward(self, x):
		return None


class AtecModel(BaseModel):
	def __init__(self, config):
		super(AtecModel, self).__init__() 

	def forward(self, x):
		return None 