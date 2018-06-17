import os 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class BaseModel(nn.Module):
	def __init__(self):
		super(BaseModel, self).__init__()

	def forward(self, x):
		return None


class AtecModel(BaseModel):
	def __init__(self, config):
		super(AtecModel, self).__init__() 
		self.config = config
		self.encoder = Encoder(config)
		self.decoder = Decoder(config)

	def forward(self, x):
		return None 


class Encoder(nn.Module):
	def __init__(self, config):
		super(Encoder, self).__init__()
		self.input_size = config.input_size
		self.hidden_size = config.hidden_size 

		self.embedding = nn.Embedding(self.input_size, self.hidden_size)
		self.lstm = nn.LSTM(self, hidden_size, self,hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded 
		output, hidden = self.lstm(output, hidden)
		return output, hidden 

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size)


class Decoder(nn.Module):
	def __init__(self, config):
		super(Decoder, self).__init__()
		self.hidden_size = config.hidden_size
		self.output_size = config.output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input):
		output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


