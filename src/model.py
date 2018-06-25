import os 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config 
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size 

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, 
            bidirectional=True, batch_first=True)
        # self.initHidden()

    def forward(self, input): #, hidden):
        output, hidden = self.lstm(input) #, self.init_state)
        return output, hidden 

    def initHidden(self):
        self.init_state = (torch.zeros(self.config.batch_size, 2, self.hidden_size), 
            torch.zeros(self.config.batch_size, 2, self.hidden_size))


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config 
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size*2, num_layers=1, 
            bidirectional=False, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Comparator(nn.Module):
    def __init__(self, config):
        super(Comparator, self).__init__()
        self.config = config 
        self.hidden_size = config.hidden_size
        self.fc1 = nn.Linear(self.hidden_size*4, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, input):
        out = F.elu(self.fc1(input))
        out = F.elu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out 


class AtecModel(nn.Module):
    def __init__(self, config):
        super(AtecModel, self).__init__() 
        self.config = config
        self.encoder = Encoder(config)
        self.comparator = Comparator(config)
        self.softmax = nn.Softmax(1)

    def forward(self, input, indices):
        output, (ht, ct) = self.encoder(input)
        summarize_vec = torch.cat((ht[0], ht[1]), dim=1)
        unsorted_vec = reverse_from_indices(summarize_vec, indices)
        grouped_vec = concatenate_pair(unsorted_vec)
        out = self.comparator(grouped_vec)
        out = self.softmax(out)
        return out 


def reverse_from_indices(input, indices):
    reverse_indices = [0]*len(indices)
    for i, idx in enumerate(indices):
        reverse_indices[idx] = i 
    return input[reverse_indices]


def concatenate_pair(input):
    n = len(input)
    assert n % 2 == 0
    first = list(range(0, n, 2))
    second = list(range(1, n, 2))
    sent1 = input[first]
    sent2 = input[second]
    pairs = torch.cat((sent1, sent2), dim=1)
    return pairs




