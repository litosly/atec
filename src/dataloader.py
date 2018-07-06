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
import pickle 

import torch 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

####################################################################################################


class AtecDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.data_path = "E:\ATEC_data/"
        super(AtecDataset, self).__init__()
        self.config = config 
        self.mode = mode 
        self.load_embedded_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        true_index = (index//1000,index%1000)
        sentence1 = self.read_specific_file_and_sentence(sentence_i = 1, file_idx = true_index[0], sentence_idx = true_index[1],data_path = self.data_path)
        sentence2 = self.read_specific_file_and_sentence(sentence_i = 2, file_idx = true_index[0], sentence_idx = true_index[1],data_path = self.data_path)
        data = (sentence1,sentence2)

        # print('------------alright-----------------')
        # print(data[0])
        return data, self.labels[index]
        # print('------------alright-----------------')
        # print(self.data[index][0])
        # return self.data[index], self.labels[index]

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

    def read_specific_file_and_sentence(self, sentence_i = 1, file_idx = 1, sentence_idx = 1, data_path = "../data/"):
        if sentence_i == 1:
            path = data_path + 'sparse1/'
            filename = 'sparsed' + str(file_idx+1) +'.npy'
            return [vector.toarray()[0] for vector in np.load(path+filename)[sentence_idx]] 
        else:
            path = data_path + 'sparse2/'
            filename = 'sparsed' + str(file_idx+41) +'.npy'
            return [vector.toarray()[0] for vector in np.load(path+filename)[sentence_idx]] 



    def load_embedded_data(self):
        self.num_of_data_files = 1
        
        self.num_of_sentences = self.num_of_data_files*1000 #from how ivan's data is structured
        # self.sentences1 = self.read_sentence(num_files = self.num_of_data_files,sentence_i=1,data_path = self.data_path)
        # self.sentences2 = self.read_sentence(num_files = self.num_of_data_files,sentence_i=2,data_path = self.data_path)

        # self.sentences1 = self.read_sentence_sparse(num_files = self.num_of_data_files, sentence_i=1,data_path = self.data_path)
        # self.sentences2 = self.read_sentence_sparse(num_files = self.num_of_data_files, sentence_i=2,data_path = self.data_path)
        
        # shape: (1000*num_files) X num_of_vectors (each vector is of size 4096)
        # print("Number of sentences used for training: ", self.num_of_sentences)
        
        #each element in the input sequence should be like the following format
        #x_1 = (sentences1[0],sentences2[0])

        # self.data = [(self.sentences1[i], self.sentences2[i]) for i in range(self.num_of_sentences)]
        

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


class AtecDatasetDopeIO(Dataset):
    def __init__(self, config, mode="train"):
        super(AtecDatasetDopeIO, self).__init__()
        self.config = config 
        self.mode = mode 
        self.load_data_indices()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file_name = self.data_paths[index]
        path = os.path.join(self.config.embedding_dir, file_name)
        with open(path, "rb") as f:
            embedding = pickle.load(f)
        # further preprocessing on embedding if using sparse or etc.
        return embedding, self.labels[index]

    def load_data_indices():
        if os.path.exists(self.config.embedding_dir):
            os.makedirs((self.config.embedding_dir))
            self.write_embeddings_to_file()
        self.data_paths = []	# get list of embedding file names 
        self.labels = []	# get list of labels 

    def write_embedding_to_file():
        with open(self.config.embedding_dir) as infile:
            for i, line in enumerate(infile):
                embedding = None 
                file_name = str(i) + ".pkl"
                with open(file_name, "wb") as f:
                    pickle.dump(embedding, f)
            print("All embedding written to files")



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



 # some helper functions 

def initialize_line_reader(path):
    with open(path, 'r') as f:
        for line in f:
            yield line 
    
def get_batch(line_reader, batch_size):
    buffer = []
    try:
        for _ in range(batch_size):
            line = line_reader.next()
            buffer.append(line)
    except StopIteration:
        print("end")
    except (IOError, OSError):
        print("processing file")
    return buffer 


"""
notes, each time initialize a reader, get batch sequentially 

e.g. 
file "temp.txt" consists of lines:
1233434\n
1234\n
14\n
1\n

reader = initialize_line_reader("temp.txt")
data_batch = get_batch(reader, 2)
while len(data_batch) != 0:
    print(data_batch)
    data_batch = get_batch(reader, 2)

output: 
['1233434\n', '1234\n']
['14\n', '1\n']
end


-> somehow replace the "for i, (data, labels, indices, lengths) in enumerate(self.train_dataloader):" 
in train_step in solver.py with the above, reinitialize the reader on a different epoch

ENJOY :) 
"""