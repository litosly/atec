import pandas as pd
import numpy as np
import pickle as pkl


import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import jieba



df = pd.read_csv('atec_nlp_sim_train.csv',error_bad_lines=False,header = None)


def get_data_for_one_line(line):
    x = []
    temp = line.split('\t')[1]
    if temp[0]=='\ufeff':
        temp = temp[1:]
#     if(temp)
        
    x.append(temp)
    x.append(line.split('\t')[2])
    y = line.split('\t')[3]
    return x,y

def get_data_for_mutiple_line(original_data):
    data = []
    for i in range(len(original_data)):
        x,y = get_data_for_one_line(original_data[i])
        data.append([x,y])
    return data



######################################################################

if __name__ == '__main__':
	pass
	# data_1 = get_data_for_mutiple_line(df[0]) 

