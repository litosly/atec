import os 
import argparse 
import numpy as np 

import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torch.optim.lr_scheduler import MultiStepLR

from model import Encoder, Comparator, AtecModel
from dataloader import get_dataloaders, get_test_dataloader
from torch.nn.utils.rnn import pad_packed_sequence



def main(config):
    # load data 
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(config)

    # define model, loss, optimizer, scheduler, logger 
    model = AtecModel(config)
    criterion = nn.CrossEntropyLoss()
    trainable_params = list(model.encoder.parameters())+list(model.comparator.parameters())
    optimizer = Adam(trainable_params, lr=config.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
     
    # training iterations 
    total_steps = len(train_dataloader)

    for epoch in range(config.num_epoch):
        scheduler.step()
        for i, (data, labels, indices, lengths) in enumerate(train_dataloader):
            logits = model(data, indices)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log loss, could visualize in tensorboard if needed 
            if (i+1) % config.log_step == 0:
                print( 'Epoch [%d/%d], Step[%d/%d], loss: %.4f, ' 
                      % (epoch+1, config.num_epochs, i+1, total_steps, loss.data[0]))

        # save the model per epoch, only save parameters 
        if (epoch+1) % config.save_step == 0:
            model_path = os.path.join(config.model_dir, 'model-%d.pkl' %(epoch+1))
            torch.save(model.state_dict(), model_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data related 
    parser.add_argument('--data_dir', type=str, default="", help='data directory')
    parser.add_argument('--model_dir', type=str, default="", help='model directory')
    parser.add_argument('--train_file', type=str, default="train.txt", help='training data path')
    parser.add_argument('--valid_file', type=str, default="valid.txt", help='validation data path')
    parser.add_argument('--test_file', type=str, default="test.txt", help='test data path')

    # training specs 
    parser.add_argument('--input_size', type=int, default=5, help='interval between training model saves')
    parser.add_argument('--hidden_size', type=int, default=10, help='interval between training model saves')
    parser.add_argument('--output_size', type=int, default=5, help='interval between training model saves')

    parser.add_argument('--batch_size', type=int, default=3, help='number of workers for parallelizationp')
    parser.add_argument('--lr', type=float, default=0.01, help='training learning rate')
    parser.add_argument('--num_epoch', type=int, default=3, help='number of trianing epochs')
    parser.add_argument('--log_step', type=int, default=10, help='interval between training status logs')
    parser.add_argument('--num_worker', type=int, default=4, help='number of workers for parallelizationp')
    parser.add_argument('--save_step', type=int, default=10, help='interval between training model saves')

    args = parser.parse_args()
    print(args)
    print('-------------Good Justin--------------')
    main(args)
