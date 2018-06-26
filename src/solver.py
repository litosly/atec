#####################################
# Train/Evaluate
# Author: Justin
# Modifier: Litos
# Date modified: 2018-06-26
#####################################


import os 
import argparse 
import numpy as np 

import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

from model import Encoder, Comparator, AtecModel
from dataloader import get_dataloader#, get_test_dataloader
from torch.nn.utils.rnn import pad_packed_sequence


class Solver(object):
    """
    """
    def __init__(self, config, reuse=False):
        self.config = config 
        self.reuse = reuse 

        self.build_model()
        if reuse:
            # remember to manually load_data by specifying modes=[...]
            self.load_model(self.config.model_path)
        else:
            self.load_data()

    def load_data(self, modes=["train", "valid", "test"]):
        self.train_dataloader = get_dataloader(self.config, mode="train") if "train" in modes else None
        self.valid_dataloader = get_dataloader(self.config, mode="valid") if "valid" in modes else None
        self.test_dataloader = get_dataloader(self.config, mode="test") if "test" in modes else None

    def build_model(self):
        # can add preprocessing logic layer here 
        self.model = AtecModel(self.config)

        # training stuff 
        self.criterion = nn.CrossEntropyLoss()
        self.trainable_params = list(self.model.encoder.parameters())+list(self.model.comparator.parameters())
        self.optimizer = Adam(self.trainable_params, lr=self.config.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[10, 20, 30], gamma=0.1)

        # bookkeeping stuff 
        self.writer =  SummaryWriter(self.config.log_dir)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), model_path)

    def train(self):
        for epoch in range(self.config.num_epoch):
            self.scheduler.step()
            self.train_step(epoch)

            # save the model per epoch, only save parameters 
            if (epoch+1) % self.config.save_step == 0:
                model_path = os.path.join(self.config.model_dir, 'model-%d.pkl' %(epoch+1))
                self.save_model(self.model, model_path)
            
            # log model performance over epochs
            valid_acc = self.evaluate(self.valid_dataloader)
            test_acc = self.evaluate((self.test_dataloader))
            self.writer.add_scalars('data/accuracy', {'valid': valid_acc.data[0],
                'test': test_acc.data[0]}, epoch)
            print( 'Epoch [%d/%d], valid acc: %.4f, test acc: %.4f' 
                      % (epoch+1, self.config.num_epoch, valid_acc.data[0], test_acc.data[0]))

        self.close_log(self.writer)

    def train_step(self, epoch):
        total_steps = len(self.train_dataloader.dataset) // self.config.batch_size + 1
        for i, (data, labels, indices, lengths) in enumerate(self.train_dataloader):
            logits = self.model(data, indices)
            preds = torch.argmax(logits, dim=1).long()
            loss = self.criterion(logits, labels)
            acc = self.metric(preds, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log loss, could visualize in tensorboard if needed 
            if (i+1) % self.config.log_step == 0:
                self.writer.add_scalar('data/loss', loss.data[0], epoch*total_steps+i)
                self.writer.add_scalar('data/train_acc', acc.data[0], epoch*total_steps+i)
                print( 'Epoch [%d/%d], Step[%d/%d], loss: %.4f, acc: %.4f' 
                      % (epoch+1, self.config.num_epoch, i+1, total_steps, loss.data[0], acc.data[0]))

    def inference(self, data, indices):
        logits = self.model(data, indices)
        preds = torch.argmax(logits, dim=1).long()
        return preds 

    def evaluate(self, dataloader):
        accs = []
        for i, (data, labels, indices, lengths) in enumerate(dataloader):
            preds = self.inference(data, indices)
            acc = self.metric(preds, labels)
            accs.append(acc.data[0])
        return sum(accs) / len(accs)

    def metric(self, preds, labels):
        # accuracy 
        res = torch.eq(preds, labels)
        acc = torch.sum(res) /  len(res)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        acc_f1 = (tp+tn)/(tp+tn+fp+fn)
        f1 = 2*precision*recall/(precision+recall)
        return acc 

    def close_log(self, writer, log_path="./all_scalars.json"):
        # export scalar data to JSON for external processing
        writer.export_scalars_to_json(log_path)
        writer.close()







