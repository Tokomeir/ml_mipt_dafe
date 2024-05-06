import random
from pathlib import Path
from time import gmtime, strftime

import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2


from for_task6.model import * 
from for_task6.load_dataset import *

class BaseTrainProcess:
    def __init__(self, hyp):
        self.best_loss = 1e100
        self.best_acc = 0.0
        self.current_epoch = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.hyp = hyp

        self.lr_scheduler: Optional[torch.optim.lr_scheduler] = None
        self.model: Optional[torch.nn.modules] = None
        self.optimizer: Optional[torch.optim] = None
        self.criterion: Optional[torch.nn.modules] = None

        self.train_loader: Optional[Dataloader] = None
        self.valid_loader: Optional[Dataloader] = None

        #self.init_params()

    def _inner_init_data(self, X_train, y_train, X_val, y_val, train_transform, valid_transform):
        pass

    def _init_data(self, X_train, y_train, X_val, y_val, train_transform, valid_transform):
        train_dataset, valid_dataset = self._inner_init_data(X_train, y_train, X_val, y_val, train_transform, valid_transform) 
        print('Train size:', len(train_dataset), 'Valid size:', len(valid_dataset))

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.hyp['batch_size'],
                                       shuffle=True,
                                       num_workers=self.hyp['n_workers'],
                                       pin_memory=True,
                                       drop_last=True
                                      )

        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.hyp['batch_size'],
                                       shuffle=True,
                                       num_workers=self.hyp['n_workers'],
                                       pin_memory=True,
                                       drop_last=True
                                      )

    def _inner_init_model(self):
        pass

    def _inner_init_criterion(self):
        pass


    def _inner_init_scheduler(self):
        pass

    def _init_model(self):
        self._inner_init_model()
        model_params = [params for params in self.model.parameters() if params.requires_grad]
        # self.optimizer = LARS(model_params, lr=0.2, weight_decay=1e-4)
        self.optimizer = torch.optim.AdamW(model_params, lr=self.hyp['lr'], weight_decay=self.hyp['weight_decay'])
        self._inner_init_scheduler()
        self._inner_init_criterion()

    def init_params(self, X_train, y_train, X_val, y_val, train_transform=None, valid_transform=None):
        self._init_data(X_train, y_train, X_val, y_val, train_transform, valid_transform)
        self._init_model()

    def save_checkpoint(self, loss_valid, path):
        if loss_valid[0] <= self.best_loss:
            self.best_loss = loss_valid[0]
            self.save_model(path)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.mainscheduler.state_dict()
        }, path)

    def _inner_train_step(self, pbar):
        pass

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()

        cum_loss = 0.0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}')
        cum_loss = self._inner_train_step(pbar)

        cum_loss /= len(self.train_loader)
        return [cum_loss]

    def _inner_valid_step(self, pbar):
        pass

    def valid_step(self):
        self.model.eval()

        cum_loss = 0.0

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                    desc=f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}')
        cum_loss = self._inner_valid_step(pbar)

        cum_loss /= len(self.valid_loader)
        return [cum_loss]

    def scheduler_step(self):
        pass

    def run(self, should_save_checkpoint = False):
        best_w_path = 'best.pt'
        last_w_path = 'last.pt'
        
        self.train_losses = []
        self.valid_losses = []

        for epoch in range(self.hyp['epochs']):
            self.current_epoch = epoch

            loss_train = self.train_step()
            self.train_losses.append(loss_train)
                
            self.scheduler_step()

            lr = self.optimizer.param_groups[0]["lr"]

            loss_valid = self.valid_step()
            self.valid_losses.append(loss_valid)
            
            if should_save_checkpoint:
                self.save_checkpoint(loss_valid, best_w_path)
            
        if should_save_checkpoint:
            self.save_model(last_w_path)
        torch.cuda.empty_cache()

        return self.train_losses, self.valid_losses


class  SimCLRTrainProcess(BaseTrainProcess):
    def __init__(self, hyp):
        super(SimCLRTrainProcess, self).__init__(hyp)

    def _inner_init_data(self, X_train, y_train, X_val, y_val, train_transform, valid_transform):
        return load_datasets(X_train, y_train, X_val, y_val, train_transform, valid_transform, crop_coef=1.4)
    
    def _inner_init_model(self):
        self.model = PreModel()
        self.model.to(self.device)

    def _inner_init_scheduler(self):
        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (epoch + 1) / 10.0)
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            500,
            eta_min=0.05,
            last_epoch=-1,
        )

    def _inner_init_criterion(self):
        self.criterion = SimCLR_Loss(batch_size=self.hyp['batch_size'], 
                                     temperature=self.hyp['temperature']).to(self.device)

    def _inner_train_step(self, pbar):
        cum_loss = 0.0
        proc_loss = 0.0
        for idx, (xi, xj, _, _) in pbar:
            xi, xj = xi.to(self.device), xj.to(self.device)

            with torch.set_grad_enabled(True):
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss

            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            s = f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}'
            pbar.set_description(s)

        return cum_loss

    def _inner_valid_step(self, pbar):
        cum_loss = 0.0
        proc_loss = 0.0
        for idx, (xi, xj, _, _) in pbar:
            xi, xj = xi.to(self.device), xj.to(self.device)

            with torch.set_grad_enabled(False):
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss

            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            s = f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}'
            pbar.set_description(s)

        return cum_loss

    def scheduler_step(self):
        if self.current_epoch < 10:
            self.warmupscheduler.step()
        else:
            self.mainscheduler.step()
