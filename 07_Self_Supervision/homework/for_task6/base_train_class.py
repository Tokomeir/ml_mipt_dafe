import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
        cum_acc = 0.0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}')
        cum_loss, cum_acc = self._inner_train_step(pbar)

        return [cum_loss, cum_acc]

    def _inner_valid_step(self, pbar):
        pass

    def valid_step(self):
        self.model.eval()

        cum_loss = 0.0
        cum_acc = 0.0

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                    desc=f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}')
        cum_loss, cum_acc = self._inner_valid_step(pbar)

        return [cum_loss, cum_acc]

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