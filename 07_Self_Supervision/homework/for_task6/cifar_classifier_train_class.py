import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from for_task6.base_train_class import BaseTrainProcess
from for_task6.cifar_classifier import CustomDataset, CIFARСlassifier

class CIFARClassifierTrainProcess(BaseTrainProcess):
    def __init__(self, hyp, encoder, emb_size):
        super(CIFARClassifierTrainProcess, self).__init__(hyp)
        self.encoder = encoder
        self.emb_size = emb_size
        
    def _inner_init_data(self, X_train, y_train, X_val, y_val, train_transform, valid_transform):
        train_dataset = CustomDataset(X_train, y_train, train_transform)
        valid_dataset = CustomDataset(X_val, y_val, valid_transform)
        return train_dataset, valid_dataset
    
    def _inner_init_model(self):
        self.model = CIFARСlassifier(self.encoder, self.emb_size)
        self.model.to(self.device)

    def _inner_init_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        
    def _inner_init_criterion(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _inner_train_step(self, pbar):
        cum_loss = 0.0
        proc_loss = 0.0
        
        cum_acc = 0.0
        proc_acc = 0.0
        
        for idx, (x, y) in pbar:
            x, y = x.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(True):
                out = self.model(x)
                loss = self.criterion(out, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss
            
            prob, pred = torch.softmax(out.detach(), dim=1).topk(k=1)
            acc = f1_score(y.detach().cpu(), pred.detach().cpu(), average='macro')
            cum_acc += acc
            
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)
            proc_acc = (proc_acc * idx + acc) / (idx + 1)

            s = f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}, F1: {proc_acc:4.3f}, Loss: {proc_loss:4.3f}'
            pbar.set_description(s)
            
        cum_loss /= pbar.total
        cum_acc /= pbar.total
        return [cum_loss, cum_acc]

    def _inner_valid_step(self, pbar):
        cum_loss = 0.0
        proc_loss = 0.0
        
        cum_acc = 0.0
        proc_acc = 0.0
        
        for idx, (x, y) in pbar:
            x, y = x.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(False):
                out = self.model(x)
                loss = self.criterion(out, y)

            cur_loss = loss.detach().cpu().numpy()
            prob, pred = torch.softmax(out.detach(), dim=1).topk(k=1)
            acc = f1_score(y.detach().cpu(), pred.detach().cpu(), average='macro')
            cum_acc += acc
            cum_loss += cur_loss

            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)
            proc_acc = (proc_acc * idx + acc) / (idx + 1)
            
            s = f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}, F1: {proc_acc:4.3f}, Loss: {proc_loss:4.3f}'
            pbar.set_description(s)
            
        cum_loss /= pbar.total
        cum_acc /= pbar.total
        return [cum_loss, cum_acc]
    
    def scheduler_step(self):
        if self.train_losses:
            self.lr_scheduler.step(self.train_losses[-1][0])