from for_task6.base_train_class import BaseTrainProcess
from for_task6.model import * 
from for_task6.SimCLR_load_dataset import *
import torch.optim as optim

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

        cum_loss /= pbar.total
        return [cum_loss, 0]

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

        cum_loss /= pbar.total
        return [cum_loss, 0]

    def scheduler_step(self):
        if self.current_epoch < 10:
            self.warmupscheduler.step()
        else:
            self.mainscheduler.step()
