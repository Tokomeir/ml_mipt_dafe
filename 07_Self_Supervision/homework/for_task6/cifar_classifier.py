import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from for_task6.model import ProjectionHead

class CIFARСlassifier(nn.Module):
    def __init__(self, encoder, emb_size, class_num = 10):
        super(CIFARСlassifier, self).__init__()
        self.encoder = encoder
        
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.projector = ProjectionHead(emb_size, 2048, class_num)
    
    def forward(self,x):
        out = self.encoder(x)
        xp = self.projector(torch.squeeze(out))
        return xp
    
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, transform_augment=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform_augment = transform_augment
                    
    def __len__(self):
        return len(self.x_data) 
    
    def __getitem__(self, item):
        image = self.x_data[item]
        label = self.y_data[item]
        
        if self.transform_augment is not None:
            image = (image * 255).astype(np.uint8)
            image = self.transform_augment(image=image)['image']
        else:
            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1)  # switch to dim, h, w
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label