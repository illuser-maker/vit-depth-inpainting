from os import name
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from pytorch_lightning import LightningModule
from model.coarse_net import CoarseNet
from model.loss import MaskedL1Loss, MaskedMSELoss

class CoarseNetTraining(LightningModule):
    def __init__(self, token_channels, token_hidden, image_shape, depth=3, heads=16):
        super().__init__()
        self.model = CoarseNet(token_channels, token_hidden, image_shape, depth, heads)
        self.loss = MaskedL1Loss()
    
    def training_step(self, batch, batch_idx):
        rgb, dep, gt = batch['rgb'], batch['dep'], batch['gt']
        mask = (dep > 0).float()
        coarse_depth = self.model(rgb, dep, mask)
        loss = self.loss(coarse_depth, gt)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, dep, gt = batch['rgb'], batch['dep'], batch['gt']
        mask = (dep > 0).float()
        coarse_depth = self.model(rgb, dep, mask)
        loss = self.loss(coarse_depth, gt)

        mask = (dep > 0)
        diff_sqr = torch.pow(coarse_depth[mask] - gt[mask], 2)
        num_valid = mask.sum()
        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)

        diff = torch.abs(gt[mask] - coarse_depth[mask]) / gt[mask]
        rel = diff.median()

        metrics = {'val_loss': loss, 'val_rmse': rmse, 'val_rel': rel}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=5e-4)

if name == "__main__":
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from data.nyu import NYU

    train_data = NYU('D:\\Jupyter_D\\VKR\\nyudepthv2', 'D:\\Jupyter_D\\VKR\\NLSPN_ECCV20-master\\data_json\\nyu.json', 
            True, 35000, 'train', coarse=True,  coarse_height=200)
    val_data = NYU('D:\\Jupyter_D\\VKR\\nyudepthv2', 'D:\\Jupyter_D\\VKR\\NLSPN_ECCV20-master\\data_json\\nyu.json', 
            True, 35000, 'val', coarse=True,  coarse_height=200)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    task = CoarseNetTraining(128, 64, train_data[0]['dep'].shape[1:])
    task.cuda()
    
    trainer = pl.Trainer(gpus=1, max_epochs=50)
    trainer.fit(task, train_loader, val_loader)
    torch.save(task.model.state_dict(), 'CoarseNet_trained')