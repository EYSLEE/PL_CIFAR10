import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
import torchmetrics
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
wandb.finish()
wandb_logger=WandbLogger()




class LitCIFAR10(LightningModule):
    def __init__(self,lr=1e-3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)          
        )
        self.fc_layer = nn.Sequential(
        	# [100,64*3*3] -> [100,100]
            nn.Linear(1024,100),                                              
            nn.ReLU(),
            # [100,100] -> [100,10]
            nn.Linear(100,10)                                                   
        )       
        
        self.lr=lr
        self.accuracy = Accuracy()
        self.save_hyperparameters()
    def forward(self,x):
        batch_size,channels,width,height=x.size()
        x=self.layer(x)
        x=x.view(batch_size,-1)
        x=self.fc_layer(x)
       
        return x

    def training_step(self,batch ,batch_idx):
        x,y=batch
        logits=self(x)
        loss=F.nll_loss(logits,y)
        self.log('train_loss',loss)
        self.log('train_acc', self.accuracy(logits, y))
        return loss 
    def validation_step(self,batch,batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.nll_loss(logits, y)
        self.log('valid_loss', val_loss)

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        test_loss = F.nll_loss(logits, y)

        self.log('test_loss', test_loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

class CIFAR10DataModule(LightningDataModule):

    def __init__(self, data_dir='./', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called one ecah GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = train_test_split(cifar10_train)
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        '''returns training dataloader'''
        cifar10_train = DataLoader(self.cifar10_train, batch_size=self.batch_size)
        return cifar10_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        cifar10_val = DataLoader(self.cifar10_val, batch_size=self.batch_size)
        return cifar10_val

    def test_dataloader(self):
        '''returns test dataloader'''
        cifar10_test = DataLoader(self.cifar10_test, batch_size=self.batch_size)
        return cifar10_test



wandb.login()
wandb_logger=WandbLogger(project='CIFAR10')

# dataset=CIFAR10(os.getcwd(),download=True,transform=transforms.ToTensor())
# train,test=train_test_split(dataset,test_size=0.2)
# train_loader=DataLoader(train,batch_size=64,num_workers=8)
# test_dataloader=DataLoader(test,batch_size=64,num_workers=8)


model=LitCIFAR10()
cifar=CIFAR10DataModule()
cifar.setup()

trainer=pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu',
    devices=1,
    max_epochs=10
)
trainer.fit(model,cifar)
trainer.test(model,datamodule=cifar)
wandb.finish()
