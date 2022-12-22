import torch
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision,Recall

from models.convnet import ConvNet
class MNIST_ConvNet(pl.LightningModule):
    
    @staticmethod
    def get_metrics(prefix=""):
        metrics = MetricCollection([
            Accuracy(num_classes=10,multiclass=True),
            Precision(num_classes=10,multiclass=True),
            Recall(num_classes=10,multiclass=True)
        ],prefix=prefix)
        return metrics

    def __init__(self,config) -> None:
        super(MNIST_ConvNet,self).__init__()

        self.config = config
        
        print(f"{self.__class__.__name__} : load model")
        self.model = ConvNet()

        self.train_metrics = self.get_metrics(prefix="train_")
        self.valid_metrics = self.get_metrics(prefix="valid_")
        self.test_metrics = self.get_metrics(prefix="test_")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.config.lr,weight_decay=self.config.wd)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:self.config.lrd**epoch)
        return {'optimizer':optimizer,'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred,y)
        perf = self.train_metrics(pred,y)
        self.log_dict(perf,on_step=False,on_epoch=True,prog_bar=True)
        return {"loss":loss}

    @torch.no_grad()
    def _shared_eval_step(self,batch,metrics):
        x, y = batch
        pred = self.model(x)
        perf = metrics(pred,y)
        return perf

    def validation_step(self, batch, batch_idx):
        perf = self._shared_eval_step(batch,metrics=self.valid_metrics)
        self.log_dict(perf,on_step=False,on_epoch=True,prog_bar=True)

    def test_step(self, batch, batch_idx):
        perf = self._shared_eval_step(batch,metrics=self.test_metrics)
        self.log_dict(perf,on_step=False,on_epoch=True,prog_bar=True)
