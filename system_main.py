import os
import pytorch_lightning as pl

from engine.data_drive import DataDrive as DD
from engine.mnist_convnet import MNIST_ConvNet as TM # repalce $ MNIST_ConvNet $

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def init_config(config_args):
    config = config_args.parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    pl.seed_everything(config.rand_seed)
    print(config,"\n")
    return config

class MLSystem():

    def __init__(self, config, splits, monitor="valid_Accuracy") -> None:

        self.config = config
        self.logger = TensorBoardLogger(save_dir="./_logs/",name=TM.__name__)

        self.data_drive = DD(self.config,splits=splits)
        self.model_ckpt = ModelCheckpoint(
            mode="min",
            save_top_k=3,
            save_last=True,
            monitor=monitor,
            dirpath=f"./_ckpt/{TM.__name__}/",
            filename=f"{{epoch:02d}}_{{{monitor}:.2f}}"
        )
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            auto_select_gpus=True,
            min_epochs=1,
            max_epochs=self.config.max_epochs,
            check_val_every_n_epoch=1,
            callbacks=[self.model_ckpt],
            logger=self.logger,
            gradient_clip_val=self.config.clip,
            profiler="simple"
        )

    def fit(self,Task_Model=TM):
        print("start fitting...")
        self.data_drive.setup(stage='fit')
        task_model = Task_Model(config=self.config)
        self.trainer.fit(datamodule=self.data_drive,model=task_model)

    def test(self,Task_Model=TM):
        print("start testing...")
        self.data_drive.setup(stage='test')
        task_model = Task_Model.load_from_checkpoint(self.model_ckpt.best_model_path,config=self.config)
        self.trainer.test(datamodule=self.data_drive,model=task_model)

    def predict(self):
        # print(ckpt_callback1.best_model_path)
        pass

if __name__ == "__main__":
    from config import MyConfigArgs
    from datasets.mnist import MNIST
    
    import torchvision
    config = init_config(config_args=MyConfigArgs())

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    )
    splits = {
        "train":MNIST(category="train",transform=transform),
        "valid":MNIST(category="valid",transform=transform),
        "test": MNIST(category="test",transform=transform)
    }

    system = MLSystem(config=config,splits=splits)
    system.fit()
    system.test()

