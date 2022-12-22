import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DataDrive(pl.LightningDataModule):

    def __init__(self,config,splits=None) -> None:
        super(DataDrive,self).__init__()
        self.config = config
        self.splits = splits # ?

    def prepare_data(self) -> None:
        pass

    def setup(self, stage="fit") -> None:
        assert (stage == 'fit' or stage == 'test')
        if stage == 'fit':
            self.train_set = self.splits["train"] # MNIST(category="train")
            self.valid_set = self.splits["valid"] # MNIST(category="valid")
        elif stage == 'test':
            self.test_set = self.splits["test"] # MNIST(category="test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )