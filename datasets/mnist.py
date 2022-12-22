import os
from PIL import Image 
from common.black_box import read_image_file,read_label_file
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class MNIST(Dataset):
    
    def __init__(self,data_src='./data_src/',category='train',transform=None,target_transform=None) -> None:
        super(MNIST,self).__init__()
        assert (category=='train' or category=="valid" or category=="test")
        self.data_src = data_src
        self.category = category

        self.data, self.targets = self._load_data()
        
        self.transform = transform
        self.target_transform = target_transform

    def _load_data(self):
        if self.category != "test":
            image_file = "train-images-idx3-ubyte"
            label_file = "train-labels-idx1-ubyte"
            
            image = read_image_file(os.path.join(self.raw_folder, image_file))
            label = read_label_file(os.path.join(self.raw_folder, label_file))

            train_image,valid_image,train_label,valid_label = train_test_split(image,label,test_size=0.1)
            if self.category=="train":
                image, label = train_image, train_label
            elif self.category=="valid":
                image, label = valid_image, valid_label
        else:
            image_file = "t10k-images-idx3-ubyte"
            label_file = "t10k-labels-idx1-ubyte"

            image = read_image_file(os.path.join(self.raw_folder, image_file))
            label = read_label_file(os.path.join(self.raw_folder, label_file))

        return image, label
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(),mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.data_src, self.__class__.__name__, 'raw')