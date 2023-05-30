import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class MyData(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_path = glob.glob(os.path.join(image_dir, '*.raw'))
        self.label_path = glob.glob(os.path.join(label_dir, '*.raw'))

        # 因为label 和 image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_path.sort()
        self.label_path.sort()

    def __getitem__(self, idx):
        image_item_path = self.image_path[idx]
        label_item_path = self.label_path[idx]

        image = np.fromfile(image_item_path, dtype=np.float32).reshape(1, 512, 512)
        label = np.fromfile(label_item_path, dtype=np.float32).reshape(1, 512, 512)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        assert len(self.image_path) == len(self.label_path)
        return len(self.image_path)


if __name__ == '__main__':
    image_dir = "../data/train/image"
    label_dir = "../data/train/label"
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    dataset = MyData(image_dir, label_dir, transform=None)
    print("数据个数：", len(dataset))

    pic = dataset.__getitem__(1)["image"]
    plt.imshow(pic[0, :, :], cmap="gray")
    plt.show()

    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    print("batch的个数：", len(dataloader))

    writer = SummaryWriter("../logs/dataset_logs")
    step = 0

    for batch in dataloader:
        image = batch["image"]
        label = batch["label"]
        print(image.shape)
        # print(label.shape)
        writer.add_images("dataloader", image, step)
        step = step + 1

    writer.close()
