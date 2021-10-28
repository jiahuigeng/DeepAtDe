import os
from torchvision import datasets, transforms
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
DATA_PATH = "./data"
imagen_path = os.path.join(DATA_PATH, "imagen")

tt = transforms.Compose([transforms.ToTensor()])
tp = transforms.Compose([transforms.ToPILImage()])

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs    # img paths
        self.labs = labs    # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab



def imagen_dataset(imagen_path, shape_img):
    images_all = []
    labels_all = []
    # logger.info(imagen_path)
    with open(os.path.join(imagen_path, "samples.txt")) as f:
        for line in f:
            line = line.split("\t")
            images_all.append(os.path.join("./", line[0]))
            labels_all.append(line[1])
    transform = transforms.Compose([transforms.Resize(size=shape_img), tt])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def imagen_dataset_(imagen_path, shape_img):
    images_all = []
    labels_all = []
    # logger.info(imagen_path)
    with open(os.path.join(imagen_path, "samples.txt")) as f:
        for line in f:
            line = line.split("\t")
            images_all.append(os.path.join("./", line[0]))
            labels_all.append(line[1])
    transform = transforms.Compose([transforms.Resize(size=shape_img), tt])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst

def get_dataset(args):
    width = 2**(args.data_scale+5)
    shape_img = (width, width)
    channel = 3
    transform = transforms.Compose([transforms.Resize(size=shape_img), tt])
    if args.dataset == "cifar10":
        num_classes = 10
        trn_set = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)  # , transform=transformations
        tst_set = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
        dst = datasets.CIFAR10(DATA_PATH, download=False)

    elif args.dataset == "cifar100":
        num_classes = 100
        trn_set = datasets.CIFAR100(root=DATA_PATH, train=True, download=True, transform=transform)  # , transform=transformations
        tst_set = datasets.CIFAR100(root=DATA_PATH, train=False, download=True, transform=transform)
        dst = datasets.CIFAR100(DATA_PATH, download=False)

    elif args.dataset == "imagen":
        num_classes = 200
        trn_set = imagen_dataset(imagen_path, shape_img)
        tst_set = imagen_dataset(imagen_path, shape_img)
        dst = imagen_dataset(imagen_path, shape_img)

    if args.task == "train":
        trn_loader = torch.utils.data.DataLoader(trn_set, batch_size=args.batchsize, shuffle=False, drop_last=True)
        tst_loader = torch.utils.data.DataLoader(tst_set, batch_size=args.batchsize, shuffle=False, drop_last=False)
        return trn_loader, tst_loader, channel, num_classes, shape_img
    else:
        return dst, channel, num_classes, shape_img




    # if args.task == "train":
    #     if args.dataset == "cifar10":
    #         trn_set = datasets.CIFAR10(root=DATA_PATH, train=True, download=True) # , transform=transformations
    #         tst_set = datasets.CIFAR10(root=DATA_PATH, train=False, download=True)
    #     elif args.dataset == "cifar100":
    #         trn_set = datasets.CIFAR10(root=DATA_PATH, train=True, download=True) # , transform=transformations
    #         tst_set = datasets.CIFAR10(root=DATA_PATH, train=False, download=True)
    #     else:
    #         trn_set = imagen_dataset(imagen_path, shape_img)
    #         tst_set =  imagen_dataset(imagen_path, shape_img)
    #
    #     trn_loader = torch.utils.data.DataLoader(trn_set, batch_size=args.batchsize, shuffle=True, drop_last=True)
    #     tst_loader = torch.utils.data.DataLoader(tst_set, batch_size=args.batchsize, shuffle=False, drop_last=False)
    #     return trn_loader, tst_loader, channel, num_classes
    #
    #
    # dst, channel, num_classes, hidden =None, None, None, None
    #
    #
    # hidden = int(shape_img[0] * shape_img[1] * 3 / 4)
    # if args.dataset == "cifar10":
    #     num_classes = 10
    #     # dst = datasets.CIFAR10(DATA_PATH, download=True, transform=tt)
    #     dst = datasets.CIFAR10(DATA_PATH, download=False)
    #
    # elif args.dataset == "cifar100":
    #     num_classes = 100
    #     dst = datasets.CIFAR100(DATA_PATH, download=False)
    #     # dst = datasets.CIFAR100(DATA_PATH, download=False, transform=tt)
    #
    # elif args.dataset == "imagen":
    #     num_classes = 200
    #     dst = imagen_dataset(imagen_path, shape_img)
    #
    # else:
    #     NotImplementedError("dataset undefined")
    # # logger.info("size:", width, "hidden:", hidden)
    # return dst, channel, num_classes, hidden

def data_initializer(data_init):
    dummy_data = None
    if data_init == "uniform":
        return dummy_data



