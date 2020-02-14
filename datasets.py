from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import os.path as osp


def setup_data_loader(dataset, args):
    if dataset == 'ImageNet-C' or dataset == 'IN-C':
        return setup_imagenetc_data_loader(args)
    elif dataset == 'ImageNet' or dataset == 'IN':
        return setup_imagenet_data_loader(args)
    elif dataset == 'SIN':
        return setup_sin_data_loader(args)
    else:
        raise ValueError(f'Dataset {dataset} is not available')


def setup_imagenet_data_loader(args):
    n_worker = args.workers
    valdir = osp.join(args.datadir_clean, 'val')
    traindir = osp.join(args.datadir_clean, 'train')
    
    if args.arch == 'inceptionV3':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        val_transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor()
            ])
    else:
        # transforms for a resnet
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])
    
    train_dataset = datasets.ImageFolder(traindir, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_worker)
    val_dataset = datasets.ImageFolder(valdir, val_transforms)    
    indices_subsample = np.random.choice(len(val_dataset), args.test_subset_size, replace=False)
    test_loader_subsample = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset=val_dataset, indices=indices_subsample),
        batch_size=args.test_batch_size, shuffle=False, num_workers=n_worker)
    test_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.test_batch_size, shuffle=False, num_workers=n_worker)
    train_loader_retrain = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=n_worker)

    return train_loader, test_loader, test_loader_subsample, train_loader_retrain


def setup_imagenetc_data_loader(args):
    n_worker = args.workers
    data_loaders_names = {
        'Brightness': 'brightness',
        'Contrast': 'contrast',
        'Defocus Blur': 'defocus_blur',
        'Elastic Transform': 'elastic_transform',
        'Fog': 'fog',
        'Frost': 'frost',
        'Gaussian Noise': 'gaussian_noise',
        'Glass Blur': 'glass_blur',
        'Impulse Noise': 'impulse_noise',
        'JPEG Compression': 'jpeg_compression',
        'Motion Blur': 'motion_blur',
        'Pixelate': 'pixelate',
        'Shot Noise': 'shot_noise',
        'Snow': 'snow',
        'Zoom Blur': 'zoom_blur'
    }
    
    data_loaders = {}    
    for name, path in data_loaders_names.items():
        data_loaders[name] = {}
        for severity in range(1, 6):
            dset = datasets.ImageFolder(args.imagenetc_path + path +
                                        '/' + str(severity) + '/',
                                        transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))
            data_loaders[name][str(severity)] = torch.utils.data.DataLoader(
                dset, batch_size=args.test_batch_size, shuffle=True, num_workers=n_worker)
    return data_loaders   
    

def setup_sin_data_loader(args):
    n_worker = args.workers
    traindir = osp.join(args.datadir_sin, 'train')
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.sin_batch_size, shuffle=True, num_workers=n_worker)
    
    return train_loader
