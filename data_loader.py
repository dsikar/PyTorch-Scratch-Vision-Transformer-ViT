import os
import torch
from torchvision import datasets, transforms


def get_transforms(args, is_train=True):
    transforms_list = [
        transforms.Resize([args.image_size, args.image_size])
    ]
    
    if is_train:
        if args.augmentation in ['standard', 'randaugment']:
            if args.dataset in ['mnist']:
                transforms_list.append(transforms.RandomCrop(args.image_size, padding=2))
            elif args.dataset in ['fashionmnist', 'cifar10', 'cifar100']:
                transforms_list.append(transforms.RandomCrop(args.image_size, padding=4))
                transforms_list.append(transforms.RandomHorizontalFlip())
        
        if args.augmentation == 'randaugment':
            if args.dataset in ['cifar10', 'cifar100']:
                transforms_list.append(transforms.RandAugment())

    transforms_list.append(transforms.ToTensor())
    
    # Dataset specific normalization
    if args.dataset == 'mnist' or args.dataset == 'fashionmnist':
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif args.dataset == 'svhn':
        transforms_list.append(transforms.Normalize([0.4376821, 0.4437697, 0.47280442], 
                                                 [0.19803012, 0.20101562, 0.19703614]))
    elif args.dataset == 'cifar10':
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], 
                                                 [0.2470, 0.2435, 0.2616]))
    elif args.dataset == 'cifar100':
        transforms_list.append(transforms.Normalize([0.5071, 0.4867, 0.4408], 
                                                 [0.2675, 0.2565, 0.2761]))
    
    return transforms.Compose(transforms_list)


def get_loader(args):
    if args.dataset == 'mnist':
        train = datasets.MNIST(os.path.join(args.data_path, args.dataset), 
                             train=True, download=True,
                             transform=get_transforms(args, is_train=True))
        test = datasets.MNIST(os.path.join(args.data_path, args.dataset),
                            train=False, download=True,
                            transform=get_transforms(args, is_train=False))
    
    elif args.dataset == 'fashionmnist':
        train = datasets.FashionMNIST(os.path.join(args.data_path, args.dataset),
                                    train=True, download=True,
                                    transform=get_transforms(args, is_train=True))
        test = datasets.FashionMNIST(os.path.join(args.data_path, args.dataset),
                                   train=False, download=True,
                                   transform=get_transforms(args, is_train=False))
    
    elif args.dataset == 'svhn':
        train = datasets.SVHN(os.path.join(args.data_path, args.dataset),
                            split='train', download=True,
                            transform=get_transforms(args, is_train=True))
        test = datasets.SVHN(os.path.join(args.data_path, args.dataset),
                           split='test', download=True,
                           transform=get_transforms(args, is_train=False))

    elif args.dataset == 'cifar10':
        train = datasets.CIFAR10(os.path.join(args.data_path, args.dataset),
                               train=True, download=True,
                               transform=get_transforms(args, is_train=True))
        test = datasets.CIFAR10(os.path.join(args.data_path, args.dataset),
                              train=False, download=True,
                              transform=get_transforms(args, is_train=False))

    elif args.dataset == 'cifar100':
        train = datasets.CIFAR100(os.path.join(args.data_path, args.dataset),
                                train=True, download=True,
                                transform=get_transforms(args, is_train=True))
        test = datasets.CIFAR100(os.path.join(args.data_path, args.dataset),
                               train=False, download=True,
                               transform=get_transforms(args, is_train=False))

    else:
        print("Unknown dataset")
        exit(0)

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=args.batch_size*2,
                                              shuffle=False,
                                              num_workers=args.n_workers,
                                              drop_last=False)

    return train_loader, test_loader
