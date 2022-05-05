from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10, CIFAR100, FashionMNIST, ImageFolder

def load_val_dataset(opt):
    if opt.dataset == 'MNIST':        
        data_test = MNIST(opt.data,
                        train=False,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ]), download=True)           
        data_test_loader = DataLoader(data_test, batch_size=64, num_workers=32, shuffle=False)
    else:  
        if opt.dataset == 'SVHN':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971)),
            ])
            data_test = SVHN(opt.data,
                            'test',
                            transform=transform_test,
                            download=True)
        elif opt.dataset == 'FASHION':
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2856,), (0.3385,)),
            ])
            data_test = FashionMNIST(opt.data + '/FashionMNIST',
                            train=False,
                            transform=transform_test,
                            download=True)
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if opt.dataset == 'cifar10': 
                data_test = CIFAR10(opt.data,
                                train=False,
                                transform=transform_test,
                                download=True)
            if opt.dataset == 'cifar100':
                data_test = CIFAR100(opt.data,
                                train=False,
                                transform=transform_test,
                                download=True)
            if opt.dataset == 'tiny-imagenet':
                transform_test = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                data_test = ImageFolder(opt.data + 'tiny_imagenet/val/dataset',
                                transform=transform_test)
        data_test_loader = DataLoader(data_test,shuffle=True, batch_size=opt.batch_size, num_workers=32)

    return data_test, data_test_loader

def load_train_dataset(opt):
    if opt.dataset == 'MNIST':        
        data_test = MNIST(opt.data,
                        train=True,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ]), download=True)           
        data_test_loader = DataLoader(data_test, batch_size=64, num_workers=32, shuffle=False)
    else:  
        if opt.dataset == 'SVHN':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971)),
            ])
            data_test = SVHN(opt.data,
                            'test',
                            transform=transform_test,
                            download=True)
        elif opt.dataset == 'FASHION':
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2856,), (0.3385,)),
            ])
            data_test = FashionMNIST(opt.data + '/FashionMNIST',
                            train=True,
                            transform=transform_test,
                            download=True)
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if opt.dataset == 'cifar10': 
                data_test = CIFAR10(opt.data,
                                train=True,
                                transform=transform_test,
                                download=True)
            if opt.dataset == 'cifar100':
                data_test = CIFAR100(opt.data,
                                train=True,
                                transform=transform_test,
                                download=True)
            if opt.dataset == 'tiny-imagenet':
                transform_test = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                data_test = ImageFolder(opt.data + 'tiny_imagenet/train/dataset',
                                transform=transform_test)
        data_test_loader = DataLoader(data_test,shuffle=True, batch_size=opt.batch_size, num_workers=32)

    return data_test, data_test_loader