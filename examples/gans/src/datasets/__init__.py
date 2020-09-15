import torchvision.datasets as datasets
from torchvision import transforms
from torchmeta.datasets import Omniglot
from torchmeta.datasets import DoubleMNIST
from torchmeta.transforms import Categorical
from torchvision.transforms import Compose, Resize, ToTensor


def get_dataset(dataset_name, dataset_path=None, image_size=64):
    if dataset_name in ['MNIST']:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset_train = datasets.MNIST(root=dataset_path,
                                       download=True,
                                       transform=transform,
                                       train=True)
        dataset_test = datasets.MNIST(root=dataset_path,
                                      download=True,
                                      transform=transform,
                                      train=False)
        num_channels = 1
        num_train_classes = len(dataset_train.classes)
        num_test_classes = len(dataset_test.classes)

    elif dataset_name in ['Omniglot']:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset_train = Omniglot(root=dataset_path,
                           num_classes_per_task=1,
                           transform=transform,
                           target_transform=None,
                           meta_train=True,
                           download=True,
                           use_vinyals_split=False)
        dataset_test = Omniglot(root=dataset_path,
                           num_classes_per_task=1,
                           transform=transform,
                           target_transform=None,
                           meta_test=True,
                           download=True,
                           use_vinyals_split=False)
        num_channels = 1

        num_train_classes = dataset_train.dataset.num_classes
        num_test_classes = dataset_test.dataset.num_classes

    elif dataset_name in ['cifar10']:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset_train = datasets.CIFAR10(root=dataset_path,
                                         download=True,
                                         transform=transform,
                                         train=True)
        dataset_test = datasets.CIFAR10(root=dataset_path,
                                        download=True,
                                        transform=transform,
                                        train=False)
        num_channels = 3
        num_train_classes = len(dataset_train.classes)
        num_test_classes = len(dataset_test.classes)
    
    elif dataset_name in ['celeba']:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset_train = datasets.CelebA(root=dataset_path,
                                        download=True,
                                        transform=transform,
                                        split='train')
        dataset_test = datasets.CelebA(root=dataset_path,
                                       download=True,
                                       transform=transform,
                                       split='test')
        num_channels = 3

        # TODO: revisit this if it's true?
        num_train_classes = 0
        num_test_classes = 0

    elif dataset_name in ['DoubleMNIST']:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset_train = DoubleMNIST(root=dataset_path,
                           num_classes_per_task=1,
                           transform=transform,
                           target_transform=None,
                           meta_train=True,
                           download=True)
        dataset_test = DoubleMNIST(root=dataset_path,
                           num_classes_per_task=1,
                           transform=transform,
                           target_transform=None,
                           meta_test=True,
                           download=True)
        num_channels = 1

        num_train_classes = dataset_train.dataset.num_classes
        num_test_classes = dataset_test.dataset.num_classes

    return dataset_train, dataset_test, num_channels, num_train_classes, \
           num_test_classes
