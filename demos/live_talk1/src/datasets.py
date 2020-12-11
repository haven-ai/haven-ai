import torchvision


def get_dataset(dataset_name, train_flag, datadir):
    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                                             download=True,
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(
                                                     (0.5,), (0.5,)),
                                             ])
                                             )

    if dataset_name == "fashionmnist":
        dataset = torchvision.datasets.FashionMNIST(datadir, train=train_flag,
                                                    download=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.5,), (0.5,)),
                                                    ])
                                                    )
    return dataset
