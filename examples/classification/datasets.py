def get_datasets():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))])

    # train set
    train_set = torchvision.datasets.MNIST(savedir_base,
                                           train=True,
                                           download=True,
                                           transform=transform)

    val_set = torchvision.datasets.MNIST(savedir_base, train=False,
                                         download=True, transform=transform)