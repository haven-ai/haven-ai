import torchvision

from torch.utils.data import DataLoader


def get_loader(dataset_name, datadir, split, batch_size=32):
    if dataset_name == 'mnist':
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))])

        if split == 'train':
            train = True 
        else:
            train = False 

        dataset = torchvision.datasets.MNIST(datadir,
                                                train=train,
                                                download=True,
                                                transform=transform)
        loader = DataLoader(dataset, shuffle=True,
                                  batch_size=batch_size)
        
        return loader