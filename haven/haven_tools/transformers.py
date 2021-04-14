# Transformers
import collections
import numpy as np
import torch

from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_transformer(transform, split):
    if transform == "resize_normalize":
        normalize_transform = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize_transform])

    if transform == "rgb_normalize":
        normalize_transform = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose([transforms.ToTensor(), normalize_transform])

    elif transform == "resize":
        return ComposeJoint(
            [
                [transforms.Resize((224, 224)), transforms.Resize((224, 224)), None],
                [transforms.ToTensor(), None, transforms.ToTensor()],
                [transforms.Normalize(mean=mean, std=std), None, None],
                [None, ToLong(), None],
            ]
        )

    elif transform == "basic":
        transform = ComposeJoint(
            [[transforms.ToTensor(), None], [transforms.Normalize(mean=mean, std=std), None], [None, ToLong()]]
        )
        return transform

    elif transform == "basic_flip":
        transform = ComposeJoint(
            [
                RandomHorizontalFlipJoint(),
                [transforms.ToTensor(), None, transforms.ToTensor()],
                [transforms.Normalize(mean=mean, std=std), None, None],
                [None, ToLong(), None],
            ]
        )
        return transform


# helpers


class ComposeJoint(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = self._iterate_transforms(transform, x)

        return x

    def _iterate_transforms(self, transforms, x):
        """Credit @fmassa:
        https://gist.github.com/fmassa/3df79c93e82704def7879b2f77cd45de
        """
        if isinstance(transforms, collections.Iterable):
            for i, transform in enumerate(transforms):
                x[i] = self._iterate_transforms(transform, x[i])
        else:

            if transforms is not None:
                x = transforms(x)

        return x


class ToLong(object):
    def __call__(self, x):
        return torch.LongTensor(np.asarray(x))
