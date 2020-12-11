import torch
import tqdm
import pylab as plt

from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, exp_dict, device):
        super().__init__()
        # - define network
        self.network = Mlp()
        self.to(device)

        # - define optimizer
        opt_dict = exp_dict['opt']
        if opt_dict['name'] == "adam":
            self.opt = torch.optim.Adam(
                self.parameters(), lr=opt_dict['lr'],  betas=(0.9, 0.99))

        elif opt_dict['name'] == "adagrad":
            self.opt = torch.optim.Adagrad(
                self.parameters(), lr=opt_dict['lr'])

        elif opt_dict['name'] == 'sgd':
            self.opt = torch.optim.SGD(self.parameters(), lr=opt_dict['lr'])

        self.device = device
        self.exp_dict = exp_dict

    @torch.no_grad()
    def val_on_dataset(self, dataset, metric_name):
        self.eval()

        if metric_name == 'softmax_acc':
            metric_function = softmax_accuracy
        elif metric_name == 'softmax_loss':
            metric_function = softmax_loss

        loader = torch.utils.data.DataLoader(
            dataset, drop_last=False, batch_size=128)

        score_sum = 0.
        for batch in tqdm.tqdm(loader, desc='Computing %s' % metric_name):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            score_sum += metric_function(self.network,
                                         images, labels).item() * images.shape[0]

        score = float(score_sum / len(loader.dataset))

        return score

    def train_on_dataset(self, dataset):
        self.train()

        sampler = torch.utils.data.RandomSampler(
            data_source=dataset, replacement=True,
            num_samples=10000)

        loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                             batch_size=self.exp_dict["batch_size"],
                                             drop_last=True)

        for batch in tqdm.tqdm(loader, desc='Training'):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            loss = softmax_loss(self.network, images, labels)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return loss

    @torch.no_grad()
    def vis_on_dataset(self, dataset, fname):
        self.eval()

        f = plt.figure()
        n_images = 5
        for i in range(n_images):
            images, labels = dataset[i]
            images = images[None].to(self.device)
            labels = torch.as_tensor(labels)[None].to(self.device)
            probs = torch.softmax(self.network.forward(
                images.view(images.shape[0], -1)), dim=1)
            score, pred = probs.max(dim=1)

            # figure
            f.add_subplot(1, n_images, i + 1)
            plt.title('P: %d - G: %d' % (pred[0], labels[0]))
            plt.imshow(images[0, 0, :, :, None].cpu().numpy(), cmap='gray')
            plt.axis('off')

        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close()

    def get_state_dict(self):
        return {'opt': self.opt.parameters(),
                'network': self.network.parameters()}

    def set_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])
        self.opt.load_state_dict(state_dict['opt'])


# -------
# Metrics
# -------
def softmax_accuracy(network, images, labels):
    logits = network(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc


def softmax_loss(network, images, labels):
    logits = network(images)
    loss = torch.nn.CrossEntropyLoss()(logits, labels.long().view(-1))

    return loss

# --------
# Networks
# --------


class Mlp(nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[512, 256]):
        super().__init__()

        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=True) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], 10, bias=True)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = F.relu(Z)

        logits = self.output_layer(out)

        return logits
