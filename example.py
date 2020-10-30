import os
import argparse
import pandas as pd
import pprint
import torch
import torch
import tqdm
import torchvision
from torch.utils.data import DataLoader
from torch import nn


from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_jobs as hj

EXP_GROUPS = {'mnist_learning_rates':
              [
                  {'lr': 1e-3, 'model': 'mlp', 'dataset': 'mnist'},
                  {'lr': 1e-4, 'model': 'mlp', 'dataset': 'mnist'}
              ]
              }

# --- Main Single Experiment Function
def trainval(exp_dict, savedir_base, reset=False):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)

    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    # Dataset
    # -----------

    # train loader
    train_loader = get_loader(dataset_name=exp_dict['dataset'], datadir=savedir_base,
                                       split='train')

    # val loader
    val_loader = get_loader(dataset_name=exp_dict['dataset'], datadir=savedir_base,
                                     split='val')

    # Model
    # -----------
    model = get_model(model_name=exp_dict['model'])

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, 'model.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')

    if os.path.exists(score_list_path):
        # resume experiment
        model.set_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ------------
    print('Starting experiment at epoch %d' % (s_epoch))

    for e in range(s_epoch, 10):
        score_dict = {}

        # Train the model
        train_dict = model.train_on_loader(train_loader)

        # Validate the model
        val_dict = model.val_on_loader(val_loader)

        # Get metrics
        score_dict['train_loss'] = train_dict['train_loss']
        score_dict['val_acc'] = val_dict['val_acc']
        score_dict['epoch'] = e

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print('Checkpoint Saved: %s' % savedir)

    print('experiment completed')

# --- Dataloader Getter Function


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

# --- Model Getter Function


def get_model(model_name):
    if model_name == 'mlp':
        return MLP()


class MLP(nn.Module):
    def __init__(self, input_size=784, n_classes=10):
        """Constructor."""
        super().__init__()

        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, 256)])
        self.output_layer = nn.Linear(256, n_classes)

        self.opt = torch.optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, x):
        """Forward pass of one batch."""
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = torch.nn.functional.relu(Z)
        logits = self.output_layer(out)

        return logits

    def get_state_dict(self):
        return {'model': self.state_dict(),
                'opt': self.opt.state_dict()}

    def load_state_dict(self, state_dict):
        self.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['opt'])

    def train_on_loader(self, train_loader):
        """Train for one epoch."""
        self.train()
        loss_sum = 0.

        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        for i, batch in enumerate(train_loader):
            loss_sum += float(self.train_on_batch(batch))

            pbar.set_description("Training - loss: %.4f " %
                                 (loss_sum / (i + 1)))
            pbar.update(1)

        pbar.close()
        loss = loss_sum / n_batches

        return {"train_loss": loss}

    @torch.no_grad()
    def val_on_loader(self, val_loader):
        """Validate the model."""
        self.eval()
        se = 0.
        n_samples = 0

        n_batches = len(val_loader)
        pbar = tqdm.tqdm(desc="Validating", total=n_batches, leave=False)

        for i, batch in enumerate(val_loader):
            gt_labels = batch[1]
            pred_labels = self.predict_on_batch(batch)

            se += float((pred_labels.cpu() == gt_labels).sum())
            n_samples += gt_labels.shape[0]

            pbar.set_description("Validating -  %.4f acc" % (se / n_samples))
            pbar.update(1)

        pbar.close()

        acc = se / n_samples

        return {"val_acc": acc}

    def train_on_batch(self, batch):
        """Train for one batch."""
        images, labels = batch
        images, labels = images, labels

        self.opt.zero_grad()
        probs = torch.nn.functional.log_softmax(self(images), dim=1)
        loss = torch.nn.functional.nll_loss(probs, labels, reduction="mean")
        loss.backward()

        self.opt.step()

        return loss.item()

    def predict_on_batch(self, batch, **options):
        """Predict for one batch."""
        images, labels = batch
        images = images
        probs = torch.nn.functional.log_softmax(self(images), dim=1)

        return probs.argmax(dim=1)


JOB_CONFIG = {'image': 'registry.console.elementai.com/%s/ssh' % 
                      os.environ['EAI_ACCOUNT_ID'] ,
      'data': ['eai.colab.public:/mnt/public'],
      'restartable':True,
      'resources': {
          'cpu': 4,
          'mem': 8,
          'gpu': 1
      },
      'interactive': False,
      'bid':9999,
      }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-j', '--run_jobs',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)

    args = parser.parse_args()

    # select exp group
    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list += EXP_GROUPS[exp_group_name]
    
    if not args.run_jobs:
      # run experiments
      for exp_dict in exp_list:
          # do trainval
          trainval(exp_dict=exp_dict,
                   savedir_base=args.savedir_base,
                   reset=args.reset)
        
    else:
        # launch jobs
        from haven import haven_jobs as hjb
        import job_configs as jc
        
        jm = hjb.JobManager(exp_list=exp_list, 
                    savedir_base=args.savedir_base, 
                    account_id=os.environ['EAI_ACCOUNT_ID'],
                    workdir=os.path.dirname(os.path.realpath(__file__)),
                    job_config=JOB_CONFIG,
                    )

        command = ('python trainval.py -ei <exp_id> -sb %s -d %s -nw 2' %  
                  (args.savedir_base, args.datadir))
        print(command)
        jm.launch_menu(command=command)
