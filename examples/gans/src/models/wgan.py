import torch
import torchvision.utils as vutils
import torch.autograd as autograd
import os
import src.utils as ut
from src import models
import tqdm


class WGan(torch.nn.Module):
    def __init__(self, netG, netD, optG, optD, device,
                 image_size, batch_size, lambda_gp, d_iterations):
        super().__init__()
        self.device = device
        self.netG = netG
        self.optG = optG
        self.netD = netD
        self.optD = optD
        self.batch_size = batch_size
        self.image_size = image_size
        self.fixed_noise = torch.randn(self.batch_size, self.netG.nz, 1, 1,
                                       device=self.device)
        self.iteration = 0
        self.lambda_gp = lambda_gp
        self.d_iterations = d_iterations
        self.errG = torch.Tensor([0]).to(self.device)

    def get_state_dict(self):
        state_dict = {'optG': self.optG.state_dict(),
                      'netG': self.netG.state_dict(),
                      'optD': self.optD.state_dict(),
                      'netD': self.netD.state_dict()}
        return state_dict

    def load_state_dict(self, state_dict):
        self.optG.load_state_dict(state_dict['optG'])
        self.netG.load_state_dict(state_dict['netG'])
        self.optD.load_state_dict(state_dict['optD'])
        self.netD.load_state_dict(state_dict['netD'])

    def train_on_batch(self, batch):
        self.iteration += 1

        ############################
        # (1) Train Discriminator
        ###########################
        self.optD.zero_grad()

        real_images = batch[0].to(self.device)
        batch_size = real_images.size(0)
        real_output = self.netD(real_images)

        noise = torch.randn(batch_size, self.netG.nz, 1, 1, device=self.device)
        fake_images = self.netG(noise)
        fake_output = self.netD(fake_images)

        # Gradient penalty
        gp = self._compute_gradient_penalty(real_images.detach(),
                                            fake_images.detach())

        # Adversarial loss
        errD = self._compute_d_loss(real_output, fake_output, gp,
                                    self.lambda_gp)
        # TODO: clean this up so you don't compute it twice
        emd = torch.mean(real_output) - torch.mean(fake_output)
        
        errD.backward()
        self.optD.step()

        ############################
        # (2) Train Generator every d_iterations
        ###########################
        self.optG.zero_grad()

        if self.iteration % self.d_iterations == 0:
            # Generate a batch of images
            fake_images = self.netG(noise)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_output = self.netD(fake_images)
            self.errG = self._compute_g_loss(fake_output)

            self.errG.backward()
            self.optG.step()

        return {
            'losses': {
                'loss_D': errD.item(),
                'loss_G': self.errG.item(),
                'wasserstein_loss_emd': emd.item()
            }
        }

    @torch.no_grad()
    def eval_on_batch(self, batch, savedir, epoch, summary_writer):
        self.eval()
        images_path = os.path.join(savedir, 'images')
        os.makedirs(images_path, exist_ok=True)
        fake = self.netG(self.fixed_noise)
        fake_save_path = os.path.join(images_path,
                                      'fake_samples_epoch_%04d.png' %
                                      epoch)
        vutils.save_image(fake.detach(), fake_save_path,
                          normalize=True)


    def _compute_d_loss(self, real_output, fake_output, gp, lambda_gp):
        return -torch.mean(real_output) + torch.mean(fake_output) + \
            lambda_gp * gp

    def _compute_g_loss(self, fake_output):
        return -torch.mean(fake_output)

    def _compute_gradient_penalty(self, real_images, fake_images):
        """Calculates the gradient penalty loss for WGAN GP"""
        batch_size = real_images.size(0)
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images))\
                       .requires_grad_(True)
        d_interpolates = self.netD(interpolates)
        fake = torch.ones(batch_size).to(self.device).requires_grad_(False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_on_loader(self, train_loader):
        self.train()

        n_batches = len(train_loader)
        pbar = tqdm.tqdm(total=n_batches, miniters=max(n_batches/100, 1), ncols=180)
        agg_results = {}
        for i, batch in enumerate(train_loader):
            results = self.train_on_batch(batch)
            for loss_name, loss_value in results['losses'].items():
                if loss_name in agg_results:
                    agg_results[loss_name] += loss_value 
                else:
                    agg_results[loss_name] = loss_value

            # mesg = 'Epoch {}/{}:\t'.format(epoch, num_epochs)
            mesg = ''
            for name, loss in results['losses'].items():
                mesg += '{}: {:.6f}  '.format(name, loss)
            if 'others' in results:
                for name, info in results['others'].items():
                    mesg += '{}: {:.6f}  '.format(name, info)
            pbar.update(1)
            pbar.set_description(mesg, refresh=False)
        pbar.close()

        avg_results = {}
        for agg_loss_name, agg_loss_value in agg_results.items():
            avg_results[agg_loss_name] = agg_loss_value  / n_batches

        return avg_results
    
    def val_on_loader(self, test_loader, savedir, epoch):
        batch = iter(test_loader).next()
        self.val_on_batch(batch, savedir, epoch)