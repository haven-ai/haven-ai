from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F

# Residual network.
# WGAN-GP paper defines a residual block with up & downsampling.
# See the official implementation (given in the paper).
# I use architectures described in the official implementation,
# since I find it hard to deduce the blocks given here from the text alone.

# from https://github.com/ozanciga/gans-with-pytorch/blob/master/wgan-gp/models.py

####################
# HELPER FUNCTIONS #
####################

class Conv2d(nn.Module):
    def __init__(self, n_input, n_output, k_size, padding, stride=1, bias=True,
                 he_init=True):
        super().__init__()
        self.conv = nn.Conv2d(n_input, n_output, k_size, stride=stride,
                              padding=padding, bias=bias)
        
        # Weight init
        if he_init:
            self.apply(_he_weights_init)
        else:
            self.apply(_xavier_weights_init)

    def forward(self, x):
        return self.conv(x)

class ConvTranspose2d(nn.Module):
    def __init__(self, n_input, n_output, k_size, padding, stride=1, bias=True,
                 he_init=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_input, n_output, k_size,
                                       stride=stride, padding=padding,
                                       bias=bias)
        
        # Weight init
        if he_init:
            self.apply(_he_weights_init)
        else:
            self.apply(_xavier_weights_init)

    def forward(self, x):
        return self.conv(x)

class Linear(nn.Module):
    def __init__(self, n_input, n_output, bias=True, he_init=True):
        super().__init__()
        self.linear = nn.Linear(n_input, n_output, bias=bias)
        
        # Weight init
        if he_init:
            self.apply(_he_weights_init)
        else:
            self.apply(_xavier_weights_init)

    def forward(self, x):
        return self.linear(x)

class MeanPoolConv(nn.Module):
    def __init__(self, n_input, n_output, k_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        conv1 = Conv2d(n_input, n_output, k_size, stride=1,
                       padding=(k_size-1)//2, bias=True, he_init=he_init)
        self.model = nn.Sequential(conv1)

    def forward(self, x):
        out = (x[:,:,::2,::2] + x[:,:,1::2,::2] +
               x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4.0
        out = self.model(out)
        return out


class ConvMeanPool(nn.Module):
    def __init__(self, n_input, n_output, k_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        conv1 = Conv2d(n_input, n_output, k_size, stride=1,
                       padding=(k_size-1)//2, bias=True, he_init=he_init)
        self.model = nn.Sequential(conv1)

    def forward(self, x):
        out = self.model(x)
        out = (out[:,:,::2,::2] + out[:,:,1::2,::2] +
               out[:,:,::2,1::2] + out[:,:,1::2,1::2]) / 4.0
        return out


class UpsampleConv(nn.Module):
    def __init__(self, n_input, n_output, k_size, he_init=True):
        super(UpsampleConv, self).__init__()

        self.model = nn.Sequential(
            nn.PixelShuffle(2),
            Conv2d(n_input, n_output, k_size, stride=1,
                   padding=(k_size-1)//2, bias=True, he_init=he_init)
        )

    def forward(self, x):
        x = x.repeat((1, 4, 1, 1)) # Weird concat of WGAN-GPs upsampling process.
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size, n_cond=None,
                 resample='up', norm='bn', spatial_dim=None):
        super(ResidualBlock, self).__init__()

        self.resample = resample

        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = Conv2d(n_output, n_output, k_size,
                                padding=(k_size-1)//2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
            self.ln1_dims = [n_input, spatial_dim, spatial_dim]
            self.ln2_dims = [n_output, spatial_dim*2, spatial_dim*2]
        elif resample == 'down':
            self.conv1 = Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
            self.out_dim = n_input
            self.ln1_dims = [n_input, spatial_dim, spatial_dim]
            self.ln2_dims = [n_input, spatial_dim, spatial_dim]
        else:
            self.conv1 = Conv2d(n_input, n_output, k_size,
                                padding=(k_size-1)//2)
            self.conv2 = Conv2d(n_output, n_output, k_size,
                                padding=(k_size-1)//2)
            if n_input == n_output:
                self.conv_shortcut = nn.Identity()
            else:
                self.conv_shortcut = Conv2d(n_input, n_output, 1, padding=0,
                                            he_init=False)
            self.out_dim = n_output
            self.ln1_dims = [n_input, spatial_dim, spatial_dim]
            self.ln2_dims = [n_output, spatial_dim, spatial_dim]

        if norm == 'bn':
            self.norm1 = nn.BatchNorm2d(n_input)
            self.norm2 = nn.BatchNorm2d(self.out_dim)
        elif norm == 'ln':
            self.norm1 = nn.LayerNorm(self.ln1_dims)
            self.norm2 = nn.LayerNorm(self.ln2_dims)
        elif norm == 'in':
            self.norm1 = nn.InstanceNorm2d(n_input)
            self.norm2 = nn.InstanceNorm2d(self.out_dim)
        elif norm == 'condin':
            self.norm1 = CondInstanceNorm(n_input, n_cond*2, n_cond)
            self.norm2 = CondInstanceNorm(self.out_dim, n_cond*2, n_cond)
        elif norm == 'condbn':
            self.norm1 = CondBatchNorm(n_input, n_cond*2, n_cond)
            self.norm2 = CondBatchNorm(self.out_dim, n_cond*2, n_cond)
        elif norm is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x, cond=None):
        residual = self.conv_shortcut(x)
        if cond is not None:
            out = self.norm1(x, cond)
            out = F.relu(out)
            out = self.conv1(out)
            out = self.norm2(out, cond)
            out = F.relu(out)
            out = self.conv2(out)
        else:
            out = self.norm1(x)
            out = F.relu(out)
            out = self.conv1(out)
            out = self.norm2(out)
            out = F.relu(out)
            out = self.conv2(out)
        return residual + out

class DiscBlock1(nn.Module):
    def __init__(self, num_channels, ndf):
        super(DiscBlock1, self).__init__()

        self.conv1 = Conv2d(num_channels, ndf, 3, padding=1)
        self.conv2 = ConvMeanPool(ndf, ndf, 3)
        self.conv_shortcut = MeanPoolConv(num_channels, ndf, 1, he_init=False)

        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.conv2
        )

    def forward(self, x):
        return self.conv_shortcut(x) + self.model(x)

def _he_weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1 and not hasattr(model, 'conv'):
        nn.init.kaiming_normal_(model.weight.data, mode='fan_in',
                                nonlinearity='relu')
        if hasattr(model, 'bias'):
            nn.init.zeros_(model.bias.data)
    elif classname.find('Linear') != -1 and not hasattr(model, 'linear'):
        nn.init.kaiming_normal_(model.weight.data, mode='fan_in',
                                nonlinearity='relu')
        if hasattr(model, 'bias'):
            nn.init.zeros_(model.bias.data)

def _xavier_weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1 and not hasattr(model, 'conv'):
        nn.init.xavier_normal_(model.weight.data)
        if hasattr(model, 'bias'):
            nn.init.zeros_(model.bias.data)
    elif classname.find('Linear') != -1 and not hasattr(model, 'linear'):
        nn.init.xavier_normal_(model.weight.data)
        if hasattr(model, 'bias'):
            nn.init.zeros_(model.bias.data)


class CondInstanceNorm(nn.InstanceNorm2d):
    """Conditional instance norm."""
    def __init__(self, n_input, n_in, n_cond, eps=1e-5, momentum=0.1):
        """Constructor.
        - `n_input`: number of input features
        - `n_in`: number of features for shift and scale convs
        - `n_cond`: number of cond features
        """
        super().__init__(n_input, eps, momentum, affine=False)
        self.eps = eps
        self.shift_conv = nn.Sequential(
            Conv2d(n_cond, n_in, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            Conv2d(n_in, n_input, 1, padding=0, bias=True),
        )
        self.scale_conv = nn.Sequential(
            Conv2d(n_cond, n_in, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            Conv2d(n_in, n_input, 1, padding=0, bias=True),
        )

    def forward(self, input, cond):
        """Forward method."""
        shift = self.shift_conv.forward(cond)
        scale = self.scale_conv.forward(cond)
        norm_features = super().forward(input)
        output = norm_features * scale + shift
        return output


class CondBatchNorm(nn.BatchNorm2d):
    """Conditional batch norm."""
    def __init__(self, n_input, n_bn, n_cond, eps=1e-5, momentum=0.1):
        """Constructor.
        - `n_input`: number of input features
        - `n_bn`: number of features for shift and scale convs
        - `n_cond`: number of cond features
        """
        super().__init__(n_input, eps, momentum, affine=False)
        self.eps = eps
        self.shift_conv = nn.Sequential(
            Conv2d(n_cond, n_bn, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            Conv2d(n_bn, n_input, 1, padding=0, bias=True),
        )
        self.scale_conv = nn.Sequential(
            Conv2d(n_cond, n_bn, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            Conv2d(n_bn, n_input, 1, padding=0, bias=True),
        )

    def forward(self, input, cond):
        """Forward method."""
        shift = self.shift_conv.forward(cond)
        scale = self.scale_conv.forward(cond)
        norm_features = super().forward(input)
        output = norm_features * scale + shift
        return output

###############################
# GENERATOR AND DISCRIMINATOR #
###############################

class Generator(nn.Module):
    def __init__(self, nz, ngf, num_channels, image_size, nef=None,
                 num_train_classes=None,
                 num_eval_classes=None,
                 conditioning_input=None,
                 conditioning_type=None,
                 norm='bn'):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.num_channels = num_channels
        self.image_size = image_size
        self.nef = nef
        self.num_train_classes = num_train_classes
        self.num_eval_classes = num_eval_classes
        self.conditioning_input = conditioning_input
        self.conditioning_type = conditioning_type
        self.norm = norm

        # Initial layers and conditioning layers
        self.input_layers, self.cond_layers = \
            self.get_initial_layers(nz, ngf, nef, num_channels,
                                    num_train_classes)

        if self.conditioning_type in ['condin', 'condbn']:
            self.norm = self.conditioning_type

        # Residual blocks
        # Input is ngf x 4 x 4
        # Output is ngf x image_size x image_size
        self.res_layers = self.get_residual_layers(ngf, nef, image_size,
                                                   self.norm)

        # Final layers of generator
        # Input is ngf x image_size x image_size
        # Output is num_channels x image_size x image_size
        self.final_layers = self.get_final_layers(ngf, num_channels)

    def forward(self, input, cond=None):
        # Initial conv
        # Input is image B x nz x 1 x 1
        # Output is B x ngf//out_divider x 4 x 4
        input = self.input_layers(input)

        if self.conditioning_type in ['condin', 'condbn']:
            # Image/label encoding
            # Output is B x nef x 1 x 1
            cond = self.cond_layers(cond)

            # Residual blocks
            # Input is B x ngf x 4 x 4
            # Output is B x ngf x image_size x image_size
            for layer in self.res_layers.values():
                input = layer(input, cond=cond)

        elif self.conditioning_type in ['concat', 'product']:
            # Image/label encoding
            # Input is B x num_channels x image_size x image_size if image
            # Input is B x num_classes x 1 x 1 if label
            # Output is B x ngf//out_divider x 4 x 4
            cond = self.cond_layers(cond)

            # Combining encodings
            # Input is B x ngf//out_divider x 4 x 4
            # Output is B x ngf x 4 x 4
            if self.conditioning_type == 'concat':
                input = torch.cat([input, cond], 1)
            else:
                input = input * cond

            # Residual blocks
            # Input is B x ngf x 4 x 4
            # Output is B x ngf x image_size x image_size
            for layer in self.res_layers.values():
                input = layer(input)
        else:
            # Residual blocks
            # Input is B x ngf x 4 x 4
            # Output is B x ngf x image_size x image_size
            for layer in self.res_layers.values():
                input = layer(input)

        # Final layers
        # Input is B x ngf x image_size x image_size
        # Output is B x num_channels x image_size x image_size
        out = self.final_layers(input)

        return out

    def get_initial_layers(self, nz, ngf, nef, num_channels, num_train_classes):
        input_layers = OrderedDict([])
        cond_layers = OrderedDict([])

        if self.conditioning_type == 'concat':
            out_divider = 2
        else:
            out_divider = 1

        # Input layers
        # Input is Z, output is ngf//out_divider x 4 x 4
        input_layers['deconv1_input'] = \
            ConvTranspose2d(nz, ngf // out_divider, 4, padding=0,
                            he_init=False)

        # Conditioning layers
        if self.conditioning_input == 'image':
            # Encoding image to a latent vector encoding
            # Input is num_channels x image_size x image_size
            # Output is nef x 1 x 1
            cond_layers['encoder'] = \
                Encoder(num_channels, self.image_size, ngf, nef)
            
            # Input is a latent vector encoding nef x 1 x 1
            # Output is ngf//out_divider x 4 x 4
            cond_layers['deconv1_cond'] = \
                ConvTranspose2d(nef, ngf // out_divider, 4, padding=0,
                                he_init=False)
        elif self.conditioning_input == 'label':
            # Input is a one hot label vector num_classes x 1 x 1
            # Output is ngf//out_divider x 4 x 4
            cond_layers['deconv1_cond'] = \
                ConvTranspose2d(num_train_classes, ngf // out_divider,
                                4, padding=0, he_init=False)

        if self.conditioning_type in ['condin', 'condbn']:
            del cond_layers['deconv1_cond']
            if self.conditioning_input == 'label':
                cond_layers['identity'] = nn.Identity()

        return nn.Sequential(input_layers), nn.Sequential(cond_layers)

    def get_residual_layers(self, ngf, nef, image_size, norm):
        res_layers = OrderedDict([])

        i = 1
        dim = 4  # Input is ngf x 4 x 4
        while dim <= image_size/2:
            # ngf x dim*2 x dim*2
            res_layers['res' + str(i)] = \
                ResidualBlock(ngf, ngf, 3, resample='up', norm=norm,
                              spatial_dim=dim, n_cond=nef)
            dim *= 2
            i += 1

        return nn.ModuleDict(res_layers)

    def get_final_layers(self, ngf, num_channels):
        final_layers = OrderedDict([])

        # Input is ngf x image_size x image_size
        # Output is 3 x image_size x image_size
        final_layers['norm'] = nn.BatchNorm2d(ngf)
        final_layers['relu'] = nn.ReLU(inplace=True)
        final_layers['conv'] = Conv2d(ngf, num_channels, 3, padding=1,
                                      he_init=False)
        final_layers['tanh'] = nn.Tanh()

        return nn.Sequential(final_layers)


class Discriminator(nn.Module):
    def __init__(self, ndf, num_channels, image_size,
                 nef=None,
                 num_train_classes=None,
                 num_eval_classes=None,
                 conditioning_input=None,
                 conditioning_type=None,
                 norm=None):
        super().__init__()
        self.ndf = ndf
        self.num_channels = num_channels
        self.image_size = image_size
        self.nef = nef
        self.num_train_classes = num_train_classes
        self.num_eval_classes = num_eval_classes
        self.conditioning_input = conditioning_input
        self.conditioning_type = conditioning_type
        self.norm = norm

        # Initial layers and conditioning layers
        self.input_layers, self.cond_layers = \
            self.get_initial_layers(num_channels, ndf, num_train_classes, nef)

        if self.conditioning_type in ['condin', 'condbn']:
            self.norm = self.conditioning_type

        # Residual blocks
        # Input is ndf x image_size/2 x image_size/2
        # Output is ndf x 8 x 8
        self.res_layers = self.get_residual_layers(ndf, nef, image_size/2,
                                                   self.norm)

        # Final layers
        # Input is ndf x 8 x 8
        # Output is 1 x 1 x 1
        self.final_layers = self.get_final_layers(ndf)

    def forward(self, input, cond=None):
        # Initial conv
        # Input is image B x num_channels x image_size x image_size
        # Output is B x ndf//out_divider x image_size/2 x image_size/2
        input = self.input_layers(input)

        if self.conditioning_type in ['condin', 'condbn']:
            # Image/label encoding
            # Output is B x nef x 1 x 1
            cond = self.cond_layers(cond)

            # Residual blocks
            # Input is B x ndf x image_size/2 x image_size/2
            # Output is B x ndf x 8 x 8
            for layer in self.res_layers.values():
                input = layer(input, cond=cond)

        elif self.conditioning_type in ['concat', 'product']:
            # Image/label encoding
            # Input is B x num_channels x image_size x image_size if image
            # Input is B x num_classes x image_size x image_size if label
            # Output is B x ndf//out_divider x image_size/2 x image_size/2
            cond = self.cond_layers(cond)

            # Combining encodings
            # Input is B x ndf//out_divider x image_size/2 x image_size/2
            # Output is B x ndf x image_size/2 x image_size/2
            if self.conditioning_type == 'concat':
                input = torch.cat([input, cond], 1)
            else:
                input = input * cond

            # Residual blocks
            # Input is B x ndf x image_size/2 x image_size/2
            # Output is B x ndf x 8 x 8
            for layer in self.res_layers.values():
                input = layer(input)
        else:
            # Residual blocks
            # Input is B x ndf x image_size/2 x image_size/2
            # Output is B x ndf x 8 x 8
            for layer in self.res_layers.values():
                input = layer(input)

        # Final layers
        # Input is B x ndf x 8 x 8
        # Output is B x 1 x 1 x 1
        out = self.final_layers(input)

        return out.squeeze()  # B

    def get_initial_layers(self, num_channels, ndf, num_train_classes, nef):
        input_layers = OrderedDict([])
        cond_layers = OrderedDict([])

        if self.conditioning_type == 'concat':
            out_divider = 2
        else:
            out_divider = 1

        # Input layers
        # Input is image num_channels x image_size x image_size
        # Output is ndf//out_divider x image_size/2 x image_size/2
        input_layers['DiscBlock1_input'] = \
            DiscBlock1(num_channels, ndf // out_divider)

        # Conditioning layers
        if self.conditioning_type in ['product', 'concat']:
            if self.conditioning_input == 'image':
                n_cond = num_channels
            elif self.conditioning_input == 'label':
                n_cond = num_train_classes
            # Input is image n_cond x image_size x image_size
            # If label, then it is replicated into an image in the wrapper.
            # Output is ndf//out_divider x image_size/2 x image_size/2
            cond_layers['DiscBlock1_cond'] = \
                DiscBlock1(n_cond, ndf // out_divider)

        elif self.conditioning_type in ['condin', 'condbn']:
            if self.conditioning_input == 'image':
                # Encoding image to a latent vector encoding
                # Input is num_channels x image_size x image_size
                # Output is nef x 1 x 1
                cond_layers['encoder'] = \
                    Encoder(num_channels, self.image_size, ndf, nef)
            elif self.conditioning_input == 'label':
                # Pass label through
                cond_layers['identity'] = nn.Identity()

        return nn.Sequential(input_layers), nn.Sequential(cond_layers)

    def get_residual_layers(self, ndf, nef, image_size, norm):
        res_layers = OrderedDict([])

        i = 1
        while image_size >= 16:
            # ndf x image_size/2 x image_size/2
            res_layers['res' + str(i)] = \
                ResidualBlock(ndf, ndf, 3, resample='down', norm=norm,
                              spatial_dim=int(image_size), n_cond=nef)
            image_size /= 2
            i += 1

        # Input is ndf x 8 x 8
        # Output is ndf x 8 x 8
        res_layers['res' + str(i)] = \
            ResidualBlock(ndf, ndf, 3, resample=None, norm=norm,
                          spatial_dim=int(image_size), n_cond=nef)
        res_layers['res' + str(i+1)] = \
            ResidualBlock(ndf, ndf, 3, resample=None, norm=norm,
                          spatial_dim=int(image_size), n_cond=nef)

        return nn.ModuleDict(res_layers)

    def get_final_layers(self, ndf):
        final_layers = OrderedDict([])

        # Input is ndf x 8 x 8
        # Output is ndf x 1 x 1
        final_layers['relu'] = nn.ReLU(inplace=True)
        final_layers['pool'] = nn.AvgPool2d(8, padding=0)

        # Input is ndf x 1 x 1
        # Output is 1 x 1 x 1
        final_layers['linear'] = Conv2d(ndf, 1, 1, padding=0, he_init=False)

        return nn.Sequential(final_layers)


class Encoder(torch.nn.Module):
    def __init__(self, num_channels, image_size, nef, nz):
        super().__init__()
        self.nz = nz
        self.nef = nef
        self.num_channels = num_channels
        self.image_size = image_size

        # Input is num_channels x image_size x image_size
        # Output is nz x 1 x 1
        self.disc = Discriminator(nef, num_channels, image_size)
        self.disc.final_layers = self.get_final_layers(nef, nz)

    def get_final_layers(self, nef, nz):
        final_layers = OrderedDict([])

        # Input is nef x 8 x 8
        # Output is nef x 1 x 1
        final_layers['relu'] = nn.ReLU(inplace=True)
        final_layers['pool'] = nn.AvgPool2d(8, padding=0)

        # Input is nef x 1 x 1
        # Output is nz x 1 x 1
        final_layers['linear'] = Conv2d(nef, nz, 1, padding=0, he_init=False)

        return nn.Sequential(final_layers)

    def forward(self, input):
        # Input is B x num_channels x image_size x image_size
        # Output is B x nz x 1 x 1
        out = self.disc(input).view(input.size(0), self.nz, 1, 1)
        
        return out