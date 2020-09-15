from .wgangp_resnet import Generator as StdGenerator, Discriminator as StdDiscriminator


def get_base(base_name, num_channels, num_train_classes, num_eval_classes,
              exp_dict):
    if base_name == 'wgan_resnet':
        return StdGenerator(exp_dict['nz'], exp_dict['ngf'],
                         num_channels, exp_dict['image_size'],
                         norm=exp_dict['gen_norm']),\
               StdDiscriminator(exp_dict['ndf'], num_channels,
                             exp_dict['image_size'],
                             norm=exp_dict['disc_norm'])
    