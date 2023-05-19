import argparse
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'T', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'F', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_directories(args, model_class=None):
    full_model_name = args.model_name
    if hasattr(args, 'attention_type'):
        full_model_name += '_' + args.attention_type
    args.model_signature = full_model_name
    data_time = str(datetime.datetime.now())[0:19].replace(' ', '_')

    args.model_signature += '_' + data_time.replace(':', '_')
    if hasattr(args, 'exemplar'):
        if args.exemplar:
            args.model_signature += '_' + 'exVAE'
    if hasattr(args, 'z_size'):
        args.model_signature += '_' + 'z{0}'.format(args.z_size)
    if hasattr(args, 'hidden_size') and hasattr(args, 'k'):
        args.model_signature += '_' + 'hid{0}_k{1}'.format(args.hidden_size, args.k)
    if hasattr(args, 'hidden_prior') and hasattr(args, 'num_layer_prior'):
        args.model_signature += '_' + 'hid_p{0}_layer_p{1}'.format(args.hidden_prior, args.num_layer_prior)
    if hasattr(args, 'time_step'):
        args.model_signature += '_' + 'T{0}'.format(args.time_step)
    if hasattr(args, 'read_size'):
        args.model_signature += '_' + 'rs{0}'.format(args.read_size[-1])
    if hasattr(args, 'write_size'):
        args.model_signature += '_' + 'rs{0}'.format(args.write_size[-1])
    if hasattr(args, 'lstm_size'):
        args.model_signature += '_' + 'lstm{0}'.format(args.lstm_size)
    if hasattr(args, 'beta'):
        args.model_signature += '_' + 'beta{0}'.format(args.beta)
    if hasattr(args, 'order'):
         args.model_signature += '_' + 'order{0}'.format(args.order)
    if hasattr(args, 'size_factor'):
        args.model_signature += '_' + '{0}sf'.format(args.size_factor)
    if args.model_name == 'vae_stn_var':
        if hasattr(args, 'attention_ratio'):
            args.model_signature += '_' + 'attn_ratio{0}'.format(args.attention_ratio)
    if hasattr(args, 'strength'):
        args.model_signature += '_' + 'str={0}'.format(args.strength)
    if hasattr(args, 'annealing_time'):
        if args.annealing_time is not None:
            args.model_signature += '_' + 'BetaAnneal{}'.format(args.annealing_time)
    if hasattr(args, 'shuffle_exemplar'):
        if args.shuffle_exemplar:
            args.model_signature += '_se'
    if hasattr(args, 'rate_scheduler'):
        if args.rate_scheduler:
            args.model_signature += '_rc'
    if hasattr(args, 'embedding_size'):
        args.model_signature += '_' + 'emb_sz={0}'.format(args.embedding_size)
    if model_class == 'pixel_cnn' and hasattr(args, 'latent_size'):
        args.model_signature += '_latent_size=[{0},{1}]'.format(args.latent_size[0], args.latent_size[1])
    if args.tag != '':
        args.model_signature += '_' + args.tag

    if model_class is None:
        snapshots_path = os.path.join(args.out_dir, args.dataset, full_model_name)
    else:
        snapshots_path = os.path.join(args.out_dir, args.dataset, model_class)

    args.snap_dir = snapshots_path + '/' + args.model_signature + '/'

    if args.model_name in ['ns', 'tns', 'hfsgm']:
        args.snap_dir = snapshots_path + '/' + args.model_signature + '_' + str(args.c_dim) + '_' + str(args.z_dim)  + '_' + str(args.hidden_dim)
        if args.model_name == 'tns':
            args.snap_dir += '_' + str(args.n_head)
        args.snap_dir += '/'

    if not args.debug:
        os.makedirs(snapshots_path, exist_ok=True)
        os.makedirs(args.snap_dir, exist_ok=True)
        args.fig_dir = args.snap_dir + 'fig/'
        os.makedirs(args.fig_dir, exist_ok=True)
    else:
        args.fig_dir = None
    return args


def plot_img(data, nrow=4, ncol=8, padding=2, normalize=True, saving_path=None, title=None, pad_value=0,
             figsize=(8, 8), dpi=100, scale_each=False, cmap=None, axs=None):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value, scale_each=scale_each)
    show(data_to_plot.detach().cpu(), saving_path=saving_path, title=title, figsize=figsize, dpi=dpi, cmap=cmap, axs=axs)


def show(img, title=None, saving_path=None, figsize=(8, 8), dpi=100, cmap=None, axs=None):
    npimg = img.numpy()
    if axs is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest', cmap=cmap)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        if saving_path is None :
            plt.show()
        else:
            plt.savefig(saving_path + '/' + title + '.png')
        plt.close()
    else:
        axs.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest', cmap=cmap)


def make_grid(data, nrow=4, ncol=8, padding=2, normalize=True, pad_value=0, scale_each=False):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value, scale_each=scale_each)
    return data_to_plot
