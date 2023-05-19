import torch
from utils.monitoring import str2bool, make_directories, make_grid, plot_img
from data_utils.data_loader import load_dataset_exemplar
from diffusion.model import Unet_V2_Omniglot
from diffusion.ddpm import DDPM
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
import wandb
import argparse
import numpy as np
import random
from data_utils.custom_transform import Binarize_batch, Scale_0_1_batch


scale_01, binarize = Scale_0_1_batch(), Binarize_batch(binary_threshold=0.5)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='omniglot', choices=['omniglot', 'quickdraw_clust'],
                    metavar='DATASET', help='Dataset choice.')
parser.add_argument('--download_data', type=eval, default=False, choices=[True, False])
parser.add_argument('--dataset_root', type=str, default="/media/data_cifs_lrs/projects/prj_control/data")
parser.add_argument('-od', '--out_dir', type=str, default='/media/data_cifs/projects/prj_zero_gene/exp/',
                    metavar='OUT_DIR', help='output directory for model snapshots etc.')
parser.add_argument('--input_type', type=str, default='binary',
                    choices=['binary'], help='type of the input')
parser.add_argument('--debug', default=False, action='store_true', help='debugging flag (do not save the network)')
parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH_SIZE',
                    help='input batch size for training')
parser.add_argument('--timestep', type=int, default=600, metavar='TIMESTEP',
                    help='number of time step of the diffusion model')
parser.add_argument('--n_feat', type=int, default=48, metavar='NFEAT',
                    help='number of channel of the first layer of the UNET')
parser.add_argument('--epoch', type=int, default=300, metavar='EPOCH',
                    help='number of training epoch')
parser.add_argument('--embedding_size', type=int, default=128, metavar='EMB_SIZE',
                    help='size of the embedding of the diffusion process')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                    help='learning rate of the optimizer')
parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 48, 48],
                    help='shape of the input [channel, height, width]')
parser.add_argument("--unet_dim", nargs='+', type=int, default=[1, 2, 4],
                    help='multiplier of the UNET')
parser.add_argument("--augment", type=str2bool, nargs='?', const=True, default=False, help="data augmentation")
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')
parser.add_argument('--model_name', type=str, default='ddpm', choices=['ddpm', 'cfgdm'], help="type of the diffusion model ['ddpm', 'cfgdm']")
parser.add_argument('--tag', type=str, default='', help='tag of the experiment')
parser.add_argument('--preload', default=True, action='store_true', help='preload the dataset')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument('--embedding_model', type=str, default='', help='name of the embedding model')
parser.add_argument('--clip_norm', type=float, default=0.5, help='value of the clipping norm')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument("--augment_class", type=str2bool, nargs='?', const=True, default=False, help="augment the number of class with transformation")
parser.add_argument('--sample_per_class', type=int, default=500, metavar='NB_SAMPLE', help='number of sample per class')
parser.add_argument('--generate_img', default=False, action='store_true', help='generate image at the end of the training')
parser.add_argument('--seed', type=int, default=None, metavar='SEED', help='random seed (None is no seed)')

args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

wb_name = args.model_name
if args.dataset == 'omniglot':
    wb_name += '_omniglot'
elif args.dataset == 'quickdraw_clust':
    wb_name += '_qd'

if args.device == 'meso':
    wb_name += '_osc'

if args.device == 'meso':
    args.device = torch.cuda.current_device()

if args.model_name =='ddpm':
    args.drop_out = 0.0

args.input_shape = tuple(args.input_shape)
args.unet_dim = tuple(args.unet_dim)

args = make_directories(args, model_class='diffusion')

hyperparameters = {"device": args.device,
                   "image_size": args.input_shape[-1],
                   "channels": 1,
                   "batch_size": args.batch_size,
                   "learning_rate": args.learning_rate,
                   "timesteps": args.timestep,
                   "loss_type": 'huber',
                   "epochs": args.epoch,
                   "unet_dim": args.unet_dim,
                   "embedding_size": args.embedding_size,
                   'n_feat': args.n_feat}

nb_samples = 100
if not args.debug:
    #wandb.init(project=wb_name, config=hyperparameters, entity='vb')
    wandb.init(project=wb_name, config=hyperparameters)
    wandb.run.name = args.model_signature
    wandb.run.save()

transform = transforms.Compose([
            transforms.ToTensor()
])


kwargs = {'preload': args.preload}
train_loader, test_loader, args = load_dataset_exemplar(args,
                                                        shape=[1, hyperparameters["image_size"],
                                                               hyperparameters["image_size"]], **kwargs, drop_last=True)
train_exemplar = next(iter(train_loader))[1][0:10]
test_exemplar = next(iter(test_loader))[1][0:10]
one_batch_test_image = next(iter(test_loader))[0].to(hyperparameters["device"])
embedding_model = 'stack'
one_batch_test_exemplar = next(iter(test_loader))[1].to(hyperparameters["device"])
model = Unet_V2_Omniglot(in_channels=hyperparameters['channels'], n_feat=args.n_feat, embedding_model=embedding_model, dim_mults=args.unet_dim).to(hyperparameters['device'])
diffusion_model = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=hyperparameters["timesteps"], device=hyperparameters["device"], drop_prob=args.drop_out).to(hyperparameters["device"])

param_to_optimize = [p for name, p in diffusion_model.named_parameters() if 'embedding' not in name]
print('# param : {0:,}'.format(sum(p.numel() for p in param_to_optimize)))

optimizer = Adam(param_to_optimize, lr=hyperparameters["learning_rate"])

if not args.debug:
    torch.save(args, args.snap_dir + 'param.config')

mean_loss_old = np.inf
for epoch in range(hyperparameters["epochs"]):
    pbar = tqdm(train_loader)
    mean_loss = 0
    diffusion_model.train()
    optimizer.param_groups[0]['lr'] = hyperparameters["learning_rate"] * (1 - epoch / hyperparameters["epochs"])
    for step, (image, exemplar, label) in enumerate(pbar):
        optimizer.zero_grad()
        cond = exemplar.to(hyperparameters["device"])
        image = image.to(hyperparameters["device"])
        loss = diffusion_model(image, cond)
        loss.backward()
        mean_loss += loss.item()
        optimizer.step()
        pbar.set_description(f"loss: {loss:.4f}")

        if step % 50 == 0:
            with torch.no_grad():
                loss_te = diffusion_model(one_batch_test_image, one_batch_test_exemplar)
                if not args.debug:
                    wandb.log({"loss": loss,
                           "test_loss": loss_te})
    mean_loss /= len(train_loader)

    #diffusion_model.eval()
    if epoch % 5 == 0 or epoch == hyperparameters["epochs"]-1:
        diffusion_model.eval()
        cond_tr = train_exemplar.repeat(10, 1, 1, 1).to(hyperparameters["device"])
        cond_te = test_exemplar.repeat(10, 1, 1, 1).to(hyperparameters["device"])
        with torch.no_grad():
            sample_images_train = diffusion_model.sample_c(image_size=hyperparameters["image_size"], batch_size=nb_samples,
                                        channels=hyperparameters["channels"], cond=cond_tr)
            sample_images_test = diffusion_model.sample_c(image_size=hyperparameters["image_size"], batch_size=nb_samples,
                                        channels=hyperparameters["channels"], cond=cond_te)

        to_show_tr = torch.cat([train_exemplar, sample_images_train.cpu()], dim=0)
        to_show_te = torch.cat([test_exemplar, sample_images_test.cpu()], dim=0)
        if not args.debug:
            images_tr = wandb.Image(make_grid(to_show_tr, ncol=10, nrow=11, normalize=True, scale_each=True), caption='generation training ep:{}'.format(epoch))
            images_te = wandb.Image(make_grid(to_show_te, ncol=10, nrow=11, normalize=True, scale_each=True),
                             caption='generation test ep:{}'.format(epoch))
            wandb.log({"samples": images_tr,
                   "samples test": images_te})
        else:
            plot_img(to_show_tr, ncol=10, nrow=11, normalize=True, scale_each=True)
            plot_img(to_show_te, ncol=10, nrow=11, normalize=True, scale_each=True)

    if mean_loss <= mean_loss_old:
        if not args.debug:
            torch.save(diffusion_model.state_dict(), args.snap_dir + '_best.model')
    mean_loss_old = mean_loss
    if not args.debug:
        torch.save(diffusion_model.state_dict(), args.snap_dir + '_end.model')

if not args.debug:
    torch.save(diffusion_model.state_dict(), args.snap_dir + '_end.model')


#if args.generate_img:
#    all_generation = []
#    all_exemplar = []
#    if args.dataset == 'omniglot':
#        nb_sample_per_class = 20

#    else:
#        nb_sample_per_class = args.sample_per_class
#    args.batch_size = nb_sample_per_class
#    kwargs = {'preload': args.preload}
#    train_loader, test_loader, args = load_dataset_exemplar(args,
#                                                            shape=args.input_shape,
#                                                            shuffle=False,
#                                                            **kwargs)
#    with torch.no_grad():
#        for idx_batch, (image, exemplar, label) in enumerate(test_loader):
#            if idx_batch % 10 == 0:
#                print('{}/{}'.format(idx_batch+1, len(test_loader)))

#            exemplar = exemplar.to(hyperparameters["device"])
#            data = torch.zeros_like(exemplar)
#            generation = diffusion_model.sample_c(image_size=hyperparameters["image_size"], batch_size=data.size(0),
#                                     channels=hyperparameters["channels"], cond=exemplar)

#            all_generation.append(generation.cpu().numpy())
#            all_exemplar.append(exemplar[0:1].cpu().numpy())

    all_generation = np.stack(all_generation, 0)
    all_exemplar = np.stack(all_exemplar, 0)
    np.savez_compressed(args.snap_dir + 'generated_img', data=all_generation, exemplar=all_exemplar)
    print('generation has been saved')