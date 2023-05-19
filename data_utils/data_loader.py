import torch.utils.data as data_utils
from .custom_loader import OmniglotDataset, QuickDraw_FS_clust
#from data_utils.custom_loader import QuickdrawSetsDatasetNS
from .custom_transform import Binarize, Invert, Scale_0_1, Dilate
import torchvision.transforms as tforms
import os
import PIL
from .custom_sampler import NShotTaskSampler
import torch


def load_omniglot(args, type='weak', NO_FS_flag=False, **kwargs):
    tr_eval = [tforms.Resize(args.input_shape[1:])]
    tr_train = [tforms.Resize(args.input_shape[1:])]
    if args.augment:
        tr_train += [tforms.RandomAffine((-45, 45))]
    tr_eval += [tforms.ToTensor(), Scale_0_1(), Invert()]
    tr_train += [tforms.ToTensor(), Scale_0_1(), Invert()]
    if args.input_type == 'binary':
        tr_eval += [Binarize(binary_threshold=0.5)]
        tr_train += [Binarize(binary_threshold=0.5)]

    tr_eval = tforms.Compose(tr_eval)
    tr_train = tforms.Compose(tr_train)
    dir_data = os.path.join(args.dataset_root, "omniglot", "omniglot-py")

    if type == 'weak':
        split_tr, split_te = 'weak_background', 'weak_evaluation'
    elif type == 'strong':
        split_tr, split_te = 'background', 'evaluation'
    else :
        raise NotImplementedError()

    train = OmniglotDataset(dir_data, split=split_tr, transform=tr_train,
                            preloading=args.preload, exemplar_transform=tr_eval, exemplar_type=args.exemplar_type)
    eval = OmniglotDataset(dir_data, split=split_te, transform=tr_eval, exemplar_transform=tr_eval,
                           preloading=args.preload, exemplar_type=args.exemplar_type)

    if hasattr(args, 'model_name'):
        if args.model_name in ['ns', 'tns', 'hfsgm', 'vfsddpm'] and (not NO_FS_flag):
            train = OmniglotSetsDatasetNS(train.data, train.label, split=split_tr, sample_size=args.sample_size,
                                          transform=tr_train, img_size=args.input_shape)
            eval = OmniglotSetsDatasetNS(eval.data, eval.label, split=split_te, sample_size=args.sample_size,
                                         transform=tr_eval, img_size=args.input_shape)

    return train, eval, None, None, args


def load_qd_clust(args, shape=None, NO_FS_flag=False, **kwargs):
    """
            Dataloading function for quickdraw as a training data and human_drawing as a testing dat
        """
    transforms = []
    transforms_training = []

    if shape is not None:
        expand = shape[0] if shape[0] != 1 else None
        resize = shape[1:]
        transforms += [tforms.Resize(resize, interpolation=PIL.Image.LANCZOS)]
        transforms_training += [tforms.Resize(resize, interpolation=PIL.Image.LANCZOS)]
        transforms += [tforms.ToTensor()]

        transforms_training += [tforms.ToTensor()]
        if expand is not None:
            transforms.append(lambda x: x.repeat([expand, 1, 1]))
            transforms_training.append(lambda x: x.repeat([expand, 1, 1]))

    else:
        args.input_shape = (1, 105, 105)
        transforms += [tforms.ToTensor()]
        transforms_training += [tforms.ToTensor()]
    transforms += [Scale_0_1()]
    transforms_training += [Scale_0_1()]

    if args.input_type == 'binary':
        transforms += [Binarize(binary_threshold=0.3)]
        transforms_training += [Binarize(binary_threshold=0.3)]
    trans = tforms.Compose(transforms)
    trans_training = tforms.Compose(transforms_training)

    if not hasattr(args, 'augment_class'):
        args.augment_class = False

    transform_variation = None
    if hasattr(args, 'transform_variation'):
        if args.transform_variation:
            transform_variation = [tforms.RandomHorizontalFlip(0.5),
                                   tforms.RandomVerticalFlip(0.5),
                                   tforms.RandomAffine(degrees=(-180, 180),
                                                        translate=(0.2, 0.2),
                                                       scale=(0.5, 1.5),
                                                       fillcolor=None),
                                   Scale_0_1(),
                                   Binarize(binary_threshold=0.3)]
            transform_variation = tforms.Compose(transform_variation)

    if not hasattr(args, 'sample_per_class'):
        args.sample_per_class = 500

    dir_data_qd = os.path.join(args.dataset_root, "quick_draw")

    train = QuickDraw_FS_clust(dir_data_qd, transform=trans_training, exemplar_type=args.exemplar_type,
                         augment_class=args.augment_class, transform_variation=transform_variation, train_flag=True,
                         sample_per_class=args.sample_per_class)
    test = QuickDraw_FS_clust(dir_data_qd, transform=trans_training, exemplar_type=args.exemplar_type, train_flag=False,
                        sample_per_class=args.sample_per_class)

    if hasattr(args, 'model_name'):
        if args.model_name in ['ns', 'tns', 'hfsgm', 'vfsddpm'] and (not NO_FS_flag):
            train = QuickdrawSetsDatasetNS(train.variation, train.targets, train_flag=train.train_flag, sample_size=args.sample_size,
                                          transform=trans_training)
            test = QuickdrawSetsDatasetNS(test.variation, test.targets, train_flag=test.train_flag, sample_size=args.sample_size,
                                         transform=trans_training)

    return train, test, None, None, args


def load_dataset_exemplar(args, shape=None, shuffle=True, drop_last=False, few_shot=False,
                          fixed_tasks=None, NO_FS_flag=False, **kwargs):

    if args.dataset == 'omniglot':
        train_set, test_set, train_set_exemplars, test_set_exemplars, args = load_omniglot(args, shape=shape, type='weak',
                                                                                                    NO_FS_flag=NO_FS_flag, **kwargs )

    elif args.dataset == 'quickdraw_clust':
        train_set, test_set, train_set_exemplars, test_set_exemplars, args = load_qd_clust(args, shape=shape, NO_FS_flag=NO_FS_flag,
                                                                                                ** kwargs)

    if few_shot and args.model_name in ['proto_net', 'res_net']:

        train_loader = data_utils.DataLoader(train_set,
                                             batch_sampler=NShotTaskSampler(train_set, args.episodes_per_epoch,
                                                                            args.n_train, args.k_train, args.q_train,
                                                                            fixed_tasks=fixed_tasks)
                                             )
        test_loader = data_utils.DataLoader(test_set,
                                            batch_sampler=NShotTaskSampler(test_set, args.episodes_per_epoch,
                                                                           args.n_test, args.k_test, args.q_test,
                                                                           fixed_tasks=fixed_tasks)
                                            )
    elif few_shot and args.model_name == 'maml':
        train_loader = data_utils.DataLoader(train_set,
                                             batch_sampler=NShotTaskSampler(train_set, args.epoch_len,
                                                                            args.n, args.k, args.q,
                                                                            num_tasks=args.meta_batch_size,
                                                                            fixed_tasks=fixed_tasks)
                                             )
        test_loader = data_utils.DataLoader(test_set,
                                            batch_sampler=NShotTaskSampler(test_set, args.eval_batches,
                                                                           args.n, args.k, args.q,
                                                                           num_tasks=args.meta_batch_size,
                                                                           fixed_tasks=fixed_tasks)
                                            )

    else:
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = data_utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last, generator=g)

        test_loader = data_utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last)

    return train_loader, test_loader, args

