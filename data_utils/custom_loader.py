from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as tforms
from .custom_transform import Binarize, Scale_0_1
#from utils.quickdraw_label import quickdraw_train_label, quickdraw_test_label, quickdraw_exemplar_idx, labels_to_idx
import torchvision.transforms.functional as TF
import random


class OmniglotDataset(Dataset):
    def __init__(self, root, split, transform=None, exemplar_transform=None, target_tranform=None, preloading=False, exemplar_type='first'):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if split not in ('background', 'evaluation', 'weak_background',
                         'weak_evaluation'):
            raise(ValueError, 'split must be one of (background, evaluation)')
        self.transform = transform
        self.exemplar_transform = exemplar_transform
        self.target_transform = target_tranform
        self.exemplar_type = exemplar_type
        self.preloading = preloading

        self.split = split
        self.root = root

        df_path = os.path.join(self.root, 'preload_df.pkl')

        if os.path.exists(df_path) and self.preloading:
            self.df = pd.read_pickle(df_path)
        else:
            self.df_train = pd.DataFrame(self.index_subset(self.root, 'background'))
            self.df_test = pd.DataFrame(self.index_subset(self.root, 'evaluation'))
            self.df = pd.concat([self.df_train, self.df_test], ignore_index=True)
            if self.preloading:
                self.df.to_pickle(df_path)

        if self.split == 'background':
            self.df = self.df[self.df['subset'] == 'background']
        elif self.split == 'evaluation':
            self.df = self.df[self.df['subset'] == 'evaluation']
        elif self.split == 'weak_evaluation':
            character_list = ['character01', 'character02', 'character03']
            self.df = self.df[self.df['character_number'].isin(character_list)]
        elif self.split == 'weak_background':
            character_list = ['character01', 'character02', 'character03']
            self.df = self.df[~self.df['character_number'].isin(character_list)]

        self.unique_characters = sorted(self.df['class_name'].unique())

        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        self.df = self.df.reset_index(drop=True)
        self.df = self.df.assign(id=self.df.index.values)

        if self.exemplar_type == 'first':
            self.df_exemplar = self.df[self.df['sample_number'].isin(['1'])]
        elif self.exemplar_type == 'shuffle':
            self.df_exemplar = self.df
        elif self.exemplar_type == 'prototype':
            if self.split == 'weak_evaluation':
                path_to_proto = os.path.join(self.root, 'te_proto_weak.pkl')
                flag = torch.load(path_to_proto)
                flag2 = torch.arange(flag.size(0))[flag == 1]
                self.df_exemplar = self.df[self.df['id'].isin(flag2)]
            elif self.split == 'weak_background':
                path_to_proto = os.path.join(self.root, 'tr_proto_weak.pkl')
                flag = torch.load(path_to_proto)
                flag2 = torch.arange(flag.size(0))[flag == 1]
                self.df_exemplar = self.df[self.df['id'].isin(flag2)]


            else :
                Exception('prototype exemplar implemented only for weak split for now')

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        if self.preloading:
            self.data, self.label = self.preload()

    def preload(self):
        all_images, all_classes = [], []

        path_preload = os.path.join(self.root, 'preload_'+ self.split )
        path_preload += '.pkl'
        if not os.path.exists(path_preload):
            progress_bar = tqdm(total=len(self.df))
            for idx_image in range(len(self.df)):
                image = Image.open(self.datasetid_to_filepath[idx_image], mode="r").convert("L")
                character_class = self.datasetid_to_class_id[idx_image]

                all_images.append(tforms.functional.pil_to_tensor(image))
                all_classes.append(character_class)
                progress_bar.update(1)
            progress_bar.close()
            all_images = torch.stack(all_images, dim=0)
            all_classes = torch.tensor(all_classes)
            torch.save([all_images, all_classes], path_preload)
        else:
            [all_images, all_classes] = torch.load(path_preload)

        return all_images, all_classes

    def __getitem__(self, item):
        if self.preloading:
            image = tforms.functional.to_pil_image(self.data[item], mode="L")
            character_class = self.label[item].item()
        else:
            image = Image.open(self.datasetid_to_filepath[item], mode="r").convert("L")
            character_class = self.datasetid_to_class_id[item]

        if self.exemplar_type == 'shuffle':
            item_exemplar = self.df_exemplar[self.df_exemplar['class_id'] == character_class].sample(1).index.values[0]
        elif self.exemplar_type == 'first':
            item_exemplar = self.df_exemplar[self.df_exemplar['class_id'] == character_class].index.values[0]
        elif self.exemplar_type == 'prototype':
            item_exemplar = self.df_exemplar[self.df_exemplar['class_id'] == character_class].index.values[0]
        if self.preloading:
            image_exemplar = tforms.functional.to_pil_image(self.data[item_exemplar], mode="L")
        else:
            image_exemplar = Image.open(self.datasetid_to_filepath[item_exemplar], mode="r").convert("L")
            #

        if self.transform:
            image = self.transform(image)

        if self.exemplar_transform :
            exemplar = self.exemplar_transform(image_exemplar)
        else :
            exemplar = image_exemplar

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, exemplar, character_class


    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(dir_data, subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(dir_data + '/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(dir_data + '/images_{}/'.format(subset)):
            if len(files) == 0:
                continue
            sample_number = 0
            alphabet = root.split('/')[-2]
            character_number = root.split('/')[-1]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                sample_number += 1
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'character_number': character_number,
                    'class_name': class_name,
                    'sample_number': sample_number,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class QuickDraw_FS_clust(VisionDataset):
    """ The QuickDraw dataset

        Args:
            root (string): Root directory of dataset where the archive *.pt are stored.
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
        """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            transform_variation: Optional[Callable] = None,

            target_transform: Optional[Callable] = None,
            sample_per_class: Optional[int] = 500,
            exemplar_type: Optional[str] = 'first',
            augment_class: Optional[bool] = False,
            train_flag:Optional[bool] = None

    ) -> None:
        super(QuickDraw_FS_clust, self).__init__(root, transform=transform, target_transform=target_transform)
        self.transform_variation = transform_variation
        self.param_RandomAffine = [(-90, 90), (0.3, 0.3), (0.5, 1.5), (-30, 30, -30, 30)]
        self.train_flag = train_flag
        self.sample_per_class = sample_per_class
        self.augment_class = augment_class
        if self.augment_class:
            self.augmentation_class = [
                TF.hflip, TF.vflip,
                lambda img: TF.rotate(img, 0),
                tforms.RandomAffine(*self.param_RandomAffine)
            ]

        if self.sample_per_class == 500:
            loaded_file = np.load(root + '/all_qd_fs_shuffled.npz')
        else:
            raise ValueError("nb sample should not be different than 500")

        images = loaded_file['data']
        prototype = loaded_file['prototype']

        if train_flag is not None:
            if self.train_flag:
                selected_labels = np.arange(550)
            else:
                selected_labels = np.arange(550, len(images))
        else:
            selected_labels = np.arange(len(images))

        self.exemplar_type = exemplar_type

        if self.exemplar_type in ['prototype', 'first']:
            exemplar = prototype[selected_labels,0]
            self.exemplar = torch.from_numpy(exemplar)
        elif self.exemplar_type == 'shuffle':
            self.exemplar = None
        else:
            raise NotImplementedError()
        self.variation = images[selected_labels, :self.sample_per_class].reshape(-1, 1, 48, 48)
        self.variation = torch.from_numpy(self.variation)
        intermediary = np.arange(len(selected_labels)).reshape(-1, 1)
        self.targets = torch.from_numpy(np.repeat(intermediary, self.sample_per_class, axis=1).flatten())


        id_list = []
        class_id_list = []

        for each_elem in range(self.variation.size(0)):
            id_list.append(each_elem)
            label = self.targets[each_elem]
            class_id_list.append(int(label.item()))

        self.df = pd.DataFrame(data={'id': id_list, 'class_id': class_id_list})

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img_variation, target = self.variation[index].view(48, 48), int(self.targets[index])

        idx_exemplar = target

        if self.exemplar_type in ['prototype', 'first']:
            img_exemplar = self.exemplar[idx_exemplar].view(48, 48)
        elif self.exemplar_type == 'shuffle':
            item_exemplar = self.df[self.df['class_id'] == target].sample(1)['id'].values[0]
            img_exemplar = self.variation[item_exemplar].view(48, 48)
        img_variation = Image.fromarray(img_variation.numpy())
        img_exemplar = Image.fromarray(img_exemplar.numpy())
        if self.augment_class:
            rnd_idx = np.random.choice(np.arange(len(self.augmentation_class)), p=[0.2, 0.2, 0.2, 0.4])
            if rnd_idx == 3:
                param = self.augmentation_class[rnd_idx].get_params(*self.param_RandomAffine, img_size=[48, 48])
                img_variation = TF.affine(img_variation, *param)
                img_exemplar = TF.affine(img_exemplar, *param)


            else:
                trans = self.augmentation_class[rnd_idx]
                img_variation = trans(img_variation)
                img_exemplar = trans(img_exemplar)

        if self.transform is not None:
            img_variation = self.transform(img_variation)
            img_exemplar = self.transform(img_exemplar)
        if self.transform_variation is not None:
            img_variation = self.transform_variation(img_variation)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_variation, img_exemplar, target

    def __len__(self) -> int:
        return len(self.variation)
