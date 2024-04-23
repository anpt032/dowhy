import os
import torch
import random
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data import TensorDataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity

import tensorflow_datasets as tfds

from dowhy.causal_prediction.datasets.base_dataset import MultipleDomainDataset


class SmallNORB(VisionDataset):

    @property
    def source_files(self) -> List:
        return [
            'dataset_info.json',
            'features.json',
            'label_category.labels.txt',
            'smallnorb-test.tfrecord-00000-of-00001',
            'smallnorb-train.tfrecord-00000-of-00001'
        ]
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, '2.0.0')

    def __init__(
        self, 
        root: str = None, 
        train: bool = True, 
        transforms: Callable[..., Any] | None = None, 
        transform: Callable[..., Any] | None = None, 
        target_transform: Callable[..., Any] | None = None,
        download: bool = True
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.train = train

        if download:
            self.download()

        self.data, self.targets, self.lightings, self.azimuths = self._load_data()

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, filename))
            for filename in self.source_files
        )

    def download(self):

        if self._check_exists():
            return
        
        os.makedirs(self.raw_folder, exist_ok=True)

        try:
            tfds.load('smallnorb', data_dir=self.root, download=True)
        except Exception as e:
            raise Exception(f'Exception while downloading:\n{e}')
        
    def _load_data(self):
        # Load the dataset using TensorFlow Datasets
        dataset, info = tfds.load('smallnorb', data_dir=self.root, download=False, split='train' if self.train else 'test', with_info=True)

        images = []
        labels = []
        lightings = []
        azimuths = []
        for example in dataset:
            images.append(example['image'].numpy())
            labels.append(example['label_category'].numpy())
            lightings.append(example['label_lighting'].numpy())
            azimuths.append(example['label_azimuth'].numpy())

        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        lightings = np.array(lightings)
        azimuths = np.array(azimuths)
        return images, labels, lightings, azimuths
    

class SmallNorbCausalAttribute(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["+90%", "+95%", "0%", "0%"]
    INPUT_SHAPE = (96, 96, 1)

    def __init__(self, root, download=True) -> None:
        super().__init__()

        if root is None:
            raise ValueError("Data directory not specified!")
        
        original_dataset_tr = SmallNORB(root, train=True, download=download)
        
        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets
        # original_lightings = original_dataset_tr.lightings
        # original_azimuths = original_dataset_tr.azimuths

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        # original_lightings = original_lightings[shuffle]
        # original_azimuths = original_azimuths[shuffle]

        self.datasets = []

        environments = (0.1, 0.05, 1)
        for i, env in enumerate(environments[:-1]):
            images = original_images[:24300][i::2]
            labels = original_labels[:24300][i::2]
            self.datasets.append(self.lighting_dataset(images, labels, env))

        # test environment
        original_dataset_te = SmallNORB(root, train=False, download=download)
        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        self.datasets.append(self.lighting_dataset(original_images, original_labels, environments[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 5

    def lighting_dataset(self, images, labels, environment):

        labels = self.add_noise(labels, 0.05)

        lightings = self.lightings_from_labels(labels, environment)

        x = images
        y = labels
        a = torch.unsqueeze(lightings, 1)

        return TensorDataset(x, y, a)

    def add_noise(self, labels: List, rate: float = 0.05):

        n_changes = int(len(labels) * rate)

        indices_to_change = random.sample(range(len(labels)), n_changes)

        for index in indices_to_change:
            labels[index] = random.choice([label for label in range(5) if label != labels[index]])

        return labels

    def lightings_from_labels(self, labels, environment):

        lightings = [label*2+1 for label in labels]

        lightings = self.add_noise(lightings, environment)

        return lightings
