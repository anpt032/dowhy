import os
import torch
import random
import warnings 
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

from torch.utils.data import TensorDataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity

import tensorflow_datasets as tfds

from dowhy.causal_prediction.datasets.base_dataset import MultipleDomainDataset


class SmallNORB(VisionDataset):

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - four-legged animals", 
        "1 - human figures", 
        "2 - airplanes", 
        "3 - trucks", 
        "4 - cars"
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

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
        
        # os.makedirs(self.raw_folder, exist_ok=True)
        
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
            images.append(np.squeeze(example['image'].numpy()))
            labels.append(example['label_category'].numpy())
            lightings.append(example['label_lighting'].numpy())
            azimuths.append(example['label_azimuth'].numpy())

        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        lightings = np.array(lightings)
        azimuths = np.array(azimuths)

        # return images, labels, lightings, azimuths

        images_tensor = torch.tensor(images, dtype=torch.uint8)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        lightings_tensor = torch.tensor(lightings, dtype=torch.long)
        azimuths_tensor = torch.tensor(azimuths, dtype=torch.long)

        return images_tensor, labels_tensor, lightings_tensor, azimuths_tensor
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    

class SmallNorbCausalAttribute(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["+90%", "+95%", "-0%", "-0%"]
    INPUT_SHAPE = (5, 48, 48)

    def __init__(self, root, download=True) -> None:
        super().__init__()

        if root is None:
            raise ValueError("Data directory not specified!")
        
        original_dataset_tr = SmallNORB(root, train=True, download=download)
        
        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets
        original_lightings = original_dataset_tr.lightings
        # original_azimuths = original_dataset_tr.azimuths

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        original_lightings = original_lightings[shuffle]
        # original_azimuths = original_azimuths[shuffle]

        self.datasets = []

        environments = (0.1, 0.05, 1)
        for i, env in enumerate(environments[:-1]):
            images = original_images[:20000][i::2]
            labels = original_labels[:20000][i::2]
            lightings = original_lightings[:20000][i::2]
            self.datasets.append(self.lighting_dataset(images, labels, lightings, env))

        images = original_images[20000:]
        labels = original_labels[20000:]
        lightings = original_lightings[20000:]
        self.datasets.append(self.lighting_dataset(images, labels, lightings, environment=environments[-1]))

        # test environment
        original_dataset_te = SmallNORB(root, train=False, download=download)
        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        original_lightings = original_dataset_te.lightings
        self.datasets.append(self.lighting_dataset(original_images, original_labels, original_lightings, environments[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 5

    def lighting_dataset(self, images, labels, _lightings, environment):

        print(images.shape)

        # images = images.reshape((-1, 480, 480, ))[:, ::2, ::2]
        images = images[:, ::2, ::2]

        labels = self.add_noise(labels, 0.05)
        labels = labels.float()

        # _images, _labels, _lightings = self.lightings_selection(images, labels, lightings, environment)

        lightings = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))

        print(images.shape)

        images = torch.stack([images, images, images, images, images], dim=1)

        print(images.shape)
        print(labels.shape)
        print(lightings.shape)
        print(torch.tensor(range(len(images))))
        print((4 - lightings).long())

        images[torch.tensor(range(len(images))), (4 - lightings).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        a = torch.unsqueeze(lightings, 1)

        return TensorDataset(x, y, a)

    def add_noise(self, labels: List, rate: float = 0.05):

        n_changes = int(len(labels) * rate)

        indices_to_change = random.sample(range(len(labels)), n_changes)

        for index in indices_to_change:
            labels[index] = random.choice([label for label in range(5) if label != labels[index]])

        return labels
    
    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a + b) % 5

    def lightings_selection(self, images, labels, lightings, environment):

        _images = []
        _labels = []
        _lightings = []
        _not_hold_indices = []

        if environment != 1:
            for i in range(len(images)):
                if torch.equal(labels[i], lightings[i]):
                    _images.append(images[i])
                    _labels.append(labels[i])
                    _lightings.append(lightings[i])
                else:
                    _not_hold_indices.append(i)


            n_error_elements = int(environment*len(_images))

            for i in range(n_error_elements):
                if i < len(_not_hold_indices):
                    _images.append(images[_not_hold_indices[i]])
                    _labels.append(labels[_not_hold_indices[i]])
                    _lightings.append(lightings[_not_hold_indices[i]])
                else:
                    break
        else:
            for i in range(len(images)):
                if not torch.equal(labels[i], lightings[i]):
                    _images.append(images[i])
                    _labels.append(labels[i])
                    _lightings.append(lightings[i])

        return torch.tensor(np.array(_images), dtype=torch.uint8), torch.LongTensor(np.array(_labels)), torch.LongTensor(np.array(_lightings))
