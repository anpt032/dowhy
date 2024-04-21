import tensorflow_datasets as tfds
from base_dataset import MultipleDomainDataset
from torch.utils.data import TensorDataset


class SmallNorbCausalAttribute(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["+90%", "+80%", "-90%", "-90%"]
    INPUT_SHAPE = (96, 96, 1)

    def __init__(self, root, download=True):
        """Class for SmallNorbCausalAttribute dataset.

        :param root: The directory where data can be found (or should be downloaded to, if it does not exist).
        :param download: Binary flag indicating whether data should be downloaded
        :returns: an instance of MultipleDomainDataset class

        """

        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")

        original_dataset_tr = tfds.load(name='smallnorb', data_dir=root, download=download, split=tfds.Split.TRAIN)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets

        self.datasets = []

        images = original_images[50000:]
        labels = original_labels[50000:]
        self.datasets.append(TensorDataset(images, labels, 0))

        # test environment
        original_dataset_te = tfds.load(name='smallnorb', data_dir=root, download=download, split=tfds.Split.TEST)
        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        self.datasets.append(TensorDataset(original_images, original_labels, 0))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 5
