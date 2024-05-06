import torch
from torchvision.datasets import MNIST

from dowhy.causal_prediction_selfcode.datasets.base_dataset import MultipleDomainDataset

"""
    MNIST Causal, Independent and Causal+Independent datasets

    The class structure for datasets is adapted from OoD-Bench:
        @inproceedings{ye2022ood,
         title={OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization},
         author={Ye, Nanyang and Li, Kaican and Bai, Haoyue and Yu, Runpeng and Hong, Lanqing and Zhou, Fengwei and Li, Zhenguo and Zhu, Jun},
         booktitle={CVPR},
         year={2022}
        }

    * dataset initialized from torchvision.datasets.MNIST
    * We assume causal attribute (Acause) = color, independent attribute (Aind) = rotation
    * Environments/domains stored in list self.datasets (required for all datasets)
    * Default env structure is TensorDataset(x, y, a)
        * a is a combine tensor for all attributes (metadata) a1, a2, ..., ak
        * a's shape is (n, k) where n is the number of samples in the environment 
"""


# single-attribute Causal
class MNISTCausalAttribute(MultipleDomainDataset):
    N_STEPS = 5001
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["+90%", "+80%", "-90%", "-90%"]
    INPUT_SHAPE = (2, 14, 14)

    def __init__(self, root, download=True):
        """
            Class for MNISTCausalAttribute dataset.

            :param root: The directory where data can be found (or should be downloaded to, if it does not exist).
            :param download: Binary flag indicating whether data should be downloaded
            :returns: an instance of MultipleDomainDataset class

        """
        super().__init__()
        
        if root is None:
            raise ValueError("Data directory is not specified!")
        
        original_dataset_tr = MNIST(root, train=True, download=download)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        environment = (0.1, 0.2, 0.9)
        for i, env in enumerate(environment[:-1]):
            images = original_images[:50000][i::2]
            labels = original_labels[:50000][i::2]
            self.datasets.append(self.color_dataset(images, labels, env))


        def color_dataset(self, images, labels, environment):
            """
                Transform MNIST datasets to introduce correlation between attribute (color) and label.
                There is a direct-causal relationship between label Y and color.

                :param images: original MNIST images
                :param labels: original MNIST labels
                :param environment: Value of correlation between color and label
                :returns: TensorDataset containing transformed images, labels, and attributes (color)
            
            """

            # Subsample 2x for computational convenience
            images = images.reshape((-1, 28, 28))[:, ::2, ::2]

            # Asign a binary label based on the digit
            labels = (labels < 5).float()

            # Flip the label with probability 0.25
            labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))
            
            images = torch.stack([images, images], dim=1)
            # Apply the color to the image by zeroing out the other color channel

        def torch_bernoulli_(self, p, size):
            return (torch.rand(size) < p).float()

        def torch_xor_(self, a, b):
            return (a - b).abs()