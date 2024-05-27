from wilds.datasets.civilcomments_dataset import CivilCommentsDataset

from dowhy.causal_prediction.datasets.base_dataset import MultipleDomainDataset


class DWCivilCommentsDataset(CivilCommentsDataset):

    def __init__(self, version=None, root_dir='data', download=True, split_scheme='official'):
        super().__init__(version, root_dir, download, split_scheme)

        self.