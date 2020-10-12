from fairseq.data import BaseWrapperDataset


class LengthenDataset(BaseWrapperDataset):
    """ Lengthen dataset that only has length one by returning the same item
    """

    def __init__(self, dataset, desired_size):
        super().__init__(dataset)
        assert len(dataset) == 1
        self.desired_size = desired_size
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[0]

    def __len__(self):
        return self.desired_size
