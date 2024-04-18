from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch


class VCDataset(Dataset):
    def __init__(self, datasetX, datasetY=None, max_mask_rate=0.5, device=torch.device("cpu")):
        self.device = device
        self.specX = datasetX["spec"].to(device)
        self.specY = datasetY["spec"].to(device)
        self.meanX = datasetX["mean"].to(device)
        self.meanY = datasetY["mean"].to(device)
        self.stdX = datasetX["std"].to(device)
        self.stdY = datasetY["std"].to(device)
        self.max_mask_rate = max_mask_rate

        self.LEN = min(self.specX.shape[0], self.specY.shape[0])

    def __len__(self):
        return self.LEN

    def __getitem__(self, index):
        specX, specY = self.specX[index], self.specY[index]
        lenX, lenY = specX.shape[1], specY.shape[1]
        maskX, maskY = torch.ones_like(specX), torch.ones_like(specY)

        maskLenX = torch.randint(0, int(lenX * self.max_mask_rate), (1,))
        maskStartX = torch.randint(0, int(lenX - maskLenX), (1,))
        maskX[:, maskStartX:(maskStartX+maskLenX)] = 0
        maskLenY = torch.randint(0, int(lenY * self.max_mask_rate), (1,))
        maskStartY = torch.randint(0, int(lenY - maskLenY), (1,))
        maskY[:, maskStartY:(maskStartY+maskLenY)] = 0

        return specX.unsqueeze(0), maskX.unsqueeze(0), specY.unsqueeze(0), maskY.unsqueeze(0)

    def to(self, *args, **kwargs):
        self.specX = self.specX.to(*args, **kwargs)
        self.specY = self.specY.to(*args, **kwargs)
        self.meanX = self.meanX.to(*args, **kwargs)
        self.meanY = self.meanY.to(*args, **kwargs)
        self.stdX = self.stdX.to(*args, **kwargs)
        self.stdY = self.stdY.to(*args, **kwargs)
        self.device = self.specX.device
        return self

    def reshuffle(self):
        idxX = torch.randperm(self.specX.shape[0])
        idxY = torch.randperm(self.specY.shape[0])
        self.specX = self.specX[idxX, :, :]
        self.specY = self.specY[idxY, :, :]


class VCDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.dataset, VCDataset), \
            f"Dataset for VCDataLoader must be an instance of VCDataset, but {type(self.dataset)} given."

    def __iter__(self, *args, **kwargs):
        """
        Wrapper for super().__iter__(), which is called after a traversal through the DataLoader object.
        This customized implemention ensures that the Dataset is reshuffled after each epoch, maximizing the use of data.

        Returns:
            whatever super().__iter__() returns
        """
        self.dataset.reshuffle()
        return super().__iter__(*args, **kwargs)


if __name__ == '__main__':
    import os
    import sys
    base_folder = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    if base_folder not in sys.path:
        sys.path.append(base_folder)
    if True:
        import config

    # Trivial test for dataset class
    trainX = torch.load(os.path.abspath(os.path.join(
        base_folder, config.preprocessed_train_data, "./Ikura/Ikura.data"
    )))
    trainY = torch.load(os.path.abspath(os.path.join(
        base_folder, config.preprocessed_train_data, "./Trump/Trump.data"
    )))
    dataset = VCDataset(trainX, trainY)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    for i, (X, maskX, Y, maskY) in enumerate(dataloader):
        print(i, X.shape, maskY.shape, Y.shape, maskY.shape)
        assert X.shape == maskY.shape == Y.shape == maskY.shape, \
            "Conflicting shape."
    for i, (X, maskX, Y, maskY) in enumerate(dataloader):
        print(i, X.shape, maskY.shape, Y.shape, maskY.shape)
        assert X.shape == maskY.shape == Y.shape == maskY.shape, \
            "Conflicting shape."
