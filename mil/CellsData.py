import pickle
import torch
import numpy as np
from scipy.sparse import load_npz
from mil import PROJECT_ROOT


class CellsData:
    """
    Dataset class of the `hlce_subset.h5ad` file
    """

    def __init__(
        self, meta_path: str = "data/dataset_meta.pcl", split: str = "train"
    ) -> None:
        """Initializes the CellsData object

        Args:
            meta_path (str, optional): Path to the pickled DatasetMeta object.
                Defaults  to 'data/dataset_meta.pcl'.
            split (str, optional): Dataset split as defined in meta.
                Options: ('train', 'val', 'test', 'cv').  Defaults to 'train'.
        """
        assert split in ("train", "val", "test", "cv")

        with open(PROJECT_ROOT / meta_path, "rb") as f:
            meta = pickle.load(f)

        self.meta = meta
        self.split = split

        reversed_index_mapping = dict()
        if split in ("train", "val", "test"):
            for master_index, split_and_index in meta.index_split_dict.items():
                item_split, split_index = split_and_index
                if item_split == split:
                    reversed_index_mapping[split_index] = master_index
        elif split == "cv":
            len_val = 0
            for master_index, split_and_index in meta.index_split_dict.items():
                item_split, split_index = split_and_index
                if item_split == "val":
                    reversed_index_mapping[split_index] = master_index
                    len_val += 1
            for master_index, split_and_index in meta.index_split_dict.items():
                item_split, split_index = split_and_index
                if item_split == "train":
                    reversed_index_mapping[split_index + len_val] = master_index

        self.index_mapping = reversed_index_mapping
        self.n = len(list(reversed_index_mapping.keys()))

        X_processed = load_npz(
            PROJECT_ROOT / "data" / meta.processed_x_path.split("/")[-1]
        )
        X_embed = np.load(PROJECT_ROOT / "data" / meta.x_embed_path.split("/")[-1])[
            "arr_0"
        ]
        # self.X =torch.sparse_csr_tensor(rows, columns, data)
        self.X = dict()
        self.X_embed = dict()

        for index, master_index in self.index_mapping.items():
            bag_index = self.meta.index_bag_dict[index]
            X = X_processed[bag_index, :]
            coord = np.array(X.nonzero())
            data = X.data
            shape = X.shape
            self.X[index] = torch.sparse_coo_tensor(
                coord, data, size=shape, dtype=torch.float32
            )
            self.X_embed[index] = torch.tensor(X_embed[bag_index, :])

        y_path = PROJECT_ROOT / "data" / meta.y_path.split("/")[-1]
        y_column = np.load(y_path)["arr_0"].reshape(-1, 1)
        y_first_column = np.ones_like(y_column) - y_column
        y = np.concatenate([y_first_column, y_column], axis=1)
        y = torch.tensor(y).float()
        self.y = y

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        master_index = self.index_mapping[i]

        X = self.X[i]
        X_embed = self.X_embed[i]
        y = self.y[master_index]

        return {"bag": X, "bag_embed": X_embed, "y": y}

    def __iter__(self):
        self.iter_counter = 0
        return self

    def __next__(self):
        if self.iter_counter >= self.n:
            raise StopIteration
        else:
            self.iter_counter += 1
            return self.__getitem__(self.iter_counter - 1)
