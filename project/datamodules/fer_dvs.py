import os
import numpy as np
from tonic.dataset import Dataset
import pandas as pd
import h5py


class FerDVS(Dataset):
    sensor_size = (200, 200, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names
    available_datasets = ("CKPlusDVS", "ADFESDVS", "CASIADVS", "MMIDVS")
    classes = [
        "happy",
        "fear",
        "surprise",
        "anger",
        "disgust",
        "sadness",
    ]  # 6 labels

    def __init__(
        self,
        save_to,
        dataset="CKPlusDVS",
        train=True,
        fold=0,
        transform=None,
        target_transform=None,
    ):
        super(FerDVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        if fold < 0 or fold > 9:
            raise ValueError(
                f"The fold parameter must be an integer between 0 and 9. Got: {fold}"
            )

        if dataset not in self.available_datasets:
            raise ValueError(
                f"dataset must be one of the values {self.available_datasets}. Got: {dataset}"
            )

        self.fold = fold
        self.train = train
        self.dataset = dataset

        self.dir_path = os.path.join(self.location_on_system, self.dataset)

        # load csv file
        self.targets: pd.DataFrame = self._load_csv_file()

    def _load_csv_file(self):
        csv_path = os.path.join(self.dir_path, "folds.csv")
        folds = pd.read_csv(
            csv_path,
            delimiter=";",
            header=None,
            names=[
                "subject",
                "sequence",
                "label",
                "fold0",
                "fold1",
                "fold2",
                "fold3",
                "fold4",
                "fold5",
                "fold6",
                "fold7",
                "fold8",
                "fold9",
            ],
            dtype={
                "subject": str,
                "sequence": int,
                "label": int,
                "fold0": bool,
                "fold1": bool,
                "fold2": bool,
                "fold3": bool,
                "fold4": bool,
                "fold5": bool,
                "fold6": bool,
                "fold7": bool,
                "fold8": bool,
                "fold9": bool,
            },
        )

        # only keep the relevant data (i.e. the train or test split for the specified fold)
        fold_key = f"fold{self.fold}"  # key of the fold
        filter = folds[fold_key] == (not self.train)
        return folds[filter]

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        entry = self.targets.iloc[index]
        subject = entry["subject"]
        sequence = entry["sequence"]
        target = entry["label"] - 1  # start at 0




        # Erreur de type corrigée : target = target.astype(np.longlong)
        target = target.astype(np.longlong)             




        
                 

        data_path = os.path.join(self.dir_path, subject, f"{str(sequence).zfill(3)}.h5")
        data = h5py.File(data_path, "r")
        orig_events = np.array(data["events"])
        events = np.empty(len(orig_events), dtype=self.dtype)
        events["x"] = orig_events[:, 1]
        events["y"] = orig_events[:, 2]
        events["t"] = orig_events[:, 0]
        events["p"] = orig_events[:, 3]

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.targets)

    def _check_exists(self):
        return os.path.isdir(
            os.path.join(
                self.location_on_system, self.folder_name
            )  # check if directory exists
        ) and self._folder_contains_at_least_n_files_of_type(100, ".h5")
