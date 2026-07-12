import json, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

BASE_DATA = r"C:\Users\sivak\Downloads\crime data\TFIF\data"

class VideoFeatureDataset(Dataset):
    def __init__(self, split="train", augment=False):
        meta_path   = BASE_DATA + r"\metadata.json"
        splits_path = BASE_DATA + r"\splits.json"
        with open(meta_path)   as f: self.records = json.load(f)
        with open(splits_path) as f: splits = json.load(f)
        self.indices = splits[split]
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        rec  = self.records[self.indices[i]]
        feat = torch.load(rec["feature_path"], map_location="cpu")  # (16, 960)
        label = rec["label"]
        if self.augment:
            # Mild time-jitter: randomly permute up to 2 adjacent frames
            feat = self._time_jitter(feat)
            # Feature dropout (zeroes out random dims)
            mask = (torch.rand(feat.shape) > 0.05).float()
            feat = feat * mask
        return feat.float(), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _time_jitter(seq, max_swap=2):
        seq = seq.clone()
        T = seq.shape[0]
        for _ in range(max_swap):
            i = torch.randint(0, T-1, (1,)).item()
            seq[[i, i+1]] = seq[[i+1, i]]
        return seq


def get_loaders(batch_size=16, num_workers=0):
    train_ds = VideoFeatureDataset("train", augment=True)
    val_ds   = VideoFeatureDataset("val",   augment=False)
    test_ds  = VideoFeatureDataset("test",  augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tl, vl, tel = get_loaders(batch_size=4)
    x, y = next(iter(tl))
    print("Feature batch:", x.shape, "Labels:", y)
