import json
import torch

from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform


def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment='none',
    image=None,                         # image config
    dataset='unused',                   # ignore
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Override augment if not training
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    log_ids = get_split(f"mini_{split}" if version == "mini" else split, "argoverse2")

    return [Argoverse2GeneratedDataset(s, labels_dir, transform=transform) for s in log_ids if Path(labels_dir, f'{s}.json').exists()]


class Argoverse2GeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = Sample(**self.samples[idx])

        if self.transform is not None:
            data = self.transform(data)

        return data
