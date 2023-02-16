from . import nuscenes_dataset
from . import nuscenes_dataset_generated
from . import argoverse2_dataset

MODULES = {
    'nuscenes': nuscenes_dataset,
    'nuscenes_generated': nuscenes_dataset_generated,
    'argoverse2': argoverse2_dataset
}


def get_dataset_module_by_name(name):
    return MODULES[name]
