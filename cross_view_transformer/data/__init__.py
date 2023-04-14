from . import nuscenes_dataset
from . import nuscenes_dataset_generated
from . import argoverse2_dataset
from . import argoverse2_dataset_generated

MODULES = {
    'nuscenes': nuscenes_dataset,
    'nuscenes_generated': nuscenes_dataset_generated,
    'argoverse2': argoverse2_dataset,
    'argoverse2_generated': argoverse2_dataset_generated,
}


def get_dataset_module_by_name(name):
    return MODULES[name]
