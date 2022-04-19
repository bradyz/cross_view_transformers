from . import lyft_dataset
from . import lyft_dataset_generated

from . import argoverse_dataset
from . import argoverse_dataset_generated

from . import nuscenes_dataset
from . import nuscenes_dataset_generated


MODULES = {
    'nuscenes': nuscenes_dataset,
    'nuscenes_generated': nuscenes_dataset_generated,

    'lyft': lyft_dataset,
    'lyft_generated': lyft_dataset_generated,

    'argoverse': argoverse_dataset,
    'argoverse_generated': argoverse_dataset_generated,
}


def get_dataset_module_by_name(name):
    return MODULES[name]
