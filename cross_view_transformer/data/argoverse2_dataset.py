import json
import torch
import pdb
import pyarrow.feather as feather
from typing import List, Tuple, Any, Dict
from PIL import Image
import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from .common import get_split, INTERPOLATION, get_view_matrix, get_pose
from .transforms import Sample, LoadDataTransform, SaveDataTransform

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.datasets.sensor.constants import RingCameras, AnnotationCategories
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.map_api import ArgoverseStaticMap, DrivableAreaMapLayer
from av2.geometry.se3 import SE3
from av2.geometry.geometry import compute_interior_points_mask
from av2.rendering.rasterize import xyz_to_bev
from av2.structures.cuboid import CuboidList
from av2.utils.io import read_city_SE3_ego, read_img
import random


STATIC = [
    AnnotationCategories.BOLLARD ,
    AnnotationCategories.CONSTRUCTION_BARREL ,
    AnnotationCategories.CONSTRUCTION_CONE ,
    AnnotationCategories.MOBILE_PEDESTRIAN_CROSSING_SIGN ,
    AnnotationCategories.SIGN ,
    AnnotationCategories.STOP_SIGN ,
    AnnotationCategories.TRAFFIC_LIGHT_TRAILER ,
    AnnotationCategories.MOBILE_PEDESTRIAN_CROSSING_SIGN ,

          ]
DYNAMIC = [
    AnnotationCategories.ARTICULATED_BUS ,
    AnnotationCategories.BICYCLE ,
    AnnotationCategories.ANIMAL ,
    AnnotationCategories.BICYCLIST ,
    AnnotationCategories.BOX_TRUCK ,
    AnnotationCategories.BUS ,
    AnnotationCategories.DOG ,
    AnnotationCategories.LARGE_VEHICLE ,
    AnnotationCategories.MESSAGE_BOARD_TRAILER ,
    AnnotationCategories.MOTORCYCLE ,
    AnnotationCategories.MOTORCYCLIST ,
    AnnotationCategories.OFFICIAL_SIGNALER ,
    AnnotationCategories.PEDESTRIAN ,
    AnnotationCategories.BICYCLIST ,
    AnnotationCategories.RAILED_VEHICLE ,
    AnnotationCategories.REGULAR_VEHICLE ,
    AnnotationCategories.SCHOOL_BUS ,
    AnnotationCategories.STROLLER ,
    AnnotationCategories.TRUCK ,
    AnnotationCategories.TRUCK_CAB ,
    AnnotationCategories.VEHICULAR_TRAILER ,
    AnnotationCategories.WHEELCHAIR ,
    AnnotationCategories.WHEELED_RIDER ,
    AnnotationCategories.WHEELED_DEVICE ,
          ]
NUM_CLASSES = len(STATIC) + len(DYNAMIC)

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment="none",
    image=None,  # image config
    dataset="unused",  # ignore
    **dataset_kwargs,
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    transform = SaveDataTransform(labels_dir)

    # Format the split name
    # split = f"mini_{split}" if version == "mini" else split

    log_ids = get_split(f"mini_{split}" if version == "mini" else split, "argoverse2") # TODO: modify for argopaths (actually this seems fine)

    # av2_dataloader = AV2SensorDataLoader(dataset_dir, dataset_dir)

    sensor_dataloader = SensorDataloader(
            dataset_dir=dataset_dir,
            with_annotations=True,
            with_cache=False
        )
    
    for log_id in log_ids:
        assert Path.exists(Path(dataset_dir, split, log_id))

    return [Argoverse2Dataset(
        log_id=log_id,
        split=split,
        log_specific_dataset_path=Path(dataset_dir,split,log_id),
        synchronized_timestamps=
            sensor_dataloader.synchronization_cache.loc[(split, log_id, 'lidar')][['lidar']+[cam.value for cam in tuple(RingCameras)]],
        transform=SaveDataTransform(labels_dir), # TODO: Transform / SaveDataTransform
        ) for log_id in log_ids]


class Argoverse2Dataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                    log_id: str, 
                    split: str,
                    log_specific_dataset_path: Path,
                    synchronized_timestamps: pd.DataFrame,
                    transform=None,
                    bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
                ):

        print(f'Argoverse2Dataset building: {log_id}')
        self.log_id = log_id
        self.log_specific_dataset_path = log_specific_dataset_path
        self.poses = read_city_SE3_ego(self.log_specific_dataset_path)
        self.cameras = [PinholeCamera.from_feather(self.log_specific_dataset_path, cam.value) for cam in tuple(RingCameras)]
        self.avm = ArgoverseStaticMap.from_map_dir(Path(self.log_specific_dataset_path, 'map'), build_raster=True)
        self.synchronized_timestamps = synchronized_timestamps
        self.transform = transform

        self.view = get_view_matrix(**bev)
        self.bev_info = bev
        self.bev_shape = (bev['h'], bev['w'])
        self.intrinsics = {cam.cam_name:cam.intrinsics for cam in self.cameras}
        self.extrinsics = {cam.cam_name:cam.extrinsics for cam in self.cameras}

        # for all synchronized scenes get the following data
        self.samples = self.get_samples()
        # with open('/srv/share2/apatni30/tmp/AV2_dataset.pkl', 'wb') as pickle_file:
        #     pickle.dump(self, pickle_file)
        #     raise Exception("asdf")

    def get_samples(self) -> List[Dict[str, Any]]:
        data = []
        for i in range(len(self.synchronized_timestamps)):
            data.append(
                {
                'scene': self.log_id,
                'token': self.synchronized_timestamps.iloc[i]['lidar'].value,
                'pose': self.poses[self.synchronized_timestamps.iloc[i]['lidar'].value],
                # 'pose_inverse': self.poses[self.synchronized_timestamps.iloc[i]['lidar'].value].inverse(),
                'cam_ids': range(0,len(self.cameras)),
                'cam_channels': [cam.cam_name for cam in self.cameras],
                'intrinsics': self.intrinsics,
                'extrinsics': self.extrinsics,
                'images': [
                            Path(self.log_specific_dataset_path, 'sensors', 'cameras', camera.cam_name, f'{self.synchronized_timestamps.iloc[i][camera.cam_name].value}.jpg') for camera in self.cameras
                        ] # these are image paths
                }
            )
        return data

    def generate_bev(self, pose: SE3, timestamp: int) -> Tuple[np.ndarray, np.ndarray]:
        pose_inv = pose.inverse()
        xpoints = np.arange(-self.bev_info['h_meters']/2,self.bev_info['h_meters']/2,self.bev_info['h_meters']/self.bev_info['h'])
        ypoints = np.arange(-self.bev_info['w_meters']/2,self.bev_info['w_meters']/2,self.bev_info['w_meters']/self.bev_info['w'])
        zpoints = np.array([i/10.0 for i in range(20)])
        # xcoord, ycoord, z_coord= np.meshgrid(xpoints, ypoints)
        # xyz_coords = np.concatenate((xcoord, ycoord, z_coord),axis=2)
        # points_xy_city = location.transform_point_cloud(xyz_coords)

        # note: may have to change map() to list() if numpy upgraded >= 1.16
        points_xy_wrt_src = np.vstack(map(np.ravel, np.meshgrid(xpoints, ypoints, zpoints))).T
        points_xy_wrt_city = pose.transform_point_cloud(points_xy_wrt_src) 
        # Note that points_xy_wrt_city is a list of all the points (X,Y,Z) coords, not a 2D grid
        
        # # Project cuboids onto raster
        # dynamic_raster = np.zeros(shape=(points_xy_wrt_city.shape[0]))
        # static_raster = np.zeros(shape=(points_xy_wrt_city.shape[0]))
        
        # CMAP index -> color mapping for each point in city (N,3) (this determines dynamic vs static etc in the bev)
        cmap = np.zeros(shape=(points_xy_wrt_city.shape[0], 3))
        # Get driveable area raster (N,3) = (num points, (x,y,z))
        drivable_area_raster = self.avm.raster_drivable_area_layer.get_raster_values_at_coords(points_xy_wrt_city, 0)
        cmap[drivable_area_raster==0] = (0,0,0)
        cmap[drivable_area_raster==1] = (220,220,220)


        for annotation in CuboidList.from_feather(Path(self.log_specific_dataset_path, 'annotations.feather')):
            if annotation.timestamp_ns == timestamp:
                annotation = annotation.transform(pose)      
                _, is_interior = annotation.compute_interior_points(points_xy_wrt_city)          
                if annotation.category in STATIC:
                    # print("static")
                    # print(interior_points)
                    # print(annotation.dst_SE3_object.translation)
                    # print(annotation.width_m, annotation.length_m, annotation.height_m)
                    # print(np.count_nonzero(is_interior))
                    cmap[is_interior] = (255,127,80)            
                elif annotation.category in DYNAMIC:
                    # print("dynamic")
                    # print(interior_points)
                    # print(annotation.dst_SE3_object.translation)
                    # print(annotation.width_m, annotation.length_m, annotation.height_m)
                    # print(np.count_nonzero(is_interior))
                    cmap[is_interior] = (30,144,255)


        # Convert the Driveable Area Raster + Dynamic + Static Map into a BEV Image
        pdb.set_trace()
        output_bev = xyz_to_bev(
            xyz=points_xy_wrt_city,
            voxel_resolution=(self.bev_info['h'], self.bev_info['w'], 1),
            grid_size_m=(self.bev_info['h_meters'], self.bev_info['w_meters'], 1),
            cmap=cmap
        )
        Image.save(output_bev, f"/srv/share2/apatni30/cvt_labels_argoverse2/{self.log_id}_{random.randint(1, 100)}.png")

        return output_bev

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Additional labels for vehicles only.
        # aux, visibility = self.get_dynamic_objects(sample, anns_vehicle)

        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            bev=self.generate_bev(sample['pose'], sample['token']),
            # aux=aux,
            # visibility=visibility,
            **sample
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
