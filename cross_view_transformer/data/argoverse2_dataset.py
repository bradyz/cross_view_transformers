import numpy as np
import pandas as pd
from pathlib import Path
import torch
from typing import List, Tuple, Any, Dict

from .common import get_split, get_view_matrix
from .transforms import Sample, SaveDataTransform

from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.datasets.sensor.constants import RingCameras, AnnotationCategories
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import CuboidList
from av2.utils.io import read_city_SE3_ego, TimestampedCitySE3EgoPoses


CLASS_TO_LAYER_INDEX: Dict[Any,int] = {
    "roi": 0,
    "drivable_area": 1,
    "lane_boundary": 2,
    AnnotationCategories.RAILED_VEHICLE : 3,
    AnnotationCategories.REGULAR_VEHICLE : 3,
    AnnotationCategories.LARGE_VEHICLE : 3,
    
    AnnotationCategories.BOX_TRUCK : 4,
    AnnotationCategories.TRUCK_CAB : 4,
    AnnotationCategories.TRUCK : 4,

    AnnotationCategories.ARTICULATED_BUS : 5,
    AnnotationCategories.BUS : 5,
    AnnotationCategories.SCHOOL_BUS : 5,

    AnnotationCategories.VEHICULAR_TRAILER : 6,
    
    AnnotationCategories.BOLLARD: 7,
    AnnotationCategories.CONSTRUCTION_BARREL : 7,
    AnnotationCategories.CONSTRUCTION_CONE : 7,
    
    AnnotationCategories.PEDESTRIAN : 8,
    AnnotationCategories.OFFICIAL_SIGNALER : 8,
    AnnotationCategories.STROLLER : 8,
    AnnotationCategories.WHEELCHAIR : 8,

    AnnotationCategories.MOTORCYCLIST : 9,
    AnnotationCategories.MOTORCYCLE : 9,
    
    AnnotationCategories.WHEELED_RIDER : 10,
    AnnotationCategories.WHEELED_DEVICE : 10,
    AnnotationCategories.BICYCLE : 10,
    AnnotationCategories.BICYCLIST : 10,    

    # NOT INCLUDED AT THE MOMENT
    # AnnotationCategories.MOBILE_PEDESTRIAN_CROSSING_SIGN : 1,
    # AnnotationCategories.SIGN : 1,
    # AnnotationCategories.STOP_SIGN : 1,
    # AnnotationCategories.TRAFFIC_LIGHT_TRAILER : 1,
    # AnnotationCategories.MESSAGE_BOARD_TRAILER : 1,
    # AnnotationCategories.ANIMAL : 1,
    # AnnotationCategories.DOG : 1,
}
NUM_CLASSES = max(CLASS_TO_LAYER_INDEX.values())-min(CLASS_TO_LAYER_INDEX.values())+1

COLORING = {
    0: (0,0,0),
    1: (105,105,105),
    2: (255,255,255),
    3: (35,13,232),
    4: (157,30,150),
    5: (251,207,34),
    6: (138,89,196),
    7: (255,147,29),
    8: (189,142,0),
    9: (113,182,255),
    10: (133,41,107),
}

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
    assert num_classes == NUM_CLASSES
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)
    transform = SaveDataTransform(labels_dir)

    log_ids = get_split(f"mini_{split}" if version == "mini" else split, "argoverse2")
    
    sensor_dataloader = SensorDataloader(
            dataset_dir=dataset_dir,
            with_annotations=True,
            with_cache=False
        )
    
    for log_id in log_ids:
        assert Path.exists(Path(dataset_dir, split, log_id))

    return [Argoverse2Dataset(
        data_root=dataset_dir,
        log_id=log_id,
        split=split,
        synchronized_timestamps=
            sensor_dataloader.synchronization_cache.loc[(split, log_id, 'lidar')][['lidar']+[cam.value for cam in tuple(RingCameras)]],
        transform=transform,
        ) for log_id in log_ids]


class Argoverse2Dataset(torch.utils.data.Dataset):
    
    def __init__(self,
                data_root: Path,
                log_id: str, 
                split: str,
                synchronized_timestamps: pd.DataFrame,
                transform=None,
                bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
                ):

        print(f'Argoverse2Dataset building: {log_id}')
        self.data_root = data_root
        self.log_id = log_id
        self.scene_name = self.log_id
        self.log_specific_dataset_path = Path(data_root,split,log_id)
        self._avm = None
        self._poses = None
        self._cameras = None
        self._intrinsics = None
        self._extrinsics = None
        
        self.synchronized_timestamps = synchronized_timestamps
        self.transform = transform

        self.view = get_view_matrix(**bev)
        self.bev_info = bev
        self.bev_shape = (bev['h'], bev['w'])

        # 3D Point Cloud around Vehicle
        xpoints = np.linspace(-self.bev_info['h_meters']/2,self.bev_info['h_meters']/2,num=self.bev_info['h']*4)
        ypoints = np.linspace(-self.bev_info['w_meters']/2,self.bev_info['w_meters']/2,num=self.bev_info['w']*4)
        zpoints = np.linspace(-1,4, num=15)
        self.points_xy_wrt_src = np.vstack([np.ravel(grid) for grid in np.meshgrid(xpoints, ypoints, zpoints)]).T

        # for all synchronized scenes get the following data
        self.samples = self.get_samples()


    @property
    def avm(self) -> ArgoverseStaticMap:
        if self._avm == None:
            self._avm = ArgoverseStaticMap.from_map_dir(Path(self.log_specific_dataset_path, 'map'), build_raster=True)
        return self._avm

    @property
    def poses(self) -> TimestampedCitySE3EgoPoses:
        if self._poses == None:
            self._poses = read_city_SE3_ego(self.log_specific_dataset_path)
        return self._poses
    
    @property
    def cameras(self) -> List[PinholeCamera]:
        if self._cameras == None:
            self._cameras = [PinholeCamera.from_feather(self.log_specific_dataset_path, cam.value) for cam in sorted(list(RingCameras), key=lambda x:x.value)]
        return self._cameras

    @property
    def intrinsics(self):
        if self._intrinsics == None:
            self._intrinsics = [cam.intrinsics.K.tolist() for cam in self.cameras]
        return self._intrinsics

    @property
    def extrinsics(self):
        if self._extrinsics == None:
            self._extrinsics = [cam.extrinsics.tolist() for cam in self.cameras]
        return self._extrinsics

    def get_samples(self) -> List[Dict[str, Any]]:
        data = []

        for i in range(len(self.synchronized_timestamps)):
            image_paths = [
                        Path(
                                self.log_specific_dataset_path, 
                                'sensors', 'cameras', camera.cam_name, 
                                f'{self.synchronized_timestamps.iloc[i][camera.cam_name].value}.jpg'
                            )
                        for camera in self.cameras
                    ]
            # Ensure all these paths exist (edge case where some timestamps are negative?)
            if not all([path.exists() for path in image_paths]):
                continue

            data.append(
                {
                'scene': self.log_id,
                'token': self.synchronized_timestamps.iloc[i]['lidar'].value,
                'pose': self.poses[self.synchronized_timestamps.iloc[i]['lidar'].value].transform_matrix.tolist(),
                # 'pose_inverse': self.poses[self.synchronized_timestamps.iloc[i]['lidar'].value].inverse(),
                'cam_ids': list(range(0,len(self.cameras))),
                'cam_channels': [cam.cam_name for cam in self.cameras],
                'intrinsics': self.intrinsics,
                'extrinsics': self.extrinsics,
                'images': [str(path.relative_to(self.data_root)) for path in image_paths] # these are image paths
                }
            )
        return data

    def project_3d_point_values_onto_grid(
        self,
        points: np.ndarray, # (N,3)
        mask: np.ndarray, # (N,)
        values: np.ndarray, # (N,v)
        grid: np.ndarray    # (200, 200, v)
        ) -> np.ndarray: # grid
        assert points.shape[-1] == 3
        assert points.shape[0] == values.shape[0]
        assert points.shape[0] == mask.shape[0]

        x_points = points[:, 0].astype(np.float32)
        y_points = points[:, 1].astype(np.float32)

        min_x = np.min(x_points)
        min_y = np.min(y_points)
        max_x = np.max(x_points)
        max_y = np.max(y_points)

        # Ensures that the grid is equally wide on each side
        # This is not particularly necessary, but it's a good 
        # check for this specific use case
        assert abs(min_x) == abs(max_x) 
        assert abs(min_y) == abs(max_y)
        
        # Left/Upper Adjust all the points to be in the image coordinate from
        x_points -= min_x
        y_points -= min_y
        
        # Scale the point cloud onto the image plane 
        res_x = (max_x - min_x)/(grid.shape[1]-1)
        res_y = (max_y - min_y)/(grid.shape[0]-1)

        x_img = (-y_points / res_y).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-x_points / res_x).astype(np.int32)  # y axis is -x in LIDAR

        # Shift Pixels to upper left corner (after division)
        # floor & ceil used to prevent anything being rounded to below 0 after shift
        x_img -= int(np.floor(np.min(x_img)))
        y_img -= int(np.ceil(np.min(y_img)))
        
        grid[y_img[mask], x_img[mask]] = values[mask]

        return grid

    def point_cloud_to_bev(
        self,
        points: np.ndarray,
        layers_masks: np.ndarray,
        coloring: Dict[int, Tuple],
        img_size: Tuple[int, int]
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a 2D BEV image from the point cloud.
        
        Note that the order of layers determines which color is on top (last on top)

        Args:
            points (np.ndarray): (N,3)
            layers_masks (np.ndarray): (L,N) each layer corresponds to a class, where each pixel (n) is a bool
            coloring (Dict[int, Tuple]): Each layer's respective color (r,g,b)
            img_size (Tuple): (height, width)

        Returns:
            Tuple[np.ndarray, np.ndarray]: image of size (H,W, 3) AND binary masks for each layer (L, H, W)
        """       
        assert points.shape[-1] == 3
        assert points.shape[0] == layers_masks.shape[-1]
        x_points = points[:, 0].astype(np.float32)
        y_points = points[:, 1].astype(np.float32)

        min_x = np.min(x_points)
        min_y = np.min(y_points)
        max_x = np.max(x_points)
        max_y = np.max(y_points)

        # Ensures that the grid is equally wide on each side
        # This is not particularly necessary, but it's a good 
        # check for this specific use case
        assert abs(min_x) == abs(max_x) 
        assert abs(min_y) == abs(max_y)
        
        # Left/Upper Adjust all the points to be in the image coordinate from
        x_points -= min_x
        y_points -= min_y
        
        # Scale the point cloud onto the image plane 
        res_x = (max_x - min_x)/(img_size[1]-1)
        res_y = (max_y - min_y)/(img_size[0]-1)

        x_img = (-y_points / res_y).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-x_points / res_x).astype(np.int32)  # y axis is -x in LIDAR

        # Shift Pixels to upper left corner (after division)
        # floor & ceil used to prevent anything being rounded to below 0 after shift
        x_img -= int(np.floor(np.min(x_img)))
        y_img -= int(np.ceil(np.min(y_img)))
        
        # Build Image and Binary Semantic Segmentation Masks
        im = np.zeros([img_size[0], img_size[1], 3], dtype=np.uint8)
        binary_semantic_seg = np.zeros((img_size[0], img_size[1], len(coloring)), dtype=np.uint8)

        # Fill in pixels for image/mask
        for i in range(len(coloring)):
            im[y_img[layers_masks[i]==1], x_img[layers_masks[i]==1]] = coloring[i]
            binary_semantic_seg[y_img[layers_masks[i]==1], x_img[layers_masks[i]==1], i] |= 1

        return im, 255*binary_semantic_seg

    def generate_bev(self, pose: SE3, timestamp: int) -> Tuple[np.ndarray, np.ndarray]:
        points_xy_wrt_city = pose.transform_point_cloud(self.points_xy_wrt_src)

        # AUX LABELS
        segmentation = np.zeros((self.bev_info['h'], self.bev_info['w']), dtype=np.uint8)
        center_score = np.zeros((self.bev_info['h'], self.bev_info['w']), dtype=np.float32)
        center_offset = np.zeros((self.bev_info['h'], self.bev_info['w'], 2), dtype=np.float32)
        center_o = np.zeros((self.bev_info['h'], self.bev_info['w'], 2), dtype=np.float32)
        center_h = np.zeros((self.bev_info['h'], self.bev_info['w']), dtype=np.float32)
        center_w = np.zeros((self.bev_info['h'], self.bev_info['w']), dtype=np.float32)

        # TODO: consider removing all non-ROI points (can save on computation) 
        layer_masks = np.zeros((NUM_CLASSES, points_xy_wrt_city.shape[0]))
        
        drivable_area_raster = self.avm.raster_drivable_area_layer.get_raster_values_at_coords(points_xy_wrt_city, 0)
        layer_masks[1,drivable_area_raster==1] = 1

        for annotation in CuboidList.from_feather(Path(self.log_specific_dataset_path, 'annotations.feather')):
            if annotation.timestamp_ns == timestamp and annotation.category in CLASS_TO_LAYER_INDEX:
                _, is_interior = annotation.compute_interior_points(self.points_xy_wrt_src)          
                layer_masks[CLASS_TO_LAYER_INDEX[annotation.category], is_interior] = 1

                # Auxiliary Labels for Dynamic Vehicles
                if CLASS_TO_LAYER_INDEX[annotation.category] in [3,4,5,6,9]:
                    center = annotation.xyz_center_m
                    front = np.average(annotation.vertices_m[[0,1,2,3],:],0)
                    left = np.average(annotation.vertices_m[[1,2,5,6],:],0)

                    segmentation = self.project_3d_point_values_onto_grid(
                        self.points_xy_wrt_src,
                        is_interior,
                        255*np.ones(shape=(self.points_xy_wrt_src.shape[0],)),
                        segmentation
                    )

                    center_offset_3d = center.T - self.points_xy_wrt_src
                    center_offset_3d = center_offset_3d[:, 0:2]
                    center_offset = self.project_3d_point_values_onto_grid(
                        self.points_xy_wrt_src,
                        is_interior,
                        center_offset_3d,
                        center_offset
                    )

                    sigma = 1
                    center_score = self.project_3d_point_values_onto_grid(
                        self.points_xy_wrt_src,
                        is_interior,
                        np.exp(-(center_offset_3d ** 2).sum(-1) / (sigma ** 2)),
                        center_score
                    )

                    # orientation, h/2, w/2
                    orientation = np.zeros((self.points_xy_wrt_src.shape[0],2))
                    orientation[is_interior] = ((front - center) / (np.linalg.norm(front - center) + 1e-6))[0:2]
                    center_o = self.project_3d_point_values_onto_grid(
                        self.points_xy_wrt_src,
                        is_interior,
                        orientation,
                        center_o
                    )

                    height = np.zeros((self.points_xy_wrt_src.shape[0]))
                    height[is_interior] = np.linalg.norm(front - center) 
                    center_h = self.project_3d_point_values_onto_grid(
                        self.points_xy_wrt_src,
                        is_interior,
                        height,
                        center_h
                    )

                    width = np.zeros((self.points_xy_wrt_src.shape[0]))
                    width[is_interior] = np.linalg.norm(left - center)
                    center_w = self.project_3d_point_values_onto_grid(
                        self.points_xy_wrt_src,
                        is_interior,
                        width,
                        center_w
                    )

        segmentation = np.float32(segmentation[..., None])
        center_score = center_score[..., None]
        center_h = center_h[..., None]
        center_w = center_w[..., None]
        aux = np.concatenate((segmentation, center_score, center_offset, center_o, center_h, center_w), 2)

        image_bev, binary_semantic_masks = self.point_cloud_to_bev(self.points_xy_wrt_src,layer_masks,COLORING,(self.bev_info['h'],self.bev_info['w']))
        return image_bev, binary_semantic_masks, aux

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Additional labels for vehicles only.
        pose = np.asarray(sample['pose'],dtype=np.float32)
               
        _, bev_masks, aux = self.generate_bev(SE3(pose[:3, :3],pose[:3, 3]), sample['token'])
        visibility = np.full((self.bev_info['h'], self.bev_info['w']), 255, dtype=np.uint8)
        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            bev=bev_masks,
            aux=aux,
            visibility=visibility,
            **sample
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
