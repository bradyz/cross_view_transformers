import torch
import numpy as np
import cv2

from .common import resize, to_image
from ..data.nuscenes_dataset import CLASSES


def smooth(x, t1=0.6, c=[52, 101, 154]):
    w = np.float32([255, 255, 255])[None, None]
    c = np.float32(c)[None, None]

    m1 = x > t1

    x_viz = 255 * np.ones(x.shape + (3,), dtype=np.float32)
    x_viz[m1] = c

    opacity = m1.astype(np.float32)

    return x_viz, opacity


class NuScenesStitchViz:
    SEMANTICS = CLASSES

    def __init__(
        self,
        vehicle_threshold=0.5,
        vehicle_color=[52, 101, 154],
        road_threshold=0.5,
        road_color=[196, 199, 192],
        show_images=True,
    ):
        self.vehicle_threshold = vehicle_threshold
        self.vehicle_color = vehicle_color

        self.road_threshold = road_threshold
        self.road_color = road_color

        self.show_images = show_images

    @torch.no_grad()
    def visualize(self, batch, road_pred, vehicle_pred, b_max=8, **kwargs):
        batch_size = road_pred.shape[0]

        for b in range(min(batch_size, b_max)):
            road = road_pred[b].sigmoid().cpu().numpy().transpose(1, 2, 0).squeeze()
            road_viz, _ = smooth(road, self.road_threshold, self.road_color)

            vehicle = vehicle_pred[b].sigmoid().cpu().numpy().transpose(1, 2, 0).squeeze()
            vehicle_viz, vehicle_opacity = smooth(vehicle, self.vehicle_threshold, self.vehicle_color)

            canvas = road_viz
            canvas = vehicle_opacity[..., None] * vehicle_viz + \
                (1 - vehicle_opacity[..., None]) * canvas
            canvas = np.uint8(canvas)

            points = np.array([
                [-4.0 / 2 + 0.3, -1.73 / 2, 1],
                [-4.0 / 2 + 0.3,  1.73 / 2, 1],
                [ 4.0 / 2 + 0.3,  1.73 / 2, 1],
                [ 4.0 / 2 + 0.3, -1.73 / 2, 1],
            ])

            points = batch['view'][0].cpu().numpy() @ points.T

            canvas = canvas.astype(np.uint8)

            cv2.fillPoly(canvas, [points.astype(np.int32)[:2].T], color=(164, 0, 0))

            right = canvas
            image = None if not hasattr(batch.get('image'), 'shape') else batch['image']

            if image is not None and self.show_images:
                imgs = [to_image(image[b][i]) for i in range(image.shape[1])]

                if len(imgs) == 6:
                    a = np.hstack(imgs[:3])
                    b = np.hstack(imgs[3:])
                    left = resize(np.vstack((a, b)), right)
                else:
                    left = np.hstack([resize(x, right) for x in imgs])

                yield np.hstack((left, right))
            else:
                yield right

    def __call__(self, batch, road_pred, vehicle_pred, **kwargs):
        return list(self.visualize(batch, road_pred, vehicle_pred, **kwargs))
