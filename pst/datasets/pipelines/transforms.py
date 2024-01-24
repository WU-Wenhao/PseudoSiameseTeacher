
import cv2
import numpy as np
import copy
from torchvision import transforms

import mmcv
from mmcv.parallel import collate
from mmcv.parallel import DataContainer as DC
# from mmdet.datasets.custom import Compose
from mmdet.datasets.pipelines.compose import Compose
from mmrotate.core import norm_angle, obb2poly_np, poly2obb_np
from mmrotate.datasets.builder import ROTATED_PIPELINES

@ROTATED_PIPELINES.register_module(force=True)
class PolyRandomRotate(object):
    """Rotate img & bbox.
    Reference: https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA

    Args:
        rotate_ratio (float, optional): The rotating probability.
            Default: 0.5.
        mode (str, optional) : Indicates whether the angle is chosen in a
            random range (mode='range') or in a preset list of angles
            (mode='value'). Defaults to 'range'.
        angles_range(int|list[int], optional): The range of angles.
            If mode='range', angle_ranges is an int and the angle is chosen
            in (-angles_range, +angles_ranges).
            If mode='value', angles_range is a non-empty list of int and the
            angle is chosen in angles_range.
            Defaults to 180 as default mode is 'range'.
        auto_bound(bool, optional): whether to find the new width and height
            bounds.
        rect_classes (None|list, optional): Specifies classes that needs to
            be rotated by a multiple of 90 degrees.
        version  (str, optional): Angle representations. Defaults to 'le90'.
    """

    def __init__(self,
                 rotate_ratio=0.5,
                 mode='range',
                 angles_range=180,
                 auto_bound=False,
                 rect_classes=None,
                 version='le90'):
        self.rotate_ratio = rotate_ratio
        self.auto_bound = auto_bound
        assert mode in ['range', 'value'], \
            f"mode is supposed to be 'range' or 'value', but got {mode}."
        if mode == 'range':
            assert isinstance(angles_range, int), \
                "mode 'range' expects angle_range to be an int."
        else:
            assert mmcv.is_seq_of(angles_range, int) and len(angles_range), \
                "mode 'value' expects angle_range as a non-empty list of int."
        self.mode = mode
        self.angles_range = angles_range
        self.discrete_range = [90, 180, -90, -180]
        self.rect_classes = rect_classes
        self.version = version

    @property
    def is_rotate(self):
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.rotate_ratio

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(
            img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y)
        points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def create_rotation_matrix(self,
                               center,
                               angle,
                               bound_h,
                               bound_w,
                               offset=0):
        """Create rotation matrix."""
        center += offset
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if self.auto_bound:
            rot_im_center = cv2.transform(center[None, None, :] + offset,
                                          rm)[0, 0, :]
            new_center = np.array([bound_w / 2, bound_h / 2
                                   ]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w):
        """Filter the box whose center point is outside or whose side length is
        less than 5."""
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & \
                    (w_bbox > 5) & (h_bbox > 5)
        return keep_inds

    def __call__(self, results):
        """Call function of PolyRandomRotate."""
        if not self.is_rotate:
            results['rotate'] = False
            angle = 0
        else:
            results['rotate'] = True
            if self.mode == 'range':
                angle = self.angles_range * (2 * np.random.rand() - 1)
            else:
                i = np.random.randint(len(self.angles_range))
                angle = self.angles_range[i]

            class_labels = results['gt_labels']
            for classid in class_labels:
                if self.rect_classes:
                    if classid in self.rect_classes:
                        np.random.shuffle(self.discrete_range)
                        angle = self.discrete_range[0]
                        break

        h, w, c = results['img_shape']
        img = results['img']
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = \
            abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))
        if self.auto_bound:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos,
                 h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle,
                                                     bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)

        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)
        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])
        gt_bboxes = np.concatenate(
            [gt_bboxes, np.zeros((gt_bboxes.shape[0], 1))], axis=-1)
        if len(gt_bboxes) > 0:
            polys = obb2poly_np(gt_bboxes, self.version)[:, :-1].reshape(-1, 2)
            polys = self.apply_coords(polys).reshape(-1, 8)
        else:
            polys = np.zeros((0, 8))
        gt_bboxes = []
        for pt in polys:
            pt = np.array(pt, dtype=np.float32)
            obb = poly2obb_np(pt, self.version) \
                if poly2obb_np(pt, self.version) is not None\
                else [0, 0, 0, 0, 0]
            gt_bboxes.append(obb)
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 5)
        keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
        gt_bboxes = gt_bboxes[keep_inds, :]
        labels = labels[keep_inds]
        # if len(gt_bboxes) == 0:
        #     return None
        ### NOTE: Attention: the original code is return None when len(gt_bboxes) == 0.
        ### NOTE: But I comment the "return" for two reason:
        ### NOTE: 1. for labeled data, the labeled data can still feed the background information even there is no gt_bboxes;
        ### NOTE: 2. for unlabeled data, the unlabeled data should not been given any anno prior information when filtering at the augmentation stage
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = labels

        return results

@ROTATED_PIPELINES.register_module()
class RandomColorJitter:
    def __init__(self, jitter=[0.4, 0.4, 0.4, 0.1], p=0.5):
        self.jitter = transforms.ColorJitter(*jitter)
        self.toPIL = transforms.ToPILImage()
        self.p = p

    def __call__(self, results):
        if self.p < np.random.random():
            return results
        img = results['img']
        img = self.toPIL(img)
        img = self.jitter(img)
        img = np.asarray(img)
        results['img'] = img

        return results

@ROTATED_PIPELINES.register_module()
class RandomGrayscale:
    def __init__(self, p=0.5):
        # self.p = p
        self.scaler = transforms.RandomGrayscale(p=p)
        self.toPIL = transforms.ToPILImage()

    def __call__(self, results):
        img = results['img']
        img = self.toPIL(img)
        img = self.scaler(img)
        img = np.asarray(img)
        results['img'] = img
        
        return results

from PIL import ImageFilter
@ROTATED_PIPELINES.register_module()
class RandomGaussianBlur:
    def __init__(self, rad_range=[0.1, 2.0], p=0.5):
        self.rad_range = rad_range
        self.toPIL = transforms.ToPILImage()
        self.p = p

    def __call__(self, results):
        if self.p < np.random.random():
            return results
        rad = np.random.uniform(*self.rad_range)
        img = results['img']
        img = self.toPIL(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=rad))
        img = np.asarray(img)
        results['img'] = img

        return results

@ROTATED_PIPELINES.register_module()
class LabelState:
    def __init__(self):
        pass

    def __call__(self, results):
        if 'labeled' in results['img_info']['ann']:
            results['labeled'] = results['img_info']['ann']['labeled']
        else:
            results['labeled'] = True

        return results

@ROTATED_PIPELINES.register_module()
class SemiAugmentation:
    def __init__(self, strong_pipeline, weak_pipeline):
        self.strong_transforms = Compose(strong_pipeline)
        self.weak_transforms = Compose(weak_pipeline)

    def __call__(self, results):
        results_copy = copy.deepcopy(results)

        results['weak_aug'] = False
        results = self.strong_transforms(results)
        # if results is None:
        #     # NOTE: avoid special case in PolyRandomRotate, RRandomCrop, RMosaic
        #     # NOTE: transform will repeat until existing valid gt-bbox, handled by mmdet/datasets/pipelines/compose.py.
        #     return None
        assert results is not None, "the reason why the results should not be None can be seen in 'PolyRandomRotate' function."

        results_copy['weak_aug'] = True
        results_copy = self.weak_transforms(results_copy)

        # for k, v in results.items():
        #     # results[k] = np.concatenate(results[k], results_copy)
        #     print('key: {%s}'%k)
        #     print('type: {%s}'%type(v))
        #     # print('type in DG: {%s}'%type(v.data))

        results = collate([results, results_copy], samples_per_gpu=2)

        return results
