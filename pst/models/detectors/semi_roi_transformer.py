
import numpy as np
import cv2, time, scipy
import torch, torchvision

from mmcv import imdenormalize, imnormalize
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.models import BaseDetector
from mmdet.core.bbox.iou_calculators import build_iou_calculator

from mmrotate.models import ROTATED_DETECTORS, RoITransformer
from mmrotate.core.bbox.transforms import obb2poly_np, poly2obb_np


@ROTATED_DETECTORS.register_module()
class SemiRoITransformer(BaseDetector):
    def __init__(self, version, use_weak_rotate=False, probs=1.0, *args, **kwargs):
        super(SemiRoITransformer, self).__init__()
        semi_cfg = kwargs.pop('semi_cfg')

        self.stu_model = RoITransformer(*args, **kwargs)

        self.tea_model_a = RoITransformer(*args, **kwargs)
        self.freeze(self.tea_model_a)
        self.tea_model_b = RoITransformer(*args, **kwargs)
        self.freeze(self.tea_model_b)

        self.semi_cfg = semi_cfg
        self.interval = 0
        self.num_pseudos = 0
        self.version = version
        self.use_weak_rotate = use_weak_rotate
        self.probs = probs if isinstance(probs, list) else [probs, probs]
        self.iou_calculator = build_iou_calculator(dict(type='RBboxOverlaps2D'))

    def freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad=False

    def imnorm(self, image, img_norm_cfg):
        mean = img_norm_cfg['mean']
        std  = img_norm_cfg['std']
        to_rgb = img_norm_cfg['to_rgb']

        new_image = imnormalize(image, mean, std, to_rgb=to_rgb)
        return new_image

    def denorm(self, image, img_norm_cfg):
        if isinstance(image, torch.Tensor):
            image = image.clone()
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        mean = img_norm_cfg['mean']
        std  = img_norm_cfg['std']
        to_rgb = img_norm_cfg['to_rgb']
        new_image = imdenormalize(image, mean, std, to_bgr=to_rgb)  ### from rgb to bgr when to_rgb is True
        new_image = new_image.astype(dtype=np.uint8)
        return new_image

    def forward_train(self, imgs, img_metas, **kwargs):
        self.interval += 1
        batch_size = int(len(imgs)*2/3)
        label_size = int(len(imgs)/3)

        self.update_teacher_dual()

        start_time = time.perf_counter()
        with torch.no_grad():
            imgs_copy = imgs.clone()
            if self.use_weak_rotate:
                weak_angles = [0] * len(img_metas)
                for img_id in range(len(img_metas)):
                    if img_metas[img_id]['labeled'] or not img_metas[img_id]['weak_aug']:
                        continue
                    img, weak_angle = image_rotate(self.denorm(imgs[img_id], img_metas[img_id]['img_norm_cfg']), angles_range=self.semi_cfg.weak_angles_range)
                    imgs[img_id] = imgs.new(self.imnorm(img, img_metas[img_id]['img_norm_cfg'])).permute(2, 0, 1)
                    weak_angles[img_id] = weak_angle
            tea_results_a = self.tea_model_a.simple_test(imgs, img_metas)
            if self.use_weak_rotate:
                for img_id in range(len(img_metas)):
                    if img_metas[img_id]['labeled'] or not img_metas[img_id]['weak_aug']:
                        continue
                    result_bboxes, result_labels, result_scores = percat2concat(tea_results_a[img_id])
                    result_bboxes, result_labels, result_scores = \
                        bbox_rotate(result_bboxes, result_labels, img_metas[img_id]['img_shape'],
                                    -weak_angles[img_id], self.version, scores=result_scores)
                    tea_results_a[img_id] = concat2percat(self.CLASSES, result_bboxes, result_labels, result_scores)

            imgs = imgs_copy.clone()
            if self.use_weak_rotate:
                weak_angles = [0] * len(img_metas)
                for img_id in range(len(img_metas)):
                    if img_metas[img_id]['labeled'] or not img_metas[img_id]['weak_aug']:
                        continue
                    img, weak_angle = image_rotate(self.denorm(imgs[img_id], img_metas[img_id]['img_norm_cfg']), angles_range=self.semi_cfg.weak_angles_range)
                    imgs[img_id] = imgs.new(self.imnorm(img, img_metas[img_id]['img_norm_cfg'])).permute(2, 0, 1)
                    weak_angles[img_id] = weak_angle
            tea_results_b = self.tea_model_b.simple_test(imgs, img_metas)
            if self.use_weak_rotate:
                for img_id in range(len(img_metas)):
                    if img_metas[img_id]['labeled'] or not img_metas[img_id]['weak_aug']:
                        continue
                    result_bboxes, result_labels, result_scores = percat2concat(tea_results_b[img_id])
                    result_bboxes, result_labels, result_scores = \
                        bbox_rotate(result_bboxes, result_labels, img_metas[img_id]['img_shape'],
                                    -weak_angles[img_id], self.version, scores=result_scores)
                    tea_results_b[img_id] = concat2percat(self.CLASSES, result_bboxes, result_labels, result_scores)
            imgs = imgs_copy            ### NOTE: restore the original imgs

            if self.semi_cfg.use_rescale:
                # rescale_ratio = np.random.uniform(self.semi_cfg.rescale_ratio_range[0], self.semi_cfg.rescale_ratio_range[1])
                rescale_ratio = np.random.uniform(1 / self.semi_cfg.rescale_ratio_range[1], 1 / self.semi_cfg.rescale_ratio_range[0])
                rescale_ratio = 1 / rescale_ratio
                rescale_size = [int(s * rescale_ratio) for s in img_metas[0]['img_shape'][:2]]
                rescale_imgs = torchvision.transforms.functional.resize(imgs, size=rescale_size)

            for img_id in range(len(img_metas)):
                if not img_metas[img_id]['labeled'] and img_metas[img_id]['weak_aug']:
                    pseudo_bboxes, pseudo_labels, pseudo_scores = self.dual_filter(imgs, tea_results_a[img_id], tea_results_b[img_id])

                    pseudo_bboxes = np.array(pseudo_bboxes).reshape(len(pseudo_bboxes), 5)
                    pseudo_labels = np.array(pseudo_labels)
                    kwargs['gt_bboxes'][img_id] = imgs.new(pseudo_bboxes).reshape(len(pseudo_bboxes), 5)
                    kwargs['gt_labels'][img_id] = imgs.new(pseudo_labels).long()

                    pseudo_bboxes, pseudo_labels = self.strong_aug(pseudo_bboxes, pseudo_labels,
                        img_metas[img_id - label_size])
                    kwargs['gt_bboxes'][img_id - label_size] = imgs.new(pseudo_bboxes).reshape(len(pseudo_bboxes), 5)
                    kwargs['gt_labels'][img_id - label_size] = imgs.new(pseudo_labels).long()
                    self.num_pseudos += len(pseudo_bboxes)

        elapsed = time.perf_counter() - start_time
        # print('elapsed: ', elapsed)

        ### NOTE: training the model with only strong augmented data
        imgs = imgs[:batch_size]
        img_metas = img_metas[:batch_size]
        kwargs['gt_bboxes'] = kwargs['gt_bboxes'][:batch_size]
        kwargs['gt_labels'] = kwargs['gt_labels'][:batch_size]

        losses = self.stu_model(imgs, img_metas, **kwargs)

        if self.semi_cfg.use_rescale:
            rescale_imgs = rescale_imgs[:batch_size]
            for img_id in range(len(imgs)):
                kwargs['gt_bboxes'][img_id][:, :-1] = kwargs['gt_bboxes'][img_id][:, :-1] * rescale_ratio
            losses_rescale = self.stu_model(rescale_imgs, img_metas, **kwargs)

            if self.semi_cfg.use_rescale_feat:
                # raise NotImplementedError('consider rotation on data fed into student model')
                # feat_stu = self.stu_model.extract_feat(imgs)
                feat_stu_s = self.stu_model.extract_feat(rescale_imgs)
                loss_distill = []
                small_scale_weight = np.log2(1/rescale_ratio) - int(np.log2(1/rescale_ratio))

                with torch.no_grad():
                    ### NOTE: both imgs and rescale_imgs (rescaled from imgs) have been augmented
                    feat_tea = self.tea_model_a.extract_feat(imgs)
                    feat_tea_s = self.tea_model_b.extract_feat(rescale_imgs)
                for feat_id in range(int(np.log2(1/rescale_ratio)), len(feat_tea)-1):
                    ### NOTE: only consider the adjacent feature closest to the feature of the rescale_imgs
                    rescale_feat_size = [feat_stu_s[feat_id].shape[2], feat_stu_s[feat_id].shape[3]]
                    rescale_feat = \
                        torchvision.transforms.functional.resize(feat_tea[feat_id], rescale_feat_size) * (1 - small_scale_weight) + \
                        torchvision.transforms.functional.resize(feat_tea[feat_id+1], rescale_feat_size) * small_scale_weight

                    feat_sim = torch.nn.functional.cosine_similarity(rescale_feat, feat_tea_s[feat_id]).mean(dim=[1, 2])
                    feat_sim = feat_sim[:, None, None, None]

                    ### NOTE: Huber loss
                    loss_distill.append(
                        (torch.where(torch.abs(rescale_feat - feat_stu_s[feat_id]) <= 1, 
                            torch.nn.functional.mse_loss(
                                    rescale_feat, feat_stu_s[feat_id], reduction='none'),
                            torch.abs(rescale_feat - feat_stu_s[feat_id]) - 0.5) * feat_sim).mean())

                with torch.no_grad():
                    ### NOTE: both imgs and rescale_imgs (rescaled from imgs) have been augmented
                    feat_tea = self.tea_model_b.extract_feat(imgs)
                    feat_tea_s = self.tea_model_a.extract_feat(rescale_imgs)
                for feat_id in range(int(np.log2(1/rescale_ratio)), len(feat_tea)-1):
                    ### NOTE: only consider the adjacent feature closest to the feature of the rescale_imgs
                    rescale_feat_size = [feat_stu_s[feat_id].shape[2], feat_stu_s[feat_id].shape[3]]
                    rescale_feat = \
                        torchvision.transforms.functional.resize(feat_tea[feat_id], rescale_feat_size) * (1 - small_scale_weight) + \
                        torchvision.transforms.functional.resize(feat_tea[feat_id+1], rescale_feat_size) * small_scale_weight

                    feat_sim = torch.nn.functional.cosine_similarity(rescale_feat, feat_tea_s[feat_id]).mean(dim=[1, 2])
                    feat_sim = feat_sim[:, None, None, None]

                    ### NOTE: Huber loss
                    loss_distill.append(
                        (torch.where(torch.abs(rescale_feat - feat_stu_s[feat_id]) <= 1, 
                            torch.nn.functional.mse_loss(
                                    rescale_feat, feat_stu_s[feat_id], reduction='none'),
                            torch.abs(rescale_feat - feat_stu_s[feat_id]) - 0.5) * feat_sim).mean())

                losses['loss_distill'] = [self.semi_cfg.rescale_weight_feat * x for x in loss_distill]

            for key in losses_rescale.keys():
                if 'loss' not in key:
                    losses[key + '_s'] = losses_rescale[key]
                    continue

                if isinstance(losses_rescale[key], list):
                    losses[key + '_s'] = [self.semi_cfg.rescale_weight * x for x in losses_rescale[key]]
                else:
                    losses[key + '_s'] = self.semi_cfg.rescale_weight * losses_rescale[key]

        return losses

    def strong_aug(self, bboxes, labels, img_meta):
        if 'flip' in img_meta and img_meta['flip']:
            bboxes = bbox_flip(bboxes, img_meta['img_shape'], img_meta['flip_direction'], self.version)
        if 'rotate_angle' in img_meta:
            bboxes, labels = bbox_rotate(bboxes, labels, img_meta['img_shape'], img_meta['rotate_angle'], self.version)

        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_meta['img_shape'][1])
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_meta['img_shape'][0])
        return bboxes, labels

    def update_teacher_dual(self):
        # momentums = self.semi_cfg.momentums
        momentums = []

        stu_model = self.stu_model

        ### NOTE: different momentum for different teacher model
        tea_models = []
        if np.random.uniform() < self.probs[0]:
            tea_models.append(self.tea_model_a)
            momentums.append(self.semi_cfg.momentums[0])
        if np.random.uniform() < self.probs[1]:
            tea_models.append(self.tea_model_b)
            momentums.append(self.semi_cfg.momentums[1])
        if len(tea_models) == 0:
            ### NOTE: at least one teacher model should be updated
            if np.random.uniform() < 0.5:
                tea_models.append(self.tea_model_a)
                momentums.append(self.semi_cfg.momentums[0])
            else:
                tea_models.append(self.tea_model_b)
                momentums.append(self.semi_cfg.momentums[1])

        if isinstance(stu_model, MMDistributedDataParallel):
            stu_model = self.stu_model.module

        for tea_model, momentum in zip(tea_models, momentums):
            if isinstance(tea_model, MMDistributedDataParallel):
                tea_model = tea_model.module
            
            for (stu_name, stu_parm), (tea_name, tea_parm) in zip(
                stu_model.named_parameters(), tea_model.named_parameters()
            ):
                tea_parm.data.mul_(momentum).add_(stu_parm.data, alpha=1 - momentum)

    def dual_filter(self, imgs, results_a, results_b):
        pseudo_bboxes = []
        pseudo_labels = []
        pseudo_scores = []

        classes_a = []
        for cls_id in range(len(self.CLASSES)):
            classes_a.extend([cls_id] * len(results_a[cls_id]))
        classes_a = np.array(classes_a)
        results_a = np.concatenate(results_a)
        rbboxes_a = results_a[:, :5]
        scores_a = results_a[:, -1]

        classes_b = []
        for cls_id in range(len(self.CLASSES)):
            classes_b.extend([cls_id] * len(results_b[cls_id]))
        classes_b = np.array(classes_b)
        results_b = np.concatenate(results_b)
        rbboxes_b = results_b[:, :5]
        scores_b = results_b[:, -1]

        if len(rbboxes_a) == 0 or len(rbboxes_b) == 0:
            return np.array(pseudo_bboxes).reshape(-1, 5), np.array(pseudo_labels), np.array(pseudo_scores)

        overlaps = self.iou_calculator(imgs.new(rbboxes_b), imgs.new(rbboxes_a)).cpu()
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        max_overlaps = max_overlaps.numpy()
        argmax_overlaps = argmax_overlaps.numpy()

        fg_inds = max_overlaps > self.semi_cfg.overlap_thr
        pos_inds = classes_a[fg_inds] == classes_b[argmax_overlaps][fg_inds]
        conf_inds = (scores_a[fg_inds][pos_inds] > self.semi_cfg.score_thr) & \
            (scores_b[argmax_overlaps][fg_inds][pos_inds] > self.semi_cfg.score_thr)

        angle_conf, js_loss = JSDiv(imgs,
                    results_a[fg_inds][pos_inds][conf_inds],
                    results_b[argmax_overlaps][fg_inds][pos_inds][conf_inds])
        angle_inds = angle_conf > self.semi_cfg.angle_thr

        pseudo_bboxes = np.where(results_a[fg_inds][pos_inds][conf_inds][angle_inds][:, -1:] > results_b[argmax_overlaps][fg_inds][pos_inds][conf_inds][angle_inds][:, -1:],
            results_a[fg_inds][pos_inds][conf_inds][angle_inds][:, :-1], results_b[argmax_overlaps][fg_inds][pos_inds][conf_inds][angle_inds][:, :-1])
        pseudo_scores = np.where(results_a[fg_inds][pos_inds][conf_inds][angle_inds][:, -1] > results_b[argmax_overlaps][fg_inds][pos_inds][conf_inds][angle_inds][:, -1],
            results_a[fg_inds][pos_inds][conf_inds][angle_inds][:, -1], results_b[argmax_overlaps][fg_inds][pos_inds][conf_inds][angle_inds][:, -1])
        pseudo_labels = classes_a[fg_inds][pos_inds][conf_inds][angle_inds]     ### NOTE: since classes_a[fg_inds][pos_inds][conf_inds] == classes_b[fg_inds][pos_inds][conf_inds]

        return pseudo_bboxes, pseudo_labels, pseudo_scores

    def aug_test(self):
        raise('Not support aug_test')

    def simple_test(self, *args, **kwargs):
        """Test without augmentation."""

        if self.semi_cfg.result_from == 'teacher':
            ### NOTE: select one of the teacher model for inference
            return self.tea_model_a.simple_test(*args, **kwargs)
            # return self.tea_model_b.simple_test(*args, **kwargs)
        elif self.semi_cfg.result_from == 'student':
            return self.stu_model.simple_test(*args, **kwargs)
        else:
            raise ValueError

    def extract_feat(self):
        raise('Not support extract_feat')

    def _parse_losses(self, losses):
        loss, log_vars = super(SemiRoITransformer, self)._parse_losses(losses)

        ### NOTE: interval in log_config of default_runtime is 50
        log_vars['num_pseudos'] = self.num_pseudos * 50 if self.interval % 50 == 0 else 0
        return loss, log_vars


def JSDiv(imgs, results_1, results_2):
    if len(results_1) == 0 and len(results_2) == 0:
        return np.zeros((0)), np.zeros((0))

    from mmrotate.models.losses.gaussian_dist_loss_v1 import xy_wh_r_2_xy_sigma
    gau_1 = xy_wh_r_2_xy_sigma(imgs.new(results_1[:, :-1]))
    gau_2 = xy_wh_r_2_xy_sigma(imgs.new(results_2[:, :-1]))
    js_loss = jsd_loss(gau_1, gau_2).squeeze(-1)

    return 1 - js_loss.cpu().numpy(), js_loss

def jsd_loss(gau_1, gau_2):
    def kl_divergence(pred, target):
        mu_p, sigma_p = pred
        mu_t, sigma_t = target

        mu_p = mu_p.reshape(-1, 2)
        mu_t = mu_t.reshape(-1, 2)
        sigma_p = sigma_p.reshape(-1, 2, 2)
        sigma_t = sigma_t.reshape(-1, 2, 2)

        delta = (mu_p - mu_t).unsqueeze(-1)
        sigma_t_inv = torch.inverse(sigma_t)
        term1 = delta.transpose(-1,
                                -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)
        term2 = torch.diagonal(
            sigma_t_inv.matmul(sigma_p),
            dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) + \
            torch.log(torch.det(sigma_t) / torch.det(sigma_p)).reshape(-1, 1)
        dis = term1 + term2 - 2
        kl_dis = dis.clamp(min=0.0) / 2.
        return kl_dis

    mu_1, sigma_1 = gau_1
    mu_2, sigma_2 = gau_2
    
    sigma_1_square = sigma_1.matmul(sigma_1)
    sigma_2_square = sigma_2.matmul(sigma_2)

    sigma_square = (sigma_1_square + sigma_2_square) / 4.
    mu = (mu_1 + mu_2) / 2.

    sigma = []
    for img_id in range(len(sigma_square)):
        sigma.append(scipy.linalg.fractional_matrix_power(sigma_square.cpu()[img_id], 1/2))
    js1 = kl_divergence(tuple((mu_1, sigma_1)), tuple((mu, mu.new(sigma).unsqueeze(0))))
    js2 = kl_divergence(tuple((mu_2, sigma_2)), tuple((mu, mu.new(sigma).unsqueeze(0))))
    js = (js1 + js2) / 2

    return js

def percat2concat(results):
    results_concat = np.concatenate(results)
    results_bboxes = results_concat[:, :5]
    results_scores = results_concat[:, -1]
    results_labels = np.array([cat_id for cat_id, boxes in enumerate(results) for _ in boxes])
    return results_bboxes, results_labels, results_scores

def concat2percat(CLASSES, results, labels, scores=None):
    if scores is not None:
        results = np.concatenate((results, scores.reshape(-1, 1)), axis=1)
    ret_results = []
    for c in range(len(CLASSES)):
        if scores is not None:
            ret_results.append(np.array(results[labels == c]).reshape(-1, 6))
        else:
            ret_results.append(np.array(results[labels == c]).reshape(-1, 5))
    return ret_results

def bbox_flip(bboxes, img_shape, direction, version):
    def norm_angle(angle, angle_range):

        if angle_range == 'oc':
            return angle
        elif angle_range == 'le135':
            return (angle + np.pi / 4) % np.pi - np.pi / 4
        elif angle_range == 'le90':
            return (angle + np.pi / 2) % np.pi - np.pi / 2
        else:
            print('Not yet implemented.')

    assert bboxes.shape[-1] % 5 == 0
    orig_shape = bboxes.shape
    bboxes = bboxes.reshape((-1, 5))
    flipped = bboxes.copy()
    if direction == 'horizontal':
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
    elif direction == 'vertical':
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    elif direction == 'diagonal':
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
        return flipped.reshape(orig_shape)
    else:
        raise ValueError(f'Invalid flipping direction "{direction}"')
    if version == 'oc':
        rotated_flag = (bboxes[:, 4] != np.pi / 2)
        flipped[rotated_flag, 4] = np.pi / 2 - bboxes[rotated_flag, 4]
        flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
        flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
    else:
        flipped[:, 4] = norm_angle(np.pi - bboxes[:, 4], version)
    return flipped.reshape(orig_shape)

def bbox_rotate(bboxes, labels, img_shape, angle, version, auto_bound=False, scores=None):
    def create_rotation_matrix(center,
                                angle,
                                bound_h,
                                bound_w,
                                offset=0):
        center += offset
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if auto_bound:      ### use global variable of bbox_rotate
            rot_im_center = cv2.transform(center[None, None, :] + offset,
                                        rm)[0, 0, :]
            new_center = np.array([bound_w / 2, bound_h / 2
                                ]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def apply_coords(coords, rm_coords):
        if len(coords) == 0:
            return coords
        # coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], rm_coords)[:, 0, :]

    def filter_border(bboxes, h, w):
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & \
                    (w_bbox > 5) & (h_bbox > 5)
        return keep_inds

    if len(bboxes) == 0:
        if scores is None:
            return bboxes, labels
        else:
            return bboxes, labels, scores
    h, w, c = img_shape
    image_center = np.array((w / 2, h / 2))

    bound_w, bound_h = w, h

    rm_coords = create_rotation_matrix(image_center, angle, bound_h, bound_w)
    # rm_image = create_rotation_matrix(image_center, angle, bound_h, bound_w, offset=-0.5)

    bboxes = np.concatenate([bboxes, np.zeros((bboxes.shape[0], 1))], axis=-1)
    polys = obb2poly_np(bboxes, version)[:, :-1].reshape(-1, 2)
    polys = apply_coords(polys, rm_coords).reshape(-1, 8)

    rotated = []
    for pt in polys:
        pt = np.array(pt, dtype=np.float32)
        obb = poly2obb_np(pt, version) \
            if poly2obb_np(pt, version) is not None\
            else [0, 0, 0, 0, 0]
        rotated.append(obb)
    rotated = np.array(rotated, dtype=np.float32)
    keep_inds = filter_border(rotated, bound_h, bound_w)
    rotated = rotated[keep_inds, :]
    labels = labels[keep_inds]
    if scores is not None:
        scores = scores[keep_inds]
        return rotated, labels, scores

    return rotated, labels

def image_rotate(img, img_shape=None, angle=None, angles_range=None):
    auto_bound = False
    def create_rotation_matrix(center,
                                angle,
                                bound_h,
                                bound_w,
                                offset=0):
        center += offset
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if auto_bound:      ### use global variable of bbox_rotate
            rot_im_center = cv2.transform(center[None, None, :] + offset,
                                        rm)[0, 0, :]
            new_center = np.array([bound_w / 2, bound_h / 2
                                ]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def apply_image(img, bound_h, bound_w, rm_image, interp=cv2.INTER_LINEAR):
        if len(img) == 0:
            return img
        return cv2.warpAffine(
            img, rm_image, (bound_w, bound_h), flags=interp)

    if angle is None:
        if angles_range is None:
            angles_range = 180
        angle = angles_range * (2 * np.random.rand() - 1)
    if img_shape is None:
        h, w, c = img.shape
    else:
        h, w, c = img_shape

    image_center = np.array((w / 2, h / 2))
    bound_w, bound_h = w, h

    rm_image = create_rotation_matrix(
        image_center, angle, bound_h, bound_w, offset=-0.5)

    img = apply_image(img, bound_h, bound_w, rm_image)

    return img, angle

