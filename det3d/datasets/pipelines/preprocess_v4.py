import numpy as np
import torch

from det3d import torchie
from det3d.datasets.kitti import kitti_common as kitti
from det3d.core.evaluation.bbox_overlaps import bbox_overlaps
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import (
    build_dbsampler,
    build_anchor_generator,
    build_similarity_metric,
    build_box_coder,
)
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.anchor.target_assigner import TargetAssigner

from ..registry import PIPELINES
import time


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]



@PIPELINES.register_module
class Preprocess(object):

    def __init__(self, cfg=None, **kwargs):

        self.shuffle_points = cfg.shuffle_points  # True
        self.remove_environment = cfg.remove_environment  # False
        self.remove_unknown = cfg.remove_unknown_examples  # False

        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)  # -1
        self.add_rgb_to_points = cfg.get("add_rgb_to_points", False)  # False
        self.reference_detections = cfg.get("reference_detections", None)  # False
        self.remove_outside_points = cfg.get("remove_outside_points", False)  # False
        self.random_crop = cfg.get("random_crop", False)  # False

        self.mode = cfg.mode  # train or val
        if self.mode == "train":
            self.gt_loc_noise_std = cfg.gt_loc_noise  # [1.0, 1.0, 0.5],
            self.gt_rotation_noise = cfg.gt_rot_noise  # [-0.785, 0.785],
            self.global_rotation_noise = cfg.global_rot_noise  # [-0.785, 0.785]
            self.global_scaling_noise = cfg.global_scale_noise  # [0.95, 1.05]
            self.global_random_rot_range = cfg.global_rot_per_obj_range  # [0, 0]
            self.global_translate_noise_std = cfg.global_trans_noise  # [0.0, 0.0, 0.0]
            self.gt_points_drop = (cfg.gt_drop_percentage,)  # 0.0
            self.remove_points_after_sample = cfg.remove_points_after_sample  # True
            self.class_names = cfg.class_names  # 'Car'
            self.enable_similar_type = cfg.get("enable_similar_type", False)
            if self.enable_similar_type and 'Car' in self.class_names:
                self.class_names.append('Van')
            self.db_sampler = build_dbsampler(cfg.db_sampler)  # GT-AUG

            self.npoints = cfg.get("npoints", -1)  # -1
            self.random_select = cfg.get("random_select", False)  # False
            self.data_aug_with_context = cfg.get("data_aug_with_context", -1)
            self.data_aug_random_drop = cfg.get("data_aug_random_drop", -1)

        self.symmetry_intensity = cfg.get("symmetry_intensity", False)  # False

    def __call__(self, res, info):
        # get points
        res["mode"] = self.mode
        if res["type"] in ["KittiDataset"]:
            points = res["lidar"]["points"]

        # get gt_boxes (x,y,z(velo), w, l, h, ry), gt_names and difficulty levels
        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]
            gt_dict = {"gt_boxes": anno_dict["boxes"], "gt_names": np.array(anno_dict["names"]).reshape(-1), }

            if "difficulty" not in anno_dict:  # True, todo: we may try to make it act.
                difficulty = np.zeros([anno_dict["boxes"].shape[0]], dtype=np.int32)  # todo: all set as 0
                gt_dict["difficulty"] = difficulty
            else:
                gt_dict["difficulty"] = anno_dict["difficulty"]

        # get calib
        calib = res["calib"] if "calib" in res else None

        if self.mode == "train":
            selected = kitti.drop_arrays_by_name(gt_dict["gt_names"],
                                                 ["DontCare", "ignore"])  # todo: where is the definition of ignore???
            _dict_select(gt_dict, selected)

            # False
            if self.remove_unknown:
                remove_mask = gt_dict["difficulty"] == -1
                gt_boxes_remove = gt_dict['gt_boxes'][remove_mask]
                gt_boxes_remove[:, 3:6] += 0.25
                masks = box_np_ops.points_in_rbbox(points, gt_boxes_remove)
                points = points[np.logical_not(masks.any(-1))]
                keep_mask = np.logical_not(remove_mask)
                _dict_select(gt_dict, keep_mask)

            # discard
            gt_dict.pop("difficulty")

            # False, todo: remove those gt_boxes with too little points
            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(points, gt_dict["gt_boxes"])
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)

            # remove untargeted category objects; todo: what about the similar types
            gt_boxes_mask = np.array([n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_)

            # perform gt-augmentation
            if self.db_sampler:  # GT-AUG: filter_by_min_num_points, filter_by_difficulty
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    self.random_crop,  # False
                    gt_group_ids=None,
                    calib=calib,
                    targeted_class_names=self.class_names,
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]  # all 1.

                    gt_dict["gt_names"] = np.concatenate([gt_dict["gt_names"], sampled_gt_names], axis=0)
                    gt_dict["gt_boxes"] = np.concatenate([gt_dict["gt_boxes"], sampled_gt_boxes])
                    gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)

                    # True, remove points in original scene with location occupied by auged gt boxes.
                    if self.remove_points_after_sample:
                        # points_in_rbbox is much faster than roipool3d.pts_in_boxes3d_velo_cpu
                        masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                        points = points[np.logical_not(masks.any(-1))]

                    points = np.concatenate([sampled_points, points],
                                            axis=0)  # concat existed points and points in gt-aug boxes

            # per-object augmentation
            prep.noise_per_object_v4_(
                gt_dict["gt_boxes"],  # x,y,z(lidar), w, l, h, ry(cam)
                points,
                gt_boxes_mask,
                rotation_perturb=self.gt_rotation_noise,  # rotation noise
                center_noise_std=self.gt_loc_noise_std,  # translation noise
                global_random_rot_range=self.global_random_rot_range,  # no need here due to prep.global_rotation later.
                group_ids=None,
                num_try=100,
                data_aug_with_context=self.data_aug_with_context,
                data_aug_random_drop=self.data_aug_random_drop,
            )

            _dict_select(gt_dict, gt_boxes_mask)  # get gt_boxes of specific class

            gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32, )
            gt_dict["gt_classes"] = gt_classes

            # global augmentation
            gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)
            # gt_dict["gt_boxes"], points = prep.global_translate_(gt_dict["gt_boxes"], points, self.global_translate_noise_std)
            gt_dict["gt_boxes"], points = prep.global_rotation(gt_dict["gt_boxes"], points,
                                                               rotation=self.global_rotation_noise)
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(gt_dict["gt_boxes"], points,
                                                                 *self.global_scaling_noise)

        if self.shuffle_points:
            # shuffle is a little slow.
            # np.random.shuffle(points)
            choice = np.random.choice(np.arange(points.shape[0]), points.shape[0], replace=False)
            points = points[choice]

        # points sampling
        if self.mode == "train" and self.random_select:  # False
            if self.npoints < points.shape[0]:
                pts_depth = points[:, 2]  # should be points[:, 0] (x-axis in velo coord) here
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = (np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) if len(
                    far_idxs_choice) > 0 else near_idxs_choice)
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                if self.npoints > len(points):
                    extra_choice = np.random.choice(choice, self.npoints - len(points), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            points = points[choice]

        # False, uniformize intensity
        if self.symmetry_intensity:
            points[:, -1] -= 0.5  # translate intensity to [-0.5, 0.5]
            # points[:, -1] *= 2

        res["lidar"]["points"] = points
        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):

    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range  # [0, -40.0, -3.0, 70.4, 40.0, 1.0]
        self.voxel_size = cfg.voxel_size  # [0.05, 0.05, 0.1]
        self.max_points_in_voxel = cfg.max_points_in_voxel  # 5
        self.max_voxel_num = cfg.max_voxel_num  # 20000
        self.far_points_first = cfg.far_points_first
        self.shuffle = False

        self.voxel_generator = VoxelGenerator(
            point_cloud_range=self.range,
            voxel_size=self.voxel_size,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size  # [0.05, 0.05, 0.1 ]
        pc_range = self.voxel_generator.point_cloud_range  # [0, -40, -3, 70.4, 40, 1]
        grid_size = self.voxel_generator.grid_size  # [1408, 1600, 40]

        # remove those gt_boxes (after gt-aug, per-object-aug, and global-aug) out of valid bv_range
        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]  # [  0. , -40. ,  70.4,  40. ],
            # todo: try get_valid_mask_by_pc_valid_range in preprocess.py, how about z-axis
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)
            res["lidar"]["annotations"] = gt_dict
            self.shuffle = True

        points = res["lidar"]["points"]
        if self.far_points_first:
            points = box_np_ops.far_points_first(points, 40.0, self.max_voxel_num, self.shuffle)

        voxels, coordinates, num_points_per_voxel = self.voxel_generator.generate(points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        # pack voxelization result
        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points_per_voxel,
            num_voxels=num_voxels,
            shape=grid_size,
        )
        return res, info


@PIPELINES.register_module
class AssignTarget(object):
    '''
        This func was modified for processing only one class
    '''

    def __init__(self, **kwargs):
        # get target assigner & box_coder configs.
        assigner_cfg = kwargs["cfg"]
        target_assigner_config = assigner_cfg.target_assigner
        self.tasks = target_assigner_config.tasks  # num_class=1, class_names=["Car"]
        box_coder_cfg = assigner_cfg.box_coder  # "ground_box3d_coder"

        # get anchor_generator
        anchor_cfg = target_assigner_config.anchor_generators  # anchor_generator_range
        anchor_generators = []
        for a_cfg in anchor_cfg:
            anchor_generator = build_anchor_generator(a_cfg)
            anchor_generators.append(anchor_generator)

        # my addition
        self.target_class_names = [anchor_generator.class_name for anchor_generator in anchor_generators]
        self.target_class_ids = [1]  # for car id
        self.enable_similar_type = assigner_cfg.get("enable_similar_type", False)
        if self.enable_similar_type:
            self.target_class_ids = [1, 2]  # for car id  # todo: addition of similar type

        # get target_assigner
        target_assigners = []
        similarity_calc = build_similarity_metric(target_assigner_config.region_similarity_calculator)  # nearest iou
        positive_fraction = target_assigner_config.sample_positive_fraction  # -1
        if positive_fraction < 0:
            positive_fraction = None
        flag = 0
        for task in self.tasks:  # { num_class=1, class_names=["Car"] }
            target_assigner = TargetAssigner(
                box_coder=build_box_coder(box_coder_cfg),  # "ground_box3d_coder"
                anchor_generators=anchor_generators[flag: flag + task.num_class],
                region_similarity_calculator=similarity_calc,  # nearest iou
                positive_fraction=positive_fraction,  # None
                sample_size=target_assigner_config.sample_size,  # 512
            )
            flag += task.num_class  # 1
            target_assigners.append(target_assigner)

        # results
        self.target_assigners = target_assigners
        self.out_size_factor = assigner_cfg.out_size_factor  # 8
        feature_map_size = [1, 200, 176]
        self.anchor_dicts_by_task = [assigner.generate_anchors_dict(feature_map_size) for assigner in
                                     self.target_assigners]

    def __call__(self, res, info):
        targets = {}

        # Calculate output feature map size for anchor generation.
        grid_size = res["lidar"]["voxels"]["shape"]  # [1408, 1600,  40]

        # get anchors: [x, y, z, w(x-axis), l(y-axis), h, ry].  [(70400, 7),]
        targets["anchors"] = [anchor_dict[self.target_class_names[i]]["anchors"].reshape([-1, 7]) for i, anchor_dict in
                              enumerate(self.anchor_dicts_by_task)]

        # get gt labels of targeted classes; limit ry range in [-pi, pi].
        if res["mode"] == "train":

            gt_dict = res["lidar"]["annotations"]

            gt_mask = np.zeros(gt_dict["gt_classes"].shape, dtype=np.bool)
            for target_class_id in self.target_class_ids:
                gt_mask = np.logical_or(gt_mask, gt_dict["gt_classes"] == target_class_id)

            gt_boxes = gt_dict["gt_boxes"][gt_mask]
            gt_boxes[:, -1] = box_np_ops.limit_period(gt_boxes[:, -1], offset=0.5,
                                                      period=np.pi * 2)  # limit ry to [-pi, pi]

            gt_dict["gt_boxes"] = [gt_boxes]
            gt_dict["gt_classes"] = [gt_dict["gt_classes"][gt_mask]]
            gt_dict["gt_names"] = [gt_dict["gt_names"][gt_mask]]

            res["lidar"]["annotations"] = gt_dict

        # get anchor classification labels and localization regression labels
        if res["mode"] == "train":
            targets_dict = {}
            for idx, target_assigner in enumerate(self.target_assigners):
                targets_dict = target_assigner.assign_v2(
                    self.anchor_dicts_by_task[idx],
                    gt_dict["gt_boxes"][idx],  # (x, y, z, w, l, h, r)
                    anchors_mask=None,
                    gt_classes=gt_dict["gt_classes"][idx],
                    gt_names=gt_dict["gt_names"][idx],
                    enable_similar_type=self.enable_similar_type,
                )

            targets.update({
                "labels": [targets_dict["labels"]],
                "reg_targets": [targets_dict["bbox_targets"]],
                "reg_weights": [targets_dict["bbox_outside_weights"]],
                "positive_gt_id": [targets_dict["positive_gt_id"]],
            })

        res["lidar"]["targets"] = targets
        return res, info
