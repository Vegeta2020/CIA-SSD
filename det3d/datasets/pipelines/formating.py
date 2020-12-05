from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES


class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class Reformat(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]
        anchors = res["lidar"]["targets"]["anchors"]

        data_bundle = dict(
            metadata=meta,                     # image_prefix/shape/id/num_points_features;
            points=points,                     # dim:5, padding with batch_id like coors;
            voxels=voxels["voxels"],           # [num_voxels, max_num_points(T/5), point_dim(4)]
            shape=voxels["shape"],             # [[1408, 1600,   40], [1408, 1600,   40]]
            num_points=voxels["num_points"],   # record num_points in each voxel;
            num_voxels=voxels["num_voxels"],   # num_voxels in each sample;
            coordinates=voxels["coordinates"], # coor and batch_id of each voxel;
            anchors=anchors,                   # anchors, only one group
        )

        if res["mode"] == "val":
            data_bundle.update(dict(metadata=meta,))

        calib = res.get("calib", None)
        if calib:
            data_bundle["calib"] = calib

        if res["mode"] != "test":
            annos = res["lidar"]["annotations"]
            data_bundle.update(annos=annos,)

        if res["mode"] == "train":
            ground_plane = res["lidar"].get("ground_plane", None)
            labels = res["lidar"]["targets"]["labels"]
            reg_targets = res["lidar"]["targets"]["reg_targets"]
            reg_weights = res["lidar"]["targets"]["reg_weights"]
            positive_gt_id = dict(positive_gt_id=res["lidar"]["targets"]["positive_gt_id"])

            if ground_plane:
                data_bundle["ground_plane"] = ground_plane

            data_bundle.update(dict(labels=labels, reg_targets=reg_targets, reg_weights=reg_weights, positive_gt_id=positive_gt_id))

        return data_bundle, info


@PIPELINES.register_module
class PointCloudCollect(object):
    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, info):

        results = info["res"]

        data = {}
        img_meta = {}

        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_meta"] = DC(img_meta, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + "(keys={}, meta_keys={})".format(
            self.keys, self.meta_keys
        )
