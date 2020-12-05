## CIA-SSD: Confident IoU-Aware Single Stage Object Detector From Point Cloud (AAAI 2021) [[Paper]](https://github.com/poodarchu/det3d) 

Currently state-of-the-art single-stage object detector from point cloud on KITTI Benchmark, running with high speed of 32FPS.

**Authors**: Wu Zheng, Weiliang Tang, Sijin Chen, Li Jiang, Chi-Wing Fu.

## AP on KITTI Dataset

Val Split (11 recall points):
```
Car  AP:98.97, 90.10, 89.49
bev  AP:90.58, 88.80, 87.88
3d   AP:89.99, 79.88, 78.93
aos  AP:98.90, 89.89, 89.11
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:98.97, 90.10, 89.49
bev  AP:99.03, 90.22, 89.74
3d   AP:99.00, 90.18, 89.68
aos  AP:98.90, 89.89, 89.11
```

Test Split: [Submission link](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=b4e17f75f5baa917c4f250e832aace71682c3a84)

## Pipeline
![pipeline](https://github.com/Vegeta2020/CIA-SSD/tree/master/pictures/pipeline.png)

## Installation

## Citation
If you find this work useful in your research, please consider cite:

## License
Det3D is released under the [Apache licenes](LICENES).

## Acknowledgement
Our code are mainly based on [Det3D](https://github.com/poodarchu/det3d), we thank for their contributions!