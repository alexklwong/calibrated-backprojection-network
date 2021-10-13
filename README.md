# Unsupervised Depth Completion with Calibrated Backprojection Layers

PyTorch implementation of *Unsupervised Depth Completion with Calibrated Backprojection Layers*

Published in ICCV 2021 (ORAL)

[[publication]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wong_Unsupervised_Depth_Completion_With_Calibrated_Backprojection_Layers_ICCV_2021_paper.pdf) [[arxiv]](https://arxiv.org/pdf/2108.10531.pdf) [[poster]]() [[talk]]()

Model have been tested on Ubuntu 16.04, 20.04 using Python 3.5, 3.6, 3.7 PyTorch 1.2, 1.3

Authors: [Alex Wong](http://web.cs.ucla.edu/~alexw/)


If this work is useful to you, please cite our paper:
```
@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
```

**Table of Contents**
1. [Setting up](#setting-up)
2. [Downloading pretrained models](#downloading-pretrained-models)
3. [Running KBNet](#running-kbnet)
4. [Training KBNet](#training-kbnet)
5. [Related projects](#related-projects)
6. [License and disclaimer](#license-disclaimer)


## Setting up your virtual environment <a name="setting-up"></a>
We will create a virtual environment with the necessary dependencies
```
virtualenv -p /usr/bin/python3.7 kbnet-py37env
source kbnet-py37env/bin/activate
pip install opencv-python scipy scikit-learn scikit-image matplotlib gdown numpy gast Pillow pyyaml
pip install torch==1.3.0 torchvision==0.4.1 tensorboard==2.3.0
```

## Setting up your datasets
For datasets, we will use [KITTI][kitti_dataset] for outdoors and [VOID][void_github] for indoors. We will also use [NYUv2][nyu_v2_dataset] to demonstrate our generalization capabilities.
```
mkdir data
ln -s /path/to/kitti_raw_data data/
ln -s /path/to/kitti_depth_completion data/
ln -s /path/to/void_release data/
ln -s /path/to/nyu_v2 data/
```

In case you do not already have KITTI and VOID datasets downloaded, we provide download scripts for them:
```
bash bash/setup_dataset_kitti.sh
bash bash/setup_dataset_void.sh
```

The `bash/setup_dataset_void.sh` script downloads the VOID dataset using gdown. However, gdown intermittently fails. As a workaround, you may download them via:
```
https://drive.google.com/open?id=1GGov8MaBKCEcJEXxY8qrh8Ldt2mErtWs
https://drive.google.com/open?id=1c3PxnOE0N8tgkvTgPbnUZXS6ekv7pd80
https://drive.google.com/open?id=14PdJggr2PVJ6uArm9IWlhSHO2y3Q658v
```
which will give you three files `void_150.zip`, `void_500.zip`, `void_1500.zip`.

Assuming you are in the root of the repository, to construct the same dataset structure as the setup script above:
```
mkdir void_release
unzip -o void_150.zip -d void_release/
unzip -o void_500.zip -d void_release/
unzip -o void_1500.zip -d void_release/
bash bash/setup_dataset_void.sh unpack-only
```

For more detailed instructions on downloading and using VOID and obtaining the raw rosbags, you may visit the [VOID][void_github] dataset webpage.

## Downloading our pretrained models <a name="downloading-pretrained-models"></a>
To use our pretrained models trained on KITTI and VOID models, you can download them from Google Drive
```
gdown https://drive.google.com/uc?id=1C2RHo6E_Q8TzXN_h-GjrojJk4FYzQfRT
unzip pretrained_models.zip
```

Note: `gdown` fails intermittently and complains about permission. If that happens, you may also download the models via:
```
https://drive.google.com/file/d/1C2RHo6E_Q8TzXN_h-GjrojJk4FYzQfRT/view?usp=sharing
```

Once you unzip the file, you will find a directory called `pretrained_models` containing the following file structure:
```
pretrained_models
|---- kitti
      |---- kbnet-kitti.pth
      |---- posenet-kitti.pth
|---- void
      |---- kbnet-void1500.pth
      |---- posenet-void1500.pth
```

We also provide our PoseNet model that was trained jointly with our Calibrated Backproject Network (KBNet) so that you may finetune on them without having to relearn pose from scratch.

The pretrained weights should reproduce the numbers we reported in our [paper](https://arxiv.org/pdf/2108.10531.pdf). The table below are the comprehensive numbers:

For KITTI:
| Evaluation set        | MAE    | RMSE    | iMAE  | iRMSE |
| :-------------------- | :----: | :-----: | :---: | :---: |
| Validation            | 260.44 | 1126.85 | 1.03  | 3.20  |
| Testing (online)      | 256.76 | 1069.47 | 1.02  | 2.95  |

For VOID:
| Evaluation set           | MAE    | RMSE    | iMAE  | iRMSE  |
| :----------------------- | :----: | :-----: | :---: | :----: |
| VOID 1500 (0.5% density) | 39.80  | 95.86   | 21.16 | 49.72  |
| VOID 500 (0.15% density) | 77.70  | 172.49  | 38.87 | 85.59  |
| VOID 150 (0.05% density) | 131.54 | 263.54  | 66.84 | 128.29 |
| NYU v2 (generalization)  | 117.18 | 218.67  | 23.01 | 47.96  |

## Running KBNet <a name="running-kbnet"></a>
To run our pretrained model on the KITTI validation set, you may use
```
bash bash/kitti/run_kbnet_kitti_validation.sh
```

Our run scripts will log all of the hyper-parameters used as well as the evaluation scores based on the `output_path` argument. The expected output should be:
```
Evaluation results:
     MAE      RMSE      iMAE     iRMSE
 260.447  1126.855     1.035     3.203
     +/-       +/-       +/-       +/-
  92.735   398.888     0.285     1.915
Total time: 13187.93 ms  Average time per sample: 15.19 ms
```

Our model runs fairly fast, the reported number in the paper is 16ms for KITTI images on an Nvidia 1080Ti GPU. The above is just slightly faster than the reported number.

To run our pretrained model on the KITTI test set, you may use
```
bash bash/kitti/run_kbnet_kitti_testing.sh
```

To get our numbers, you will need to submit the outputs to the KITTI online benchmark.

To run our pretrained model on the VOID 1500 test set of 0.5% density, you may use
```
bash bash/void/run_kbnet_void1500.sh
```

You should expect the output:
```
Evaluation results:
     MAE      RMSE      iMAE     iRMSE
  39.803    95.864    21.161    49.723
     +/-       +/-       +/-       +/-
  27.521    67.776    24.340    62.204
Total time: 10399.33 ms  Average time per sample: 13.00 ms
```

We note that for all of the following experiments, we will use our model trained on denser (VOID 1500) data and test them on various density levels.

Similar to the above, for the VOID 500 (0.15%) test set, you can run:
```
bash bash/void/run_kbnet_void500.sh
```

and the VOID 150 (0.05%) test set:
```
bash bash/void/run_kbnet_void150.sh
```

To use our model trained on VOID and test it on NYU v2:
```
bash bash/void/run_kbnet_nyu_v2.sh
```


## Training KBNet <a name="training-kbnet"></a>
To train KBNet on the KITTI dataset, you may run
```
bash bash/kitti/train_kbnet_vkitti.sh
```

To train KBNet on the VOID dataset, you may run
```
bash bash/void/train_kbnet_void1500.sh
```

Note that while we do not train on VOID 500 or 150 (hence no hyper-parameters are provided), if interested you may modify the training paths to train on VOID 500:
```
--train_image_path training/void/void_train_image_500.txt \
--train_sparse_depth_path training/voidvoid_train_sparse_depth_500.txt \
--train_intrinsics_path training/void/void_train_intrinsics_500.txt \
```

and on VOID 150:
```
--train_image_path training/void/void_train_image_150.txt \
--train_sparse_depth_path training/voidvoid_train_sparse_depth_150.txt \
--train_intrinsics_path training/void/void_train_intrinsics_150.txt \
```

To monitor your training progress, you may use Tensorboard
```
tensorboard --logdir trained_kbnet/kitti/kbnet_model
tensorboard --logdir trained_kbnet/void1500/kbnet_model
```

## Related projects <a name="related-projects"></a>
You may also find the following projects useful:

- [VOICED][voiced_github]: *Unsupervised Depth Completion from Visual Inertial Odometry*. An unsupervised sparse-to-dense depth completion method, developed by the authors. The paper introduces Scaffolding for depth completion and a light-weight network to refine it. This work is published in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [VOID][void_github]: from *Unsupervised Depth Completion from Visual Inertial Odometry*. A dataset, developed by the authors, containing indoor and outdoor scenes with non-trivial 6 degrees of freedom. The dataset is published along with this work in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [XIVO][xivo_github]: The Visual-Inertial Odometry system developed at UCLA Vision Lab. This work is built on top of XIVO. The VOID dataset used by this work also leverages XIVO to obtain sparse points and camera poses.
- [GeoSup][geosup_github]: *Geo-Supervised Visual Depth Prediction*. A single image depth prediction method developed by the authors, published in the Robotics and Automation Letters (RA-L) 2019 and the International Conference on Robotics and Automation (ICRA) 2019. This work was awarded **Best Paper in Robot Vision** at ICRA 2019.
- [AdaReg][adareg_github]: *Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction.* A single image depth prediction method that introduces adaptive regularization. This work was published in the proceedings of Conference on Computer Vision and Pattern Recognition (CVPR) 2019.

We also have works in adversarial attacks on depth estimation methods:
- [Stereopagnosia][stereopagnosia_github]: *Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations.* Adversarial perturbations for stereo depth estimation, published in the Proceedings of AAAI Conference on Artificial Intelligence (AAAI) 2021.
- [Targeted Attacks for Monodepth][targeted_attacks_monodepth_github]: *Targeted Adversarial Perturbations for Monocular Depth Prediction.* Targeted adversarial perturbations attacks for monocular depth estimation, published in the proceedings of Neural Information Processing Systems (NeurIPS) 2020.

[kitti_dataset]: http://www.cvlibs.net/datasets/kitti/
[vkitti_dataset]: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/
[scenenet_dataset]: https://robotvault.bitbucket.io/scenenet-rgbd.html
[void_github]: https://github.com/alexklwong/void-dataset
[voiced_github]: https://github.com/alexklwong/unsupervised-depth-completion-visual-inertial-odometry
[xivo_github]: https://github.com/ucla-vision/xivo
[geosup_github]: https://github.com/feixh/GeoSup
[adareg_github]: https://github.com/alexklwong/adareg-monodispnet
[stereopagnosia_github]: https://github.com/alexklwong/stereopagnosia
[targeted_attacks_monodepth_github]: https://github.com/alexklwong/targeted-adversarial-perturbations-monocular-depth

## License and disclaimer <a name="license-disclaimer"></a>
This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to these [terms and conditions](license). For commercial use, please contact [UCLA TDG](https://tdg.ucla.edu).
