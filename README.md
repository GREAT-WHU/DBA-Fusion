# DBA-Fusion

>Tightly Integrating Deep Dense Visual Bundle Adjustment with Multiple Sensors for Large-Scale  Localization and Mapping


<div align=center>
<img alt="" src="./assets/abstract.png" width='500px' />
</div>


[Paper]

## What is this? 

**DBA-Fusion** is basically a VIO system which fuses DROID-SLAM-like dense bundle adjustment (DBA) with classic factor graph optimization. This work enables **online metric-scale localization and dense mapping** with excellent accuracy and robustness. Besides, this framework supports the **flexible fusion of multiple sensors** like GNSS or wheel speed sensors, extending its applicability to large-scale scenarios.  
<br />
<div align=center>
<img alt="" src="./assets/Hv.svg" width='400px' />
</div>
<br />
<div align=center>
<img alt="" src="./assets/0005.gif" width='500px' />
</div>
<div align=center>
<img alt="" src="./assets/outdoors6.gif" width='500px' />
</div>

## Update log
- [x] Code Upload (2024.2.28)
- [x] Monocular VIO Examples (2024.2.28)
- [ ] Multi-Sensor Fusion Examples
- [ ] Stereo/RGB-D VIO Support

## Installation
The pipeline of the work is based on python, and the computation part is mainly based on Pytorch (with CUDA) and GTSAM.

Use the following command to set up the python environment.

```Bash
conda env create -f environment.yaml
pip install evo --upgrade --no-binary evo
pip install gdown
```

As for GTSAM, we make some modifications to it to extend the python wrapper APIs, clone it from the following repository and install it under your python environment.

```Bash
git clone https://github.com/ZhouTangtang/gtsam.git
cd gtsam
mkdir build
cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.10.11
make python-install
```


## Run DBA-Fusion
We don't modify the model of DROID-SLAM so you can directly employ the  weight trained for DROID-SLAM. Here we use the [model](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing) pre-trained on TartanAir (provided by [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM?tab=readme-ov-file)), which shows great zero-shot performance on real-world datasets.

# 1. TUM-VI
1.1 Download the [TUM-VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset) datasets (512*512).

**(Optional)**
For better speed performance, it is recommended to convert the PNG images to a single HDF5 file through
```Bash
python dataset/prepare_tumvi.py --imagedir=${DATASET_DIR}/dataset-${SEQ}_512_16/mav0/cam0/data --imagestamp=${DATASET_DIR}/dataset-${SEQ}_512_16/mav0/cam0/data.csv --h5path=${SEQ}.h5 --calib=calib/tumvi.txt --stride 4
```

1.2  Specify the data path in [batch_tumvi.py](../batch_tumvi.py) (if you use HDF5 file, activate the "--enable_h5" and "--h5_path" arguments), run the following command 

```Bash
python batch_tumvi.py  # This would trigger demo_vio_tumvi.py automatically.
```

Look into [demo_vio_tumvi.py](../demo_vio_tumvi.py) to learn about the arguments. Data loading and almost all the parameters are specified in this **one** file.

1.3 The outputs of the program includes **a text file** which contains real-time navigation results and **a .pkl file** which contains all keyframe poses and point clouds.

To evaluate the realtime pose estimation performance, run the following command (notice to change the file paths in *evaluate_kitti.py*)

```Bash
python evaluation_scripts/evaluate_tumvi.py --seq ${SEQ}
```
or 
```Bash
python evaluation_scripts/batch_evaluate_tumvi.py
```


For 3D visualization, currently we haven't handled the realtime visualization functionality. Run the scripts in the **"visualization"** folder for post-time visualization. 

```Bash
python visualization/visualize_tumvi.py
```

# 2. KITTI-360
2.1 Download the [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php) datasets. Notice that we use the **unrectified perspective images** for the evaluation (named like "2013_05_28_drive_XXXX_sync/image_00/data_rgb").



For **IMU** data and IMU-centered **ground-truth poses**, we transform the axises to **Right-Forward-Up (RFU)** and check the synchronization. Besides, we use [OpenVINS](https://github.com/rpng/open_vins/) (in stereo VIO mode) to online refine the Camera-IMU extrinsics and time offsets (whose pre-calibrated values seem not very accurate) on the sequences. The refined parameters are used for for all the tests.  

**To reproduce the results**, just download the our prepared IMU and ground-truth data from [here](https://drive.google.com/file/d/1BO8zGvoey7IdwbWXmAdlhGPr6hiCFJ6Y/view?usp=drive_link), then uncompress it to the data path.

**(Optional)**
Similar to the TUM-VI part, you can use the following script to generate a HDF5 file for best speed performance.

```Bash
python dataset/prepare_tumvi.py --imagedir=${DATASET_DIR}/dataset-${SEQ}_512_16/mav0/cam0/data --imagestamp=${DATASET_DIR}/dataset-${SEQ}_512_16/mav0/cam0/data.csv --h5path=${SEQ}.h5 --calib=calib/tumvi.txt --stride 4
```

2.2 Run the following command

```Bash
python batch_kitti360.py
```
Dataloading and parameters are specified in [demo_vio_kitti360.py](../demo_vio_kitti360.py).

2.3 For evaluation and visualization, run
```Bash
python evaluation_scripts/evaluate_kitti360.py --seq ${SEQ}
python visualization/visualize_tumvi.py
```

# 3. Run on Your Own Dataset
To run monocular VIO on your own dataset,
* Duplicate a script from [demo_vio_kitti360.py](../demo_vio_kitti360.py) or [demo_vio_tumvi.py](../demo_vio_tumvi.py). 
* In the script, specify the data loading procedure of IMU data and images.
* Specify the camera intrinsics and camera-IMU extrinsics in the script. 
* Try it!


## Acknowledgement
DBA-Fusion is developed by [GREAT](http://igmas.users.sgg.whu.edu.cn/group) (GNSS+ REsearch, Application and Teaching) Group, School of Geodesy and Geomatics, Wuhan University. 

<br/>
<div align=center>
<img alt="" src="./assets/GREAT.png" width='300px' />
</div>
<br/>
<div align=center>
<img alt="" src="./assets/whu.png" width='300px' />
</div>
<br/>

This work is based on [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [GTSAM](https://github.com/borglab/gtsam). We use evaluation tools from [evo](https://github.com/MichaelGrupp/evo) and 3D visualization tools from [Open3d](https://github.com/MichaelGrupp/evo).
