import os
import subprocess

for i in ['0000','0002','0003','0004','0005','0006','0009','0010']:
    p = subprocess.Popen("python demo_vio_kitti360.py\
     --imagedir=/home/zhouyuxuan/data/2013_05_28_drive_%s_sync/image_00/data_rgb\
     --imagestamp=/home/zhouyuxuan/data/2013_05_28_drive_%s_sync/camstamp.txt\
     --imupath=/home/zhouyuxuan/data/2013_05_28_drive_%s_sync/imu.txt\
     --gtpath=/home/zhouyuxuan/data/2013_05_28_drive_%s_sync/gt_local.txt\
     --enable_h5\
     --h5path=/home/zhouyuxuan/DROID-SLAM/%s.h5\
     --resultpath=vins_temp_%sw.txt\
     --calib=calib/kitti_360.txt\
     --stride=2\
     --active_window=12\
     --frontend_window=5\
     --frontend_radius=2\
     --frontend_nms=1\
     --inac_range=3\
     --visual_only=0\
     --far_threshold=-1\
     --translation_threshold=0.5\
     --mask_threshold=1.0\
     --skip_edge=[-4,-5,-6]\
     --save_pkl\
     --pklpath=%s\
     --show_plot" % (i,i,i,i,i,i,i),shell=True)
    p.wait()
