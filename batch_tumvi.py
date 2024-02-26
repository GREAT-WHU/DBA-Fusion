import os
import subprocess

for i in [\
    # 'magistrale6',\
    'outdoors6',\
    # 'outdoors7',\
    # 'outdoors1',\
    # 'outdoors8',\
    # 'magistrale1',\
    # 'magistrale2',\
    # 'magistrale3',\
    # 'magistrale4',\
    # 'magistrale5',\
    # 'outdoors2',\
    # 'outdoors3',\
    # 'outdoors4',\
    # 'outdoors5',\
      ]:
    p = subprocess.Popen("python demo_vio_tumvi.py\
     --imagedir=/mnt/z/tum-vi/dataset-%s_512_16/mav0/cam0/data\
     --imagestamp=/mnt/z/tum-vi/dataset-%s_512_16/mav0/cam0/data.csv\
     --imupath=/mnt/z/tum-vi/dataset-%s_512_16/mav0/imu0/data.csv\
     --gtpath=/mnt/z/tum-vi/dataset-%s_512_16/dso/gt_imu.csv\
     --enable_h5\
     --h5path=/home/zhouyuxuan/DROID-SLAM/%s.h5\
     --resultpath=vins_temp_%s_oppoww.txt\
     --calib=calib/tumvi.txt\
     --stride=4\
     --active_window=12\
     --frontend_window=5\
     --frontend_radius=2\
     --frontend_nms=1\
     --far_threshold=0.02\
     --inac_range=3\
     --visual_only=0\
     --translation_threshold=0.2\
     --mask_threshold=-1.0\
     --skip_edge=[-4,-5,-6]\
     --save_pkl\
     --pklpath=%s_oppo\
     --show_plot" % (i,i,i,i,i,i,i),shell=True)
    p.wait()
