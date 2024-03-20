import os
import subprocess

# VIO
p = subprocess.Popen("python demo_vio_whu.py" +\
    " --imagedir=/home/zhouyuxuan/data/WUH1012/cam0" +\
    " --imagestamp=/home/zhouyuxuan/data/WUH1012/camstamp.txt" +\
    " --imupath=/home/zhouyuxuan/data/WUH1012/imu.txt" +\
    " --gtpath=/home/zhouyuxuan/data/WUH1012/IE.txt" +\
    " --resultpath=results/result_whu_vio.txt" +\
    " --calib=calib/1012.txt" +\
    " --stride=2" +\
    " --max_factors=48" +\
    " --active_window=12" +\
    " --frontend_window=5" +\
    " --frontend_radius=2" +\
    " --frontend_nms=1" +\
    " --inac_range=3" +\
    " --visual_only=0" +\
    " --far_threshold=-1" +\
    " --translation_threshold=0.25" +\
    " --mask_threshold=0.0" +\
    " --skip_edge=[]" +\
    " --save_pkl" +\
    " --use_zupt" +\
    " --pklpath=results/whu.pkl" +\
    " --show_plot",
 shell=True)
p.wait()

# VIO + wheel
p = subprocess.Popen("python demo_vio_whu.py" +\
    " --imagedir=/home/zhouyuxuan/data/WUH1012/cam0" +\
    " --imagestamp=/home/zhouyuxuan/data/WUH1012/camstamp.txt" +\
    " --imupath=/home/zhouyuxuan/data/WUH1012/imu.txt" +\
    " --gtpath=/home/zhouyuxuan/data/WUH1012/IE.txt" +\
    " --resultpath=results/result_whu_viow.txt" +\
    " --calib=calib/1012.txt" +\
    " --stride=2" +\
    " --max_factors=48" +\
    " --active_window=12" +\
    " --frontend_window=5" +\
    " --frontend_radius=2" +\
    " --frontend_nms=1" +\
    " --inac_range=3" +\
    " --visual_only=0" +\
    " --far_threshold=-1" +\
    " --translation_threshold=0.25" +\
    " --mask_threshold=0.0" +\
    " --skip_edge=[]" +\
    " --save_pkl" +\
    " --use_odo" +\
    " --odopath=/home/zhouyuxuan/data/WUH1012/odo_synthesis.txt" +\
    " --pklpath=results/whu.pkl" +\
    " --show_plot",
 shell=True)
p.wait()

# VIO + GNSS
p = subprocess.Popen("python demo_vio_whu.py" +\
        " --imagedir=/home/zhouyuxuan/data/WUH1012/cam0" +\
        " --imagestamp=/home/zhouyuxuan/data/WUH1012/camstamp.txt" +\
        " --imupath=/home/zhouyuxuan/data/WUH1012/imu.txt" +\
        " --gtpath=/home/zhouyuxuan/data/WUH1012/IE.txt" +\
        " --resultpath=results/result_whu_viog.txt" +\
        " --calib=calib/1012.txt" +\
        " --stride=2" +\
        " --max_factors=48" +\
        " --active_window=12" +\
        " --frontend_window=5" +\
        " --frontend_radius=2" +\
        " --frontend_nms=1" +\
        " --inac_range=3" +\
        " --visual_only=0" +\
        " --far_threshold=-1" +\
        " --translation_threshold=0.25" +\
        " --mask_threshold=0.0" +\
        " --skip_edge=[]" +\
        " --save_pkl" +\
        " --use_gnss" +\
        " --gnsspath=/home/zhouyuxuan/data/data_20221012103154/SEPT-PVT.flt" +\
        " --pklpath=results/whu.pkl" +\
        " --show_plot",
 shell=True)
p.wait()
