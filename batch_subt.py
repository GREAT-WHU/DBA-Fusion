import os
import subprocess

for i in [\
    # 'Handheld1_Folder',\
    'Handheld2_Folder',\
      ]:
    p = subprocess.Popen("python demo_vio_subt.py" +\
    " --imagedir=/mnt/e/subt/%s/cam_0"%i +\
    " --imagestamp=/mnt/e/subt/%s/cam_0/timestamps.txt"%i +\
    " --imupath=/mnt/e/subt/%s/imu/imu_data.csv"%i +\
    " --resultpath=results/result_%s.txt"%i +\
    " --calib=calib/subt.txt" +\
    " --stride=8" +\
    " --max_factors=48" +\
    " --active_window=12" +\
    " --frontend_window=5" +\
    " --frontend_radius=2" +\
    " --frontend_nms=1" +\
    " --far_threshold=0.02" +\
    " --inac_range=3" +\
    " --visual_only=0" +\
    " --translation_threshold=0.2" +\
    " --mask_threshold=-1.0" +\
    " --skip_edge=[-4,-5,-6]" +\
    " --save_pkl" +\
    " --pklpath=results/%s.pkl"%i +\
    " --show_plot",shell=True)
    p.wait()
