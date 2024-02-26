import sys
sys.path.append('dbaf')
sys.path.append('dbaf/geoFunc')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import argparse
from dbaf import DBAFusion

import h5py
import pickle
import re
import math
import gtsam
import quaternion

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, imagestamp, enable_h5, h5path, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    if not enable_h5:
        image_list = sorted(os.listdir(imagedir))[::stride]
        image_stamps = np.loadtxt(imagestamp,str)
        image_dict = dict(zip(image_stamps[:,1],image_stamps[:,0]))
        for t, imfile in enumerate(image_list):
            image = cv2.imread(os.path.join(imagedir, imfile))

            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])
            tt = float(image_dict[imfile])

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield tt, image[None], intrinsics
    else:
        ccount = 0
        h5_f = h5py.File(h5path,'r')
        all_keys = sorted(list(h5_f.keys()))
        for key in all_keys:
            ccount += 1
            yield pickle.loads(np.array(h5_f[key]))

if __name__ == '__main__':

    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--imagestamp", type=str, help="")
    parser.add_argument("--imupath", type=str, help="")
    parser.add_argument("--gtpath", type=str, help="")
    parser.add_argument("--enable_h5", action="store_true", help="")
    parser.add_argument("--h5path", type=str, help="")
    parser.add_argument("--resultpath", type=str, default="result.txt", help="")

    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=80)
    parser.add_argument("--image_size", default=[240, 320])

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--active_window", type=int, default=8, help="maximum frames involved in DBA")
    parser.add_argument("--inac_range", type=int, default=3, help="maximum inactive frames (whose flow wouldn't be updated) involved in DBA")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--visual_only", type=int,default=0, help="wheter to disbale the IMU")
    parser.add_argument("--far_threshold", type=float, default=0.02, help="far pixels would be downweighted (unit: m^-1)")
    parser.add_argument("--translation_threshold", type=float, default=0.2, help="avoid the insertion of too close keyframes (unit: m)")
    parser.add_argument("--mask_threshold", type=float, default=-1, help="downweight too close edges (unit: m)")
    parser.add_argument("--skip_edge", type = str, default ="[]", help="whether to add 'skip' edges in the graph (for example, [-4,-5,-6] relative to the oldest active frame)")
    parser.add_argument("--save_pkl", action="store_true")
    parser.add_argument("--pklpath", default="result.pkl", help="path to saved reconstruction")
    parser.add_argument("--show_plot", action="store_true", help="plot the image/trajectory during running")
    
    args = parser.parse_args()
    args.skip_edge = eval(args.skip_edge)

    args.stereo = False
    dbaf = None
    torch.multiprocessing.set_start_method('spawn')

    """ Load reference trajectory (for visualization) """
    all_gt ={}
    fp = open(args.gtpath,'rt')
    while True:
        line = fp.readline().strip()
        if line == '':break
        if line[0] == '#' : continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        sod = float(elem[0])
        if sod not in all_gt.keys():
            all_gt[sod] ={}
        R = quaternion.as_rotation_matrix(quaternion.from_float_array([float(elem[7]),\
                                                                       float(elem[4]),\
                                                                       float(elem[5]),\
                                                                       float(elem[6])]))
        TTT = np.eye(4,4)
        TTT[0:3,0:3] = R
        TTT[0:3,3] = np.array([ float(elem[1]), float(elem[2]), float(elem[3])])
        all_gt[sod]['T'] = TTT
    all_gt_keys =sorted(all_gt.keys())
    fp.close()

    """ Load IMU data """
    all_imu = np.loadtxt(args.imupath)
    all_odo = []
    all_gnss = []
    tstamps = []
    
    """ Load images """
    try:
        for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.imagestamp, args.enable_h5,\
                                                         args.h5path, args.calib, args.stride)):
            if args.show_plot:
                show_image(image[0])
            if dbaf is None:
                args.image_size = [image.shape[2], image.shape[3]]
                dbaf = DBAFusion(args)
                all_imu[:,0] -= 0.04   # IMU-camera time offset
                dbaf.frontend.all_imu = all_imu
                dbaf.frontend.all_gnss = all_gnss
                dbaf.frontend.all_odo = all_odo
                dbaf.frontend.all_stamp  = np.loadtxt(args.imagestamp,str)

                dbaf.frontend.all_stamp = dbaf.frontend.all_stamp[:,0].astype(np.float64)[None].transpose(1,0)
                if len(all_gt) > 0:
                    dbaf.frontend.all_gt = all_gt
                    dbaf.frontend.all_gt_keys = all_gt_keys
                
                # IMU-Camera Extrinsics
                dbaf.video.Ti1c = np.array(
                               [0.99944133,-0.00228419,-0.03334389,-0.03734697,
                                  0.03268308,-0.14183394,0.98935078,1.75837780,
                                  -0.00698916,-0.98988784,-0.14168005,0.59911765,
                                  0.00000000,0.00000000,0.00000000,1.00000000]).reshape([4,4])
                dbaf.video.Tbc = gtsam.Pose3(dbaf.video.Ti1c)

                # IMU parameters
                dbaf.video.state.set_imu_params([ 0.0003924 * 25,0.000205689024915 * 25, 0.004905 * 10, 0.000001454441043 * 500])
                dbaf.video.init_pose_sigma = np.array([1.0, 1.0, 0.0001, 1.0, 1.0, 1.0])
                dbaf.video.init_bias_sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                dbaf.frontend.translation_threshold  = args.translation_threshold
                dbaf.frontend.graph.mask_threshold   = args.mask_threshold

            dbaf.track(t, image, intrinsics=intrinsics)
        dbaf.save_vis_easy()
    except Exception as err:
        print(err)
        dbaf.save_vis_easy()
    dbaf.terminate()
