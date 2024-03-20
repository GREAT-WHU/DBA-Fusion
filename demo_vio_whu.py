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
import geoFunc.trans as trans

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
        image_stamps = np.loadtxt(imagestamp,str,delimiter=',')
        image_dict = dict(zip(image_stamps[:,1],image_stamps[:,0]))
        image_list = list(image_dict)
        ccount = 0
        for t, imfile in enumerate(image_list):
            tt = float(image_dict[imfile])
            if int(tt*10)%2 == 1: continue

            ccount += 1

            image = cv2.imread(os.path.join(imagedir, imfile))

            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])

            # image = image[192:704,:,:]
            image = image[192:576,:,:]
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy - 192])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield tt, image[None], intrinsics
    else:
        raise Exception()


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
    parser.add_argument("--resultpath", type=str, help="")

    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=80)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--max_factors", type=int, default=48, help="maximum active edges (which determines the GPU memory usage)")
    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=0.00, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=3.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--active_window", type=int, default=8)
    parser.add_argument("--inac_range", type=int, default=3)
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
    parser.add_argument("--use_gnss", action="store_true")
    parser.add_argument("--gnsspath", type=str, help="")
    parser.add_argument("--use_odo",  action="store_true")
    parser.add_argument("--odopath", type=str, help="")
    parser.add_argument("--use_zupt",  action="store_true")
    args = parser.parse_args()
    args.skip_edge = eval(args.skip_edge)

    args.stereo = False
    dbaf = None


    all_gt ={}
    Ti0i1 =np.array([[ 9.99902524e-01,  1.39619889e-02, -7.31054713e-05,  1.00000000e-02],
                     [-1.39621803e-02,  9.99888818e-01, -5.23545345e-03, -2.05000000e-01],
                     [ 0.00000000e+00,  5.23596383e-03,  9.99986292e-01, -5.00000000e-02],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    Ti0i1 = np.eye(4,4)
    Ten0 = None
    is_ref_set  = False
    fp = open(args.gtpath,'rt')
    while True:
        line = fp.readline().strip()
        if line == '':break
        if line[0] == '#' :continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        sod = float(elem[1])
        if sod not in all_gt.keys():
            all_gt[sod] ={}
        all_gt[sod]['X0']   = float(elem[2])
        all_gt[sod]['Y0']   = float(elem[3])
        all_gt[sod]['Z0']   = float(elem[4])
        all_gt[sod]['VX0']  = float(elem[15])
        all_gt[sod]['VY0']  = float(elem[16])
        all_gt[sod]['VZ0']  = float(elem[17])
        all_gt[sod]['ATTX0']= float(elem[25])
        all_gt[sod]['ATTY0']= float(elem[26])
        all_gt[sod]['ATTZ0']= -float(elem[24])
        Ren = trans.Cen([all_gt[sod]['X0'],all_gt[sod]['Y0'],all_gt[sod]['Z0']])
        ani0 = [all_gt[sod]['ATTX0']/180*math.pi,\
                all_gt[sod]['ATTY0']/180*math.pi,\
                all_gt[sod]['ATTZ0']/180*math.pi]
        Rni0 = trans.att2m(ani0)
        Rei0 = np.matmul(Ren,Rni0)
        tei0 = np.array([all_gt[sod]['X0'],all_gt[sod]['Y0'],all_gt[sod]['Z0']])
        Tei0 = np.eye(4,4)
        Tei0[0:3,0:3] = Rei0
        Tei0[0:3,3] = tei0
        if not is_ref_set:
            is_ref_set = True
            Ten0 = np.eye(4,4)
            Ten0[0:3,0:3] = trans.Cen(tei0)
            Ten0[0:3,3] = tei0
        Tn0i0 = np.matmul(np.linalg.inv(Ten0),Tei0)
        Tn0i1 = np.matmul(Tn0i0,Ti0i1)
        all_gt[sod]['T'] = Tn0i1
    all_gt_keys =sorted(all_gt.keys())
    fp.close()

    all_imu = np.loadtxt(args.imupath,delimiter=' ')

    if args.use_gnss and os.path.isfile(args.gnsspath):
        fix_map = {b'Fixed':1.0,b'Float':0.0}
        all_gnss = np.genfromtxt(args.gnsspath,converters={16: lambda x: fix_map[x]})
    else:
        all_gnss = []
    if args.use_odo and os.path.isfile(args.odopath):
        all_odo = np.genfromtxt(args.odopath)
        all_odo = all_odo[np.fabs(all_odo[:,0] - np.round(all_odo[:,0]))<0.001]
        np.random.seed(12345)
        all_odo[:,1:] += np.random.randn(all_odo.shape[0],3)*0.05
    else:
        all_odo = []

    tstamps = []
    # try:
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.imagestamp, args.enable_h5,\
                                                     args.h5path, args.calib, args.stride)):
        
        if args.show_plot:
            show_image(image[0])
        if dbaf is None:
            args.image_size = [image.shape[2], image.shape[3]]
            dbaf = DBAFusion(args)
            dbaf.frontend.all_imu    = all_imu
            dbaf.frontend.all_stamp  = np.loadtxt(args.imagestamp,str,delimiter=',')
            dbaf.frontend.all_gnss   = all_gnss
            dbaf.frontend.all_odo    = all_odo
            if len(all_gt) > 0:
                dbaf.frontend.all_gt = all_gt
                dbaf.frontend.all_gt_keys = all_gt_keys
            dbaf.video.Ti1c = np.array(
            [0.99988370,-0.00563944,-0.01418468,-0.15590000,
           0.01424932,0.01159187,0.99983149,0.63466000,
           -0.00547407,-0.99991712,0.01167088,0.04605000,
           0.00000000,0.00000000,0.00000000,1.00000000]).reshape([4,4])
            dbaf.video.tbg = np.array([-0.0125, -0.26, 0.2091])

            dbaf.video.Tbc = gtsam.Pose3(dbaf.video.Ti1c)
            dbaf.video.state.set_imu_params([ 0.0003924 * 25,0.000205689024915 * 25, 0.004905 * 10, 0.000001454441043 * 25])
            if args.use_gnss:
                dbaf.video.init_pose_sigma = np.array([1.0, 1.0, 10.0,10.0,10.0,10.0])
            else:
                dbaf.video.init_pose_sigma = np.array([0.1, 0.1, 0.0001, 0.0001,0.0001,0.0001])
            dbaf.video.init_bias_sigma = np.array([1.0,1.0,1.0, 0.1, 0.1, 0.1])
            dbaf.frontend.translation_threshold = args.translation_threshold
            dbaf.frontend.graph.mask_threshold  = args.mask_threshold
        dbaf.track(t, image, intrinsics=intrinsics)
    dbaf.save_vis_easy()
    dbaf.terminate()
