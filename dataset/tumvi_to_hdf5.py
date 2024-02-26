from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import argparse

import h5py
import pickle

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, imagestamp, h5path, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    Kn = np.eye(3)
    Kn[0,0] = fx 
    Kn[0,2] = cx 
    Kn[1,1] = fy 
    Kn[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]
    image_stamps = np.loadtxt(imagestamp,str,delimiter=',')
    image_dict = dict(zip(image_stamps[:,1],image_stamps[:,0]))
    h5_f = h5py.File(h5path,'w')
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))

        if len(calib) > 4:
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K,calib[4:],np.eye(3),Kn,(512,512),cv2.CV_32FC1)
            image = cv2.remap(image, m1, m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        tt = float(image_dict[imfile]) /1e9

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy ])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        h5_f.create_dataset('%.10f'%tt,data = np.fromstring(pickle.dumps((tt, image[None], intrinsics)),dtype='uint8'))

        yield tt, image[None], intrinsics
    h5_f.close()

if __name__ == '__main__':

    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--imagestamp", type=str, help="")
    parser.add_argument("--h5path", type=str, help="")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")
    parser.add_argument("--show_plot", action="store_true", help="")

    args = parser.parse_args()

    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.imagestamp,\
                                                     args.h5path, args.calib, args.stride)):
        if args.show_plot:
            show_image(image[0])
