import torch
import lietorch
import numpy as np
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from dbaf_frontend import DBAFusionFrontend
from collections import OrderedDict
from torch.multiprocessing import Process

from lietorch import SE3
import geom.projective_ops as pops
import droid_backends
import pickle

class DBAFusion:
    def __init__(self, args):
        super(DBAFusion, self).__init__()
        self.load_weights(args.weights) # load DroidNet weights
        self.args = args

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, save_pkl = args.save_pkl, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DBAFusionFrontend(self.net, self.video, self.args)

        self.pklpath = args.pklpath

    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """
        del self.frontend

    def save_vis_easy(self):
        mcameras = {}
        mpoints = {}
        mstamps = {}
        with torch.no_grad():
            dirty_index = torch.arange(0,self.video.count_save,device='cuda')

            stamps= torch.index_select(self.video.tstamp_save, 0 ,dirty_index)
            poses=  torch.index_select( self.video.poses_save, 0 ,dirty_index)
            disps=  torch.index_select( self.video.disps_save, 0 ,dirty_index)
            images = torch.index_select( self.video.images_save, 0 ,dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()
            points = droid_backends.iproj(SE3(poses).inv().data, disps, self.video.intrinsics[0]).cpu()
            thresh = 0.4 * torch.ones_like(disps.mean(dim=[1,2])) / 4.0  * (1.0 / torch.median(disps.mean(dim=[1,2])))
            # thresh = 0.4 * torch.ones_like(disps.mean(dim=[1,2])) 
            count = droid_backends.depth_filter(
                self.video.poses_save, self.video.disps_save, self.video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 1) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()
                mcameras[ix] = pose
                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                stamp = stamps[i].cpu()
                mpoints[ix] = {'pts':pts,'clr':clr}
                mstamps[ix] = stamp
        ddict = {'points':mpoints,'cameras':mcameras,'stamps':mstamps}
        f_save = open('reconstructions/%s.pkl' % self.pklpath, 'wb')
        pickle.dump(ddict,f_save) 

        mcameras = {}
        mpoints = {}
        mstamps = {}
        with torch.no_grad():
            dirty_index = torch.arange(0,self.video.count_save,device='cuda')

            stamps= torch.index_select(self.video.tstamp_save, 0 ,dirty_index)
            poses=  torch.index_select( self.video.poses_save, 0 ,dirty_index)
            disps=  torch.index_select( self.video.disps_save, 0 ,dirty_index)
            images = torch.index_select( self.video.images_save, 0 ,dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()
            points = droid_backends.iproj(SE3(poses).inv().data, disps, self.video.intrinsics[0]).cpu()
            thresh = 0.4 * torch.ones_like(disps.mean(dim=[1,2]))
            count = droid_backends.depth_filter(
                self.video.poses_save, self.video.disps_save, self.video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 0) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()
                mcameras[ix] = pose
                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                stamp = stamps[i].cpu()
                mpoints[ix] = {'pts':pts,'clr':clr}
                mstamps[ix] = stamp
        ddict = {'points':mpoints,'cameras':mcameras,'stamps':mstamps}
        f_save = open('reconstructions/%s_loose.pkl' % self.pklpath, 'wb')
        pickle.dump(ddict,f_save) 
