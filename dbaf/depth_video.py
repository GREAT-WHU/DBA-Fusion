import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value

from droid_net import cvx_upsample
import geom.projective_ops as pops

from multi_sensor import MultiSensorState
import gtsam
from gtsam.symbol_shorthand import B, V, X
from scipy.spatial.transform import Rotation
import copy
import logging
import geoFunc.trans as trans
from lietorch import SE3

def BA2GTSAM(H: np.ndarray, v: np.ndarray, Tbc: gtsam.Pose3):
    A = -Tbc.inverse().AdjointMap()
    # A = -np.eye(6,6)
    A = np.concatenate([A[3:6,:],A[0:3,:]],axis=0)
    ss = H.shape[0]//6
    J = np.zeros_like(H)
    for i in range(ss):
       J[(i*6):(i*6+6),(i*6):(i*6+6)] = A
    JT = J.T
    return np.matmul(np.matmul(JT,H),J),np.matmul(JT,v)

def CustomHessianFactor(values: gtsam.Values, H: np.ndarray, v: np.ndarray):
    info_expand = np.zeros([H.shape[0]+1,H.shape[1]+1])
    info_expand[0:-1,0:-1] = H
    info_expand[0:-1,-1] = v
    info_expand[-1,-1] = 0.0 # This is meaningless.
    h_f = gtsam.HessianFactor(values.keys(),[6]*len(values.keys()),info_expand)
    l_c = gtsam.LinearContainerFactor(h_f,values)
    return l_c

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, save_pkl = False, stereo=False, device="cuda:0"):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float64).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        
        ### DBAFusion
        # for .pkl saving
        self.disps_save = torch.ones(5000, ht//8, wd//8, device="cuda", dtype=torch.float)
        self.poses_save = torch.ones(5000, 7, device="cuda", dtype=torch.float)
        self.tstamp_save = torch.zeros(5000, device="cuda", dtype=torch.float64)
        self.images_save = torch.zeros(5000, ht//8, wd//8, 3, device="cuda", dtype=torch.float)
        self.count_save = 0
        self.save_pkl = save_pkl

        self.state = MultiSensorState()
        self.last_t0 = 0
        self.last_t1 = 0
        self.cur_graph = None
        self.cur_result = None
        self.marg_factor = None
        self.prior_factor = []
        self.prior_factor_map = {}
        self.cur_ii = None
        self.cur_jj = None
        self.cur_target = None
        self.cur_weight = None
        self.cur_eta = None

        self.imu_enabled = False
        self.ignore_imu = False

        self.xyz_ref = []
        
        # extrinsics, need to be set in the main .py
        self.Ti1c = None  # shape = (4,4)
        self.Tbc = None   # gtsam.Pose3
        self.tbg = None   # shape = (3)

        self.reinit = False
        self.vi_init_t1 = -1
        self.vi_init_time = 0.0
        self.gnss_init_t1 = -1
        self.gnss_init_time = 0.0
        self.ten0 = None
        
        self.init_pose_sigma =np.array([0.1, 0.1, 0.0001, 0.0001,0.0001,0.0001])
        self.init_bias_sigma =np.array([1.0,1.0,1.0, 0.1, 0.1, 0.1])

        self.logger = logging.getLogger('dba_fusion')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('dba_fusion.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.info('Start logging!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask
    
    def reproject_comp(self, ii, jj, xyz_comp):
        ii, jj = DepthVideo.format_indicies(ii,jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform_comp(Gs, self.disps[None], self.intrinsics[None], ii, jj, xyz_comp)

        return coords, valid_mask
    
    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def rm_new_gnss(self, t1):
        if (self.gnss_init_t1> 0 and self.state.gnss_valid[t1]) or self.state.odo_valid[t1]:
            graph_temp = gtsam.NonlinearFactorGraph()
            linear_point  = self.marg_factor.linearizationPoint()
            graph_temp.push_back(self.marg_factor)

            if self.state.gnss_valid[t1]:
                T1 = self.state.wTbs[t1]
                T0 = self.state.wTbs[t1-1]
                p = np.matmul(trans.Cen(self.ten0).T, self.state.gnss_position[t1] - self.ten0)
                n0pbg = self.state.wTbs[t1].rotation().rotate(self.tbg)
                p = p - n0pbg
                p = p - T1.translation() + T0.translation()
                if not linear_point.exists(X(t1-1)):
                    linear_point.insert(X(t1-1), self.cur_result.atPose3(X(t1-1)))
                gnss_factor = gtsam.GPSFactor(X(t1-1), p,\
                              gtsam.noiseModel.Robust.Create(\
                              gtsam.noiseModel.mEstimator.Cauchy(0.08),\
                  gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,1.0,5.0]))))
                graph_temp.push_back(gnss_factor)
            if self.state.odo_valid[t1]:
                v1 = np.matmul(self.state.wTbs[t1].rotation().matrix().T, self.state.vs[t1])
                v0 = np.matmul(self.state.wTbs[t1-1].rotation().matrix().T, self.state.vs[t1-1])
                v = self.state.odo_vel[t1] - v1 + v0
                if not linear_point.exists(X(t1-1)):
                    linear_point.insert(X(t1-1), self.cur_result.atPose3(X(t1-1)))
                if not linear_point.exists(V(t1-1)):
                    linear_point.insert(V(t1-1), self.cur_result.atVector(V(t1-1)))
                odo_factor = gtsam.VelFactor(X(t1-1),V(t1-1),v,gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0,2.0,2.0])))
                graph_temp.push_back(odo_factor)           
            
            h_factor = graph_temp.linearizeToHessianFactor(linear_point)
            self.marg_factor = gtsam.LinearContainerFactor(h_factor,linear_point)
            
    
    def set_prior(self, t0, t1):
        for i in range(t0,t0+2):
            self.prior_factor_map[i] = []
            self.prior_factor_map[i].append(gtsam.PriorFactorPose3(X(i),\
                                         self.state.wTbs[i], \
                                         gtsam.noiseModel.Diagonal.Sigmas(self.init_pose_sigma)))
            if not self.ignore_imu:
                self.prior_factor_map[i].append(gtsam.PriorFactorConstantBias(B(i),\
                                             self.state.bs[i], \
                                             gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma)))
            self.last_t0 = t0
            self.last_t1 = t1

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """
        with self.get_lock():
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # 1) visual-only BA
            # 2) multi-sensor BA
            if not self.imu_enabled: 
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                    target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
                for i in range(self.last_t0, min(ii.min().item(), jj.min().item())):
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save] = self.tstamp[i].clone()
                        self.disps_save[self.count_save] = self.disps[i].clone()
                        self.poses_save[self.count_save] = self.poses[i].clone()
                        self.images_save[self.count_save] = self.images[i,[2,1,0],::8,::8].permute(1,2,0) / 255.0 # might be "3::8, 3::8"?
                        self.count_save += 1

                self.last_t0 = min(ii.min().item(), jj.min().item())
                self.last_t1 = t1
            else:
                t0 = min(ii.min().item(), jj.min().item())

                """ marginalization """
                if self.last_t1!=t1 or self.last_t0 != t0:
                    if self.last_t0 > t0:
                        t0 = self.last_t0
                    elif self.last_t0 == t0:
                        t0 = self.last_t0
                    else:
                        marg_paras = []
                        graph = gtsam.NonlinearFactorGraph()
                        marg_idx = torch.logical_and(torch.greater_equal(self.cur_ii,self.last_t0),\
                                                    torch.less(self.cur_ii,t0))
                        marg_idx2 = torch.logical_and(torch.less(self.cur_ii,self.last_t1-2),\
                                                     torch.less(self.cur_jj,self.last_t1-2))
                        marg_idx = torch.logical_and(marg_idx,marg_idx2)

                        marg_ii = self.cur_ii[marg_idx]
                        marg_jj = self.cur_jj[marg_idx]
                        marg_t0 = self.last_t0 
                        marg_t1 = t0 + 1
                        if len(marg_ii) > 0:
                            marg_t0 = self.last_t0 
                            marg_t1 = torch.max(marg_jj).item()+1
                            marg_result = gtsam.Values()
                            for i in range(self.last_t0,marg_t1):
                                if i < t0:
                                    marg_paras.append(X(i))
                                    if self.save_pkl:
                                        # save marginalized results
                                        self.tstamp_save[self.count_save] = self.tstamp[i].clone()
                                        self.disps_save[self.count_save] = self.disps[i].clone()
                                        self.poses_save[self.count_save] = self.poses[i].clone()
                                        self.images_save[self.count_save] = self.images[i,[2,1,0],::8,::8].permute(1,2,0) / 255.0 # might be "3::8, 3::8"?
                                        self.count_save += 1
                                marg_result.insert(X(i), self.cur_result.atPose3(X(i)))
                                
                            marg_target = self.cur_target[marg_idx]
                            marg_weight = self.cur_weight[marg_idx]
                            marg_eta = self.cur_eta[0:marg_t1-marg_t0]
    
                            bacore = droid_backends.BACore()
                            bacore.init(self.poses, self.disps, self.intrinsics[0], torch.zeros_like(self.disps_sens),
                                marg_target, marg_weight, marg_eta, marg_ii, marg_jj, marg_t0, marg_t1, itrs, lm, ep, motion_only)
                            H = torch.zeros([(marg_t1-marg_t0)*6,(marg_t1-marg_t0)*6],dtype=torch.float64,device='cpu')
                            v = torch.zeros([(marg_t1-marg_t0)*6],dtype=torch.float64,device='cpu')
                            bacore.hessian(H,v)
                            
                            for i in range(6): H[i,i] += 0.00025  # for stability

                            Hg,vg = BA2GTSAM(H,v,self.Tbc)
                            vis_factor = CustomHessianFactor(marg_result,Hg,vg)
    
                            graph.push_back(vis_factor)

                        for i in range(self.last_t0,marg_t1):
                            if i < t0:
                                if X(i) not in marg_paras:
                                    marg_paras.append(X(i))
                                if not self.ignore_imu:
                                    marg_paras.append(V(i))
                                    marg_paras.append(B(i))
                                    graph.push_back(gtsam.gtsam.CombinedImuFactor(\
                                                X(i),V(i),X(i+1),V(i+1),B(i),B(i+1),\
                                                self.state.preintegrations[i]))
                                if self.gnss_init_t1 > 0:
                                    if self.state.gnss_valid[i]:
                                        p = np.matmul(trans.Cen(self.ten0).T, self.state.gnss_position[i] - self.ten0)
                                        n0pbg = self.state.wTbs[i].rotation().rotate(self.tbg)
                                        p = p - n0pbg
                                        gnss_factor = gtsam.GPSFactor(X(i), p,\
                                                      gtsam.noiseModel.Robust.Create(\
                                                      gtsam.noiseModel.mEstimator.Cauchy(0.08),\
                                          gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,1.0,5.0]))))
                                        graph.push_back(gnss_factor)
                                if self.state.odo_valid[i]:
                                    vb = self.state.odo_vel[i]
                                    odo_factor = gtsam.VelFactor(X(i),V(i),vb,gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0,2.0,2.0])))
                                    graph.push_back(odo_factor)
                        
                        keys = self.prior_factor_map.keys()
                        for i in sorted(keys):
                            if i < t0:
                                for iii in range(len(self.prior_factor_map[i])):
                                    graph.push_back(self.prior_factor_map[i][iii])
                            del self.prior_factor_map[i]
                        if not self.marg_factor == None:
                            graph.push_back(self.marg_factor)

                        self.marg_factor = gtsam.marginalizeOut(graph,self.cur_result,marg_paras)

                        # covariance inflation of IMU biases
                        if self.reinit == True:
                            all_keys = self.marg_factor.keys()
                            for i in range(len(all_keys)):
                                if all_keys[i] == B(t0):
                                    all_keys[i] = B(0)
                            graph = gtsam.NonlinearFactorGraph()
                            graph.push_back(self.marg_factor.rekey(all_keys))
                            b_l = gtsam.BetweenFactorConstantBias(B(0),B(t0),gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])),\
                                                                  gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma))
                            graph.push_back(b_l)
                            result_tmp = self.marg_factor.linearizationPoint()
                            result_tmp.insert(B(0),result_tmp.atConstantBias(B(t0)))
                            self.marg_factor = gtsam.marginalizeOut(graph,result_tmp,[B(0)])
                            self.reinit = False

                    self.last_t0 = t0
                    self.last_t1 = t1

                """ optimization """
                H = torch.zeros([(t1-t0)*6,(t1-t0)*6],dtype=torch.float64,device='cpu')
                v = torch.zeros([(t1-t0)*6],dtype=torch.float64,device='cpu')
                dx = torch.zeros([(t1-t0)*6],dtype=torch.float64,device='cpu') 

                bacore = droid_backends.BACore()
                active_index    = torch.logical_and(ii>=t0,jj>=t0)
                self.cur_ii     = ii[active_index]
                self.cur_jj     = jj[active_index]
                self.cur_target = target[active_index]
                self.cur_weight = weight[active_index]
                self.cur_eta    = eta[(t0-ii.min().item()):]

                bacore.init(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                    self.cur_target, self.cur_weight, self.cur_eta, self.cur_ii, self.cur_jj, t0, t1, itrs, lm, ep, motion_only)

                self.cur_graph = gtsam.NonlinearFactorGraph()
                params = gtsam.LevenbergMarquardtParams()#;params.setMaxIterations(1)

                # imu factor
                if not self.ignore_imu:
                    for i in range(t0,t1):
                        if i > t0:
                            imu_factor = gtsam.gtsam.CombinedImuFactor(\
                                X(i-1),V(i-1),X(i),V(i),B(i-1),B(i),\
                                self.state.preintegrations[i-1])
                            self.cur_graph.add(imu_factor)

                # prior factor
                keys = self.prior_factor_map.keys()
                for i in sorted(keys):
                    if i >= t0 and i < t1:
                        for iii in range(len(self.prior_factor_map[i])):
                            self.cur_graph.push_back(self.prior_factor_map[i][iii])
                
                # marginalization factor
                if self.marg_factor is not None:
                    self.cur_graph.push_back(self.marg_factor)

                # GNSS factor
                if self.gnss_init_t1 > 0:
                    for i in range(t0,t1):
                        if self.state.gnss_valid[i]:
                            p = np.matmul(trans.Cen(self.ten0).T, self.state.gnss_position[i] - self.ten0)
                            n0pbg = self.state.wTbs[i].rotation().rotate(self.tbg)
                            p = p - n0pbg
                            gnss_factor = gtsam.GPSFactor(X(i), p,\
                                          gtsam.noiseModel.Robust.Create(\
                                                      gtsam.noiseModel.mEstimator.Cauchy(0.08),\
                                          gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,1.0,5.0]))))
                            self.cur_graph.push_back(gnss_factor)
                
                # Odo factor
                for i in range(t0,t1):
                    if self.state.odo_valid[i]:
                        vb = self.state.odo_vel[i]
                        odo_factor = gtsam.VelFactor(X(i),V(i),vb,gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0,2.0,2.0])))
                        self.cur_graph.push_back(odo_factor)

                """ multi-sensor DBA iterations """
                for iter in range(2):
                    if iter > 0:
                        self.cur_graph.resize(self.cur_graph.size()-1)
                    bacore.hessian(H,v) # camera frame
                    Hgg = gtsam.BA2GTSAM(H,v,self.Tbc)
                    Hg = Hgg[0:(t1-t0)*6,0:(t1-t0)*6]
                    vg = Hgg[0:(t1-t0)*6,(t1-t0)*6]

                    initial = gtsam.Values()
                    for i in range(t0,t1):
                        initial.insert(X(i), self.state.wTbs[i]) # the indice need to be handled
                    initial_vis = copy.deepcopy(initial)
                    vis_factor = CustomHessianFactor(initial_vis,Hg,vg)
                    self.cur_graph.push_back(vis_factor)
                    
                    if not self.ignore_imu:
                        for i in range(t0,t1):
                            initial.insert(B(i),self.state.bs[i])
                            initial.insert(V(i),self.state.vs[i])

                    optimizer = gtsam.LevenbergMarquardtOptimizer(self.cur_graph, initial, params)
                    self.cur_result = optimizer.optimize()

                    # retraction and depth update
                    for i in range(t0,t1):
                        p0 = initial.atPose3(X(i))
                        p1 = self.cur_result.atPose3(X(i))
                        xi = gtsam.Pose3.Logmap(p0.inverse()*p1)
                        dx[(i-t0)*6:(i-t0)*6+6] = torch.tensor(xi)
                        if not self.ignore_imu:
                            self.state.bs[i] = self.cur_result.atConstantBias(B(i))
                            self.state.vs[i] = self.cur_result.atVector(V(i))
                        self.state.wTbs[i] = self.cur_result.atPose3(X(i))
                    dx = torch.tensor(gtsam.GTSAM2BA(dx,self.Tbc))
                    dx_dz = bacore.retract(dx)
                del bacore
            self.disps.clamp_(min=0.001)
