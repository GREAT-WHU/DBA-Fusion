import argparse
import logging
import typing

import numpy as np

import evo.common_ape_rpe as common
from evo.core import lie_algebra, sync, metrics
from evo.core.result import Result
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface, log
from evo.tools.settings import SETTINGS

import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation
import bisect
import math
import time

logger = logging.getLogger(__name__)

SEP = "-" * 80  # separator line

def ape(traj_ref: PosePath3D, traj_est: PosePath3D,
        pose_relation: metrics.PoseRelation, align: bool = False,
        correct_scale: bool = False, n_to_align: int = -1,
        align_origin: bool = False, ref_name: str = "reference",
        est_name: str = "estimate",
        change_unit: typing.Optional[metrics.Unit] = None) -> Result:
    if n_to_align >0 : 
        print('>>>>> only use the starting segment')
        n_to_align = np.where((np.array(traj_ref.timestamps)[1:]-np.array(traj_ref.timestamps)[0:-1])>100)[0][0]-1

    # Align the trajectories.
    only_scale = correct_scale and not align
    alignment_transformation = None
    if align or correct_scale:
        logger.debug(SEP)
        alignment_transformation = lie_algebra.sim3(
            *traj_est.align(traj_ref, correct_scale, only_scale, n=n_to_align))
    elif align_origin:
        logger.debug(SEP)
        alignment_transformation = traj_est.align_origin(traj_ref)

    # Calculate APE.
    logger.debug(SEP)
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    if change_unit:
        ape_metric.change_unit(change_unit)

    title = str(ape_metric)
    if align and not correct_scale:
        title += "\n(with SE(3) Umeyama alignment)"
    elif align and correct_scale:
        title += "\n(with Sim(3) Umeyama alignment)"
    elif only_scale:
        title += "\n(scale corrected)"
    elif align_origin:
        title += "\n(with origin alignment)"
    else:
        title += "\n(not aligned)"
    if (align or correct_scale) and n_to_align != -1:
        title += " (aligned poses: {})".format(n_to_align)

    ape_result = ape_metric.get_result(ref_name, est_name)
    ape_result.info["title"] = title

    logger.debug(SEP)
    logger.info(ape_result.pretty_str())

    ape_result.add_trajectory(ref_name, traj_ref)
    ape_result.add_trajectory(est_name, traj_est)
    if isinstance(traj_est, PoseTrajectory3D):
        seconds_from_start = np.array(
            [t - traj_est.timestamps[0] for t in traj_est.timestamps])
        ape_result.add_np_array("seconds_from_start", seconds_from_start)
        ape_result.add_np_array("timestamps", traj_est.timestamps)
        ape_result.add_np_array("distances_from_start", traj_ref.distances)
        ape_result.add_np_array("distances", traj_est.distances)

    if alignment_transformation is not None:
        ape_result.add_np_array("alignment_transformation_sim3",
                                alignment_transformation)

    return ape_result


if __name__ == '__main__':
    color_list = [[0,0,1],[1,0.6,1],[1,0,0]]
    plt.figure('1',figsize=[6,6])
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, help='seq',default='0010')
    args = parser.parse_args()
    args.subcommand = 'tum'
    seq = args.seq
    args.ref_file = '/home/zhouyuxuan/data/2013_05_28_drive_%s_sync/gt_local.txt' % seq
    args.pose_relation = 'trans_part'
    args.align = True
    args.correct_scale = False
    args.n_to_align = 1
    args.align_origin = False
    args.plot_mode = 'xyz'
    args.plot_x_dimension = 'seconds'
    args.plot_colormap_min = None
    args.plot_colormap_max = None
    args.plot_colormap_max_percentile = None
    args.ros_map_yaml = None
    args.plot = True
    args.est_files = ['results/result_%s.txt' %seq]
    label_list = ['DBA-Fusion (M)']
    args.save_plot = False
    args.serialize_plot = False
    for iii in range(len(args.est_files)):
        args.est_file = args.est_files[iii]

        if args.est_file.find('visual') != -1:
            args.correct_scale = True
        else:
            args.correct_scale = False
        traj_ref, traj_est, ref_name, est_name = common.load_trajectories(args)
        traj_ref_sel, traj_est_sel = sync.associate_trajectories(
            traj_ref, traj_est, 0.01,0.0,
            first_name=ref_name, snd_name=est_name)
        args.n_to_align = -1
        pose_relation = common.get_pose_relation(args)
        result = ape(traj_ref=traj_ref_sel, traj_est=traj_est_sel,
                     pose_relation=pose_relation, align=args.align,
                     correct_scale=args.correct_scale, n_to_align=args.n_to_align,
                     align_origin=args.align_origin, ref_name=ref_name,
                     est_name=est_name)
        traj_est_sel = copy.deepcopy(result.trajectories[est_name])
        T01 = result.np_arrays['alignment_transformation_sim3']
        print(T01)
        result = ape(traj_ref=traj_ref_sel, traj_est=traj_est_sel,
                     pose_relation=pose_relation, align=args.align,
                     correct_scale=False, n_to_align=-1,
                     align_origin=args.align_origin, ref_name=ref_name,
                     est_name=est_name)
        print(result)
        traj_est.transform(T01)
        
        if iii == 0:
            x_series=[]
            y_series=[]
            z_series=[]
            for i in range(len(traj_ref.poses_se3)):
                TTT = traj_ref.poses_se3[i]
                x_series.append(TTT[0,3])
                y_series.append(TTT[1,3])
                z_series.append(TTT[2,3])
            plt.plot(x_series,y_series,c=[0,0,0],linestyle = '--')

        x_series=[]
        y_series=[]
        z_series=[]
        for i in range(len(traj_est.poses_se3)):
            TTT = traj_est.poses_se3[i]
            x_series.append(TTT[0,3])
            y_series.append(TTT[1,3])
            z_series.append(TTT[2,3])
            ppp = TTT[0:3,3]
            qqq = Rotation.from_matrix(TTT[:3, :3]/np.power(np.linalg.det(TTT[:3, :3]),1.0/3)).as_quat()
        plt.plot(x_series,y_series,c=color_list[iii],label = label_list[iii])
    
    # t_series=[]
    # x_series=[]
    # y_series=[]
    # z_series=[]
    # for i in range(len(traj_ref_sel.timestamps)):
    #     T0 = traj_ref_sel.poses_se3[i]
    #     T1 = traj_est_sel.poses_se3[i]
    #     T01 = np.matmul(np.linalg.inv(T0),T1)
    #     att = Rotation.from_matrix(T01[0:3,0:3]).as_rotvec()
    #     t_series.append(traj_ref_sel.timestamps[i])
    #     x_series.append(att[0])
    #     y_series.append(att[1])
    #     z_series.append(att[2])
    # plt.figure()
    # plt.plot(t_series,x_series)
    # plt.plot(t_series,y_series)
    # plt.plot(t_series,z_series)
    # plt.show()

    print('Evaluating relative pose error ...')
    subtraj_length = [100,200,300,400,500,600,700,800]
    max_dist_difH=1
    rel_trans_error_dist = []
    rel_att_error_dist = []
    for i in range(8):
        subsection_index=[]
        max_dist_diff=0.2*subtraj_length[i]
        traj_len = len(traj_ref_sel.timestamps)
        for j in range(traj_len-2):
            k = bisect.bisect(traj_ref_sel.distances,traj_ref_sel.distances[j]+subtraj_length[i]-max_dist_difH)
            if k > 0 and k < traj_len and math.fabs(traj_ref_sel.distances[k] - (traj_ref_sel.distances[j]+subtraj_length[i]))< max_dist_difH:
                subsection_index.append([j,k])
        print("The trajectory at %dm have %d matching points... " %(subtraj_length[i],len(subsection_index)))
        rel_tran_errors = []
        rel_att_errors = []
        for ii in subsection_index:
            T_gt_1 =traj_ref_sel.poses_se3[ii[0]]
            T_gt_2 =traj_ref_sel.poses_se3[ii[1]]
            T_est_1 =traj_est_sel.poses_se3[ii[0]]
            T_est_2 =traj_est_sel.poses_se3[ii[1]]
            T_gt_12=np.matmul(np.linalg.inv(T_gt_1),T_gt_2)
            T_est_12=np.matmul(np.linalg.inv(T_est_1),T_est_2)
            T_error=np.matmul(np.linalg.inv(T_gt_12),T_est_12)
            rel_tran_error = np.linalg.norm(T_error[0:3,3])
            rel_att_error = np.linalg.norm(Rotation.from_matrix(T_error[0:3,0:3]).as_rotvec())
            rel_tran_errors.append(rel_tran_error/subtraj_length[i]*100)
            rel_att_errors.append(rel_att_error/subtraj_length[i]*100/math.pi*180)
        rel_trans_error_dist.append(np.mean(np.array(rel_tran_errors)))
        rel_att_error_dist.append(np.mean(np.array(rel_att_errors)))
    print('Relative Translation Error: %f%%' % np.mean(np.array(rel_trans_error_dist)))
    print('Relative Rotation Error: %f deg / 100 m' % np.mean(np.array(rel_att_error_dist)))
    plt.show()