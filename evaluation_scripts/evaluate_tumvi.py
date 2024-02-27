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
import matplotlib
import copy
import os

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
matplotlib.rcParams['font.family'] = 'Arial'

font0={'family':'Arial',
     'style':'normal',
    'weight':'bold',
      'color':'black',
      'size':6
}
font1={'family':'Arial',
     'style':'normal',
    'weight':'bold',
      'color':'black',
      'size':8
}


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
    color_list = [np.array([133,164,195])/255.0,[1,0.6,1],[1,0,0]]
    plt.figure('1',figsize=[6,6])
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, help='seq', default='outdoors1')
    parser.add_argument('--batch', action="store_true")
    args = parser.parse_args()
    args.subcommand = 'tum'
    seq = args.seq

    args.ref_file = '/mnt/z/tum-vi/dataset-%s_512_16/dso/gt_imu.csv' % seq
    # Convert GT format
    dd = np.loadtxt(args.ref_file,delimiter=',',comments='#')
    dd_new = np.copy(dd)
    dd_new[:,0] /= 1e9
    dd_new[:,4] = dd[:,7] 
    dd_new[:,5] = dd[:,4] 
    dd_new[:,6] = dd[:,5] 
    dd_new[:,7] = dd[:,6] 
    args.ref_file = os.path.join(os.path.dirname(args.ref_file),'gt_imu_temp.txt')
    np.savetxt(args.ref_file, dd_new,delimiter=' ')

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
    args.est_files = ['results/result_%s.txt' % seq]
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

        # use the starting part for scale estimation
        args.n_to_align = np.where((np.array(traj_ref_sel.timestamps)[1:]-np.array(traj_ref_sel.timestamps)[0:-1])>100)[0][0]-1
        pose_relation = common.get_pose_relation(args)
        result = ape(traj_ref=traj_ref_sel, traj_est=traj_est_sel,
                     pose_relation=pose_relation, align=args.align,
                     correct_scale=args.correct_scale, n_to_align=args.n_to_align,
                     align_origin=args.align_origin, ref_name=ref_name,
                     est_name=est_name)
        traj_est_sel = copy.deepcopy(result.trajectories[est_name])
        T01 = result.np_arrays['alignment_transformation_sim3']

        # metric-scale APE calculation
        result = ape(traj_ref=traj_ref_sel, traj_est=traj_est_sel,
                     pose_relation=pose_relation, align=args.align,
                     correct_scale=False, n_to_align=-1,
                     align_origin=args.align_origin, ref_name=ref_name,
                     est_name=est_name)
        print(result)

        if args.batch:
            quit()

        #  visualization
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
            plt.plot(x_series,y_series,c=[0,0,0],linestyle = '--',linewidth=0.5)

        x_series=[]
        y_series=[]
        z_series=[]
        for i in range(len(traj_est.poses_se3)):
            TTT = traj_est.poses_se3[i]
            x_series.append(TTT[0,3])
            y_series.append(TTT[1,3])
            z_series.append(TTT[2,3])
        plt.plot(x_series,y_series,c=color_list[iii],label = label_list[iii],linewidth=0.5)

    ll = max(max(x_series)-min(x_series),max(y_series)-min(y_series))
    plt.xlim([(max(x_series)+min(x_series))/2 - 0.65*ll,(max(x_series)+min(x_series))/2+0.65*ll])
    plt.ylim([(max(y_series)+min(y_series))/2 - 0.65*ll,(max(y_series)+min(y_series))/2+0.65*ll])
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    plt.tick_params(labelsize=6,direction='in')
    lg = plt.legend(loc='upper right',markerscale=3,fontsize=5,framealpha=1,ncol=1,columnspacing=0.3,handletextpad=0.3,edgecolor='black',fancybox=False)
    lg.set_zorder(200)
    lg.get_frame().set_linewidth(0.8)
    plt.gca().yaxis.set_label_coords(-.1, .5)
    plt.show()
    