import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from lietorch import SO3, SE3, Sim3
from scipy.spatial.transform import Rotation as R
import copy
import pickle
import re

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # if colors != None:
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

    rotating = False

def str2array(ss):
    elem = re.sub('\s\s+',' ',ss).split(' ')
    num=[]
    for e in elem:
        num.append(float(e))
    return np.array(num)



f = open(r'./reconstructions/outdoors6.pkl','rb')
dump_data= pickle.load(f)
print(dump_data.keys())

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='123')
vis.get_render_option().point_size = 2
opt = vis.get_render_option()
opt.background_color = np.asarray([0,0,0])

def key_action_callback(vis, action, mods):
    print(action)
    if action == 1:  # key down
        ctr = vis.get_view_control()
        view_params = ctr.convert_to_view_parameters()
        print(view_params)
    return True

# key_action_callback will be triggered when there's a keyboard press, release or repeat event
vis.register_key_action_callback(32, key_action_callback)  # space
# animation_callback is always repeatedly called by the visualizer

for ix in sorted(dump_data['points'].keys()):
    if ix < 1800 : continue
    if ix > 2800 : continue

    dd=dump_data['points'][ix]
    pts = dd['pts'] # * 17.0
    clr = dd['clr']
    pose = dump_data['cameras'][ix]
    pose[0:3,3] = pose[0:3,3]  #*  17.0
    npts_c = np.matmul(pose[0:3,0:3].T,(pts-pose[0:3,3]).T).T
    npts = np.asarray(pts)
    nclr = np.asarray(clr)
    mask0 = npts_c[:,1]> -5.0
    mask1 = np.logical_or(npts_c[:,1]> -0.0,nclr[:,0]<0.4)
    mask2 = npts_c[:,2] < 10.0
    mask = np.logical_and(np.logical_and(mask0,mask1),mask2)
    point_actor = create_point_actor(pts[mask], clr[mask])

    vis.add_geometry(point_actor)
    cam_actor = create_camera_actor(1.0,0.05)
    cam_actor.transform(pose)
    vis.add_geometry(cam_actor)

print('[INFO] Use [ , ] to adjust the perspective !!!')
print('[INFO] Use + , - to adjust the point size !!!')
vis.run()
vis.destroy_window()
quit()
