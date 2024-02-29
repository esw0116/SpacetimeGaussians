import glob, os
from PIL import Image

import numpy as np
import open3d as o3d
from tqdm import tqdm
import cv2


resolution = 4
input_dir = '../dataset/Neural3D/cook_spinach'

input_folder_list = tqdm(sorted(glob.glob(os.path.join(input_dir, 'colmap_*'))))


for input_folder in input_folder_list:
    print(f'Processing {input_folder}')
    image_folder = os.path.join(input_folder, 'images')
    depth_folder = os.path.join(input_folder, 'depths')

    img_path = os.path.join(image_folder, 'cam00.png')
    depth_path = os.path.join(depth_folder, 'cam00.npy')

    img = Image.open(img_path)
    depth = np.load(depth_path).clip(0, 1)

    W, H = img.size
    img = img.resize((W//resolution, H//resolution), Image.BICUBIC)
    depth = cv2.resize(depth, (W//resolution, H//resolution), interpolation=cv2.INTER_CUBIC)
    z = depth * 4 + 2

    W, H = img.size
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    Fx, Fy = 0.5*W*np.tan(0.414), 0.5*H*np.tan(0.414)
    Cx, Cy = W/2, H/2
    K = np.array([[Fx, 0., Cx],
                [0., Fy, Cy],
                [0.,  0.,  1.]]).astype(np.float32)

    pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*z, y*z, 1*z), axis=0).reshape(3,-1))
    pts_color_cam = np.array(img).reshape(-1,3).astype(np.float32)/255.

    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pts_coord_cam.transpose(1,0))
    pcd_o3d.colors = o3d.utility.Vector3dVector(pts_color_cam)
    o3d.io.write_point_cloud(os.path.join(input_folder, "blender00.ply"), pcd_o3d)
