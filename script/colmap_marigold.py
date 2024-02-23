import os, glob
from PIL import Image
import numpy as np
import tqdm

from Marigold.marigold_pipeline import MarigoldPipeline
from marigold_utils import colorize, Marigold_estimation



input_dir = '../dataset/Neural3D/cook_spinach'

input_folder_list = sorted(glob.glob(os.path.join(input_dir, 'colmap_*')))

marigold_model = MarigoldPipeline.from_pretrained('pretrained/Marigold_v1_merged').to('cuda')

for input_folder in input_folder_list:
    print(f'Processing {input_folder}')
    image_folder = os.path.join(input_folder, 'images')
    depth_folder = os.path.join(input_folder, 'depths')
    depthimg_folder = os.path.join(input_folder, 'depth_imgs')
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)
    if not os.path.exists(depthimg_folder):
        os.makedirs(depthimg_folder)

    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    tqdm_images = tqdm.tqdm(image_files, ncols=80)
    for img_file in tqdm_images:
        img_name = os.path.basename(img_file)
        img = Image.open(img_file)
        depth_marigold = Marigold_estimation(marigold_model, img)
        np.save(os.path.join(depth_folder, img_name[:-4]+'.npy'), depth_marigold)

        color_marigold = colorize(depth_marigold)
        Image.fromarray(color_marigold).save(os.path.join(depthimg_folder, img_name))
