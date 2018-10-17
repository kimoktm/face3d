''' 3d morphable model example
3dmm parameters --> mesh 
fitting: 2d image + 3dmm -> 3d face
'''
import os, sys
import subprocess
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh
from face3d import mesh_cython
from face3d.morphable_model import MorphabelModel

import cv2
from skimage import img_as_float
import glob as glob


# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
samples = 768; std = 1.2
h = w = 200; c = 3


# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
pncc = face3d.morphable_model.load.load_pncc_code('Data/BFM/Out/pncc_code.mat')
print('init bfm model success')

# load BG images
imgfiles = glob.glob(os.path.join('/home/karim/Documents/Data/Random', '*.jpg'))

save_folder = 'results/faces/validation'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)



for i in range(samples):
	b,g,r = cv2.split(cv2.imread(imgfiles[np.random.randint(len(imgfiles))]))
	BG = cv2.merge([r,g,b])
	BG = cv2.resize(BG, (w, h))
	BG = cv2.GaussianBlur(BG, (3, 3), 0)
	BG = img_as_float(BG)
	BG = BG.astype(np.float32)
	#BG = None

	# --- 2. generate face mesh: vertices(represent shape) & colors(represent texture)
	sp = bfm.get_shape_para('zero', std)
	ep = bfm.get_exp_para('random', std)
	vertices = bfm.generate_vertices(sp, ep)

	tp = bfm.get_tex_para('zero', std)
	colors = bfm.generate_colors(tp)
	colors = np.minimum(np.maximum(colors, 0), 1)

	# --- 3. transform vertices to proper position
	s = np.random.normal(0, 8, 1)
	t = np.random.normal(0, 6, 3)
	angles = np.random.normal(0, 10, 3)

	# s = 0
	# angles = np.array([0, 0, 0])
	# t = np.array([0, 0])

	pose = np.r_[angles.flatten(), t[:2].flatten(), s]
	transformed_vertices = bfm.transform(vertices, pose[-1], pose[:3], np.r_[pose[3:5], [0]])
	projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

	# --- 4. render(3d obj --> 2d image)
	# set prop of rendering

	light_intensity = np.array([[1, 1, 1]])
	light_position = np.array([[0, 0, 300]])
	lit_colors = mesh_cython.light.add_light(transformed_vertices, bfm.triangles, colors, light_position, light_intensity)
	colors = (0.8 * lit_colors + 1.2 * colors) / 2.0
	image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	image = mesh_cython.render.render_colors(image_vertices, bfm.triangles, colors, h, w, c, BG)

	# image = mesh_cython.render.render_colors(image_vertices, bfm.triangles, pncc, h, w, c)


	# --- params
	params = np.r_[ep.flatten(), pose.flatten()]
	params = np.float32(params)


	print(params)
	image = np.clip(image, 0, 1)
	io.imsave('{}/generated_{}.png'.format(save_folder, i), image)

	np.save('{}/generated_{}.npy'.format(save_folder, i), params)


# options = '-delay 20 -loop 0 -layers optimize' # gif. need ImageMagick.
# subprocess.call('convert {} {}/generated_*.png {}'.format(options, save_folder, save_folder + '/3dmm.gif'), shell=True)
