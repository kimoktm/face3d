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

# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')


save_folder = 'results/faces/test'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

h = w = 200; c = 3

for i in range(10):
	# --- 2. generate face mesh: vertices(represent shape) & colors(represent texture)
	sp = bfm.get_shape_para('zero')
	ep = bfm.get_exp_para('zero')
	vertices = bfm.generate_vertices(sp, ep)

	tp = bfm.get_tex_para('zero')
	colors = bfm.generate_colors(tp)
	colors = np.minimum(np.maximum(colors, 0), 1)

	# --- 3. transform vertices to proper position
	s = 8e-04
	angles = np.array([0, 0, 0])
	t = np.array([0, 0, 0])
	# t = np.random.rand(3, 1)*1e02*0.2

	# IMPORTANT: Multiply by 2 for large dataset
	t = np.random.uniform(-15, 15, 3)
	angles = np.random.uniform(-10, 10, 3)
	transformed_vertices = bfm.transform(vertices, s, angles, t)
	projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

	# --- 4. render(3d obj --> 2d image)
	# set prop of rendering
	image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	image = mesh_cython.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

	pose = np.r_[angles.flatten(), t[:2].flatten(), s]
	pose = np.float32(pose)

	print('pose, groudtruth: \n', s, angles[0], angles[1], angles[2], t[0], t[1])
	io.imsave('{}/generated_{}.jpg'.format(save_folder, i), image)
	np.save('{}/rot_{}.npy'.format(save_folder, i), angles.flatten())
	np.save('{}/trn_{}.npy'.format(save_folder, i), t[:2])
	np.save('{}/params_{}.npy'.format(save_folder, i), pose)

options = '-delay 20 -loop 0 -layers optimize' # gif. need ImageMagick.
subprocess.call('convert {} {}/generated_*.jpg {}'.format(options, save_folder, save_folder + '/3dmm.gif'), shell=True)
