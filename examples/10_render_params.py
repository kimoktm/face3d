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
import glob

sys.path.append('..')
import face3d
from face3d import mesh
from face3d import mesh_cython
from face3d.morphable_model import MorphabelModel

# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')


save_folder = 'results/faces/output'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

h = w = 200; c = 3


for filename in glob.glob('{}/predicted_*.npy'.format(save_folder)):
	i = os.path.splitext(os.path.basename(filename))[0].split('_')[-1]

	params = np.load(filename)
	params = np.float32(params)
	# sp = params[: bfm.n_shape_para][:, np.newaxis]
	# tp = params[bfm.n_shape_para : bfm.n_shape_para + bfm.n_tex_para][:, np.newaxis]
	# pose = params[bfm.n_shape_para + bfm.n_tex_para:]
	ep = params[:bfm.n_exp_para][:, np.newaxis]
	pose = params[bfm.n_exp_para:]


	# --- 2. generate face mesh: vertices(represent shape) & colors(represent texture)
	sp = bfm.get_shape_para('zero')
	#ep = bfm.get_exp_para('zero')


	tp = bfm.get_tex_para('zero')
	colors = bfm.generate_colors(tp)
	colors = np.minimum(np.maximum(colors, 0), 1)

	# --- 3. transform vertices to proper position

	s = pose[-1]
	angles = pose[:3]
	t = np.r_[pose[3:5], [0]]

	vertices = bfm.generate_vertices(sp, ep)
	transformed_vertices = bfm.transform(vertices, s, angles, t)
	projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

	# --- 4. render(3d obj --> 2d image)
	# set prop of rendering
	light_intensity = np.array([[1, 1, 1]])
	light_position = np.array([[0, 0, 300]])
	lit_colors = mesh_cython.light.add_light(transformed_vertices, bfm.triangles, colors, light_position, light_intensity)
	colors = (0.8 * lit_colors + 1.2 * colors) / 2.0
	image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	image = mesh_cython.render.render_colors(image_vertices, bfm.triangles, colors, h, w, c)



	print(pose)
	image = np.clip(image, 0, 1)

	# print('pose, groudtruth: \n', s, angles[0], angles[1], angles[2], t[0], t[1])
	io.imsave('{}/predicted_{}.jpg'.format(save_folder, i), image)

options = '-delay 20 -loop 0 -layers optimize' # gif. need ImageMagick.
subprocess.call('convert {} {}/predicted_*.jpg {}'.format(options, save_folder, save_folder + '/3dmm.gif'), shell=True)
