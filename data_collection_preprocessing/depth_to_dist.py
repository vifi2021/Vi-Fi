### convert the depth map in /Depth/ folder to distance map and save it to /Dist/ folder


import os
import time
from datetime import datetime
import pytz
import cv2
import pickle 
import numpy as np
import csv
import json
import collections
from functools import partial
# from flow_utils import readFlow, flow2img
import random


def convert_depth_map_to_dist_map(depth_map):
	# cx = 314.13
	# cy = 235.271
	# fx = 617.086
	# fy = 616.991

	# ZED2 [LEFT_CAM_HD]
	fx=528.365
	fy=527.925
	cx=638.925
	cy=359.2805
	k1=-0.0394311
	k2=0.00886432
	k3=-0.00481956
	p1=-0.000129881
	p2=-4.88565e-05

	# dist_map = np.zeros(depth_map.shape)
	# for yp in range(depth_map.shape[0]):
	# 	for xp in range(depth_map.shape[1]):
	# 		X = ((xp-cx)/fx)*depth_map[yp, xp]
	# 		Y = ((yp-cy)/fy)*depth_map[yp, xp]
	# 		dist_map[yp, xp] = (X**2 + Y**2 + depth_map[yp, xp]**2)**0.5
	# return dist_map
	

	# dist_map_faster = np.zeros(depth_map.shape)
	# print(depth_map.shape)
	X = np.repeat(np.array([range(depth_map.shape[0])]).T, depth_map.shape[1], axis=1)
	# print(X)
	X = ((X-cy)/fy) * depth_map

	Y = np.repeat(np.array([range(depth_map.shape[1])]), depth_map.shape[0], axis=0)
	# print(Y)
	Y = ((Y-cx)/fx) * depth_map
	
	dist_map_faster = np.sqrt((np.power(X,2) + np.power(Y,2) + np.power(depth_map,2)))
	# print(dist_map)
	# print(dist_map_faster)
	# print(np.nansum(dist_map - dist_map_faster))


	return dist_map_faster
	




# print(gc.get_threshold())

# date = '20211004'
# date = '20211006'
date = '20211007'

sequence_names = sorted([d for d in os.listdir('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/') if date in d and os.path.isdir(os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/',d))])
print(sequence_names)

for seq_id, sequence_name in enumerate(sequence_names):
	print(seq_id, sequence_name)
	# if sequence_name != '20210907_145803':
	# 	continue
		
	sequence_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', sequence_name)

	dist_save_dir = os.path.join(sequence_dir, 'Dist/')
	if not os.path.exists(dist_save_dir):
		os.mkdir(dist_save_dir)

	depth_dir = os.path.join(sequence_dir, 'Depth/')
	(dirpath, dirnames, filenames) = next(os.walk(depth_dir))
	filenames = [f for f in filenames if ".pkl" in f]
	# print(filenames)
	# exit()

	for depth_pkl_name in filenames:
		print(depth_pkl_name)
		depth_pkl_path = os.path.join(depth_dir, depth_pkl_name)
		# print(depth_pkl_path)
		if not os.path.isfile(depth_pkl_path):
			print("cannot find:", depth_pkl_path)
			continue
		f = open(depth_pkl_path, 'rb')
		depth_pkl = pickle.load(f, encoding='bytes')
		f.close()

		# print(depth_pkl.shape)
		dist_map = convert_depth_map_to_dist_map(depth_pkl)
		# print(dist_map.shape)


		dist_pkl_path = os.path.join(dist_save_dir, depth_pkl_name)
		# with open(dist_pkl_path, 'wb') as output:
		# 	pickle.dump(dist_map, output)
		# output.close()

		np.save(dist_pkl_path.replace('.pkl', '.npy'), dist_map)
		

