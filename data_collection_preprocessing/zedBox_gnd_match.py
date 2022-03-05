# this script associate gnd bbox with tracktor++ bounding boxes 



''' 
pseudo-code:

'''
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
import pandas as pd
import random
import copy
import itertools
from scipy.optimize import linear_sum_assignment

project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'
# date = '20211004'
# date = '20210907'
# date = '20211007'

sequence_names = []
# sequence_names += [d for d in os.listdir(project_dir) if '20210907' in d and os.path.isdir(os.path.join(project_dir,d))]
# sequence_names += [d for d in os.listdir(project_dir) if '20211004' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211006' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211007' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names = sorted(sequence_names)

for seq_id, sequence_name in enumerate(sequence_names):
	print(seq_id, sequence_name)
	# if sequence_name != '20211006_104255': 
		# continue

	sequence_dir = os.path.join(project_dir, sequence_name)
	zedbox_output_dir = sequence_dir + '/zedBox_' + sequence_name + '.txt'
	rgb_dir = os.path.join(sequence_dir, 'RGB/')
	vott_label_dir = os.path.join(sequence_dir, 'GND/vott-json-export/')
	if not os.path.isdir(vott_label_dir):
		print('export dir does not exist, continue')
		continue
	
	export_json_files = [d for d in os.listdir(vott_label_dir) if 'export.json' in d]
	if not export_json_files:
		print('GND for %s is not ready yet, skip' % sequence_name)
		continue
	export_json_file = export_json_files[0]
	# if not os.path.isfile(os.path.join(vott_label_dir, 'GND_'+sequence_name+'-export.json')):
	# 	print('GND for %s is not ready yet, skip' % sequence_name)
	# 	continue
	# TODO: save the tracktor++_gnd_match.txt somewhere
	# if not os.path.exists(openpose_labeled_dir):
	# 	os.makedirs(openpose_labeled_dir)
	output_file_name = 'zedBox_gnd_match.txt'
	outfile = open(os.path.join(sequence_dir, output_file_name), 'w')

	# read tracktor++ output txt file and parse the output txt
	with open(zedbox_output_dir, 'r') as tracktor_output_file:
		tracktor_output = tracktor_output_file.readlines()

	# read the RGB files and sort the file name according to timestamp
	rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))], key=lambda x: datetime.strptime(x.replace('20%', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'))

	# print(rgb_files)
	
	# prepare the tracktor_ts_bbox_dict:
	# format:
	# frame_name_containing_ts: [[bbox id, bbox_x, bbox_y, bbox_w, bbox_h], ...]
	tracktor_ts_bbox_dict = collections.OrderedDict()
	for line in tracktor_output:
		line_array = line.replace('\n', '').split(',')

		''' downsample if needed '''
		# if (int(line_array[0]) - 1) % 10 != 0: continue # comment this line if processing dataset for 20210907 data
		# print(line_array)
		# print(int(line_array[0])-1)

		# frame_id = (int(line_array[0])-1)//10 # uncomment this line for 2020 Dec dataset
		frame_id = (int(line_array[0])-1)//3 # uncomment this line for 2021 dataset
		if frame_id >= len(rgb_files):
			break
		key = rgb_files[frame_id]
		if key in tracktor_ts_bbox_dict:
			tracktor_ts_bbox_dict[rgb_files[frame_id]].append(line_array[1:6])
		else:
			tracktor_ts_bbox_dict[rgb_files[frame_id]] = [line_array[1:6]]

	# print(tracktor_ts_bbox_dict)

	### read vott label json file
	# with open(os.path.join(vott_label_dir, 'GND_'+sequence_name+'-export.json')) as json_file: 
	with open(os.path.join(vott_label_dir,export_json_file)) as json_file:
		vott_data = json.load(json_file, object_pairs_hook=collections.OrderedDict)
	# sort the labeled frames according to time
	labeled_frames = sorted(list(vott_data['assets'].values()), key=lambda x: datetime.strptime(x["asset"]["name"].replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'))

	### iterate all the rgb frames
	for t in range(len(labeled_frames)):
		# if not labeled_frames[t]['regions']: continue # if there is no bbox, go to the next frame

		# get the corresponding tracktor++ output line
		tracktor_frame_name = labeled_frames[t]['asset']['name'].replace('%20', ' ').replace('_', ':')
		# print(tracktor_frame_name)
		if tracktor_frame_name not in tracktor_ts_bbox_dict:
			continue
		tracktor_bboxes = tracktor_ts_bbox_dict[tracktor_frame_name]
		# print(tracktor_bboxes)


		img = cv2.imread(os.path.join(rgb_dir, tracktor_frame_name))

		# iterate through all the poses in one frame
		aff_mat = [[0] * len(labeled_frames[t]['regions']) for _ in range(len(tracktor_bboxes))]
		# print(aff_mat)
		line_to_write = [tracktor_frame_name]
		if not aff_mat or not aff_mat[0]: 
			# with open(os.path.join(openpose_labeled_dir, openpose_json_name), 'w') as outfile: 
			# 	json.dump(openpose_data, outfile)
			outfile.write(','.join(line_to_write)+'\n')
			continue # if there is no set of keypoints, go to the next frame

		for i in range(len(labeled_frames[t]['regions'])):
			# find the closest pose for that bbox center
			bbox_x = int(float(labeled_frames[t]['regions'][i]['boundingBox']['left']))
			bbox_y = int(float(labeled_frames[t]['regions'][i]['boundingBox']['top']))
			bbox_h = int(float(labeled_frames[t]['regions'][i]['boundingBox']['height']))
			bbox_w = int(float(labeled_frames[t]['regions'][i]['boundingBox']['width']))
			bbox_id = labeled_frames[t]['regions'][i]['tags'][0]
			center_coord = np.array([bbox_x + bbox_w/2, bbox_y + bbox_h/2])
			print(bbox_id, center_coord)

			### draw bounding box and id
			img = cv2.rectangle(img, (int(bbox_x), int(bbox_y)), (int(bbox_x)+int(bbox_w)-2, int(bbox_y)+int(bbox_h)-2), color=(0, 255, 0), thickness=2) 
			img = cv2.putText(img, str((bbox_id)), ((int(bbox_x)+int(bbox_x)+int(bbox_w)-2)//2, (int(bbox_y)+int(bbox_y)+int(bbox_h)-2)//2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=2)
			# keypoints = openpose_data['people']

			# to determin a pose's identity
			min_d = float('inf')
			min_idx = 0
			for j in range(len(tracktor_bboxes)):
				tracktor_bbox_id = int(tracktor_bboxes[j][0])
				tracktor_bbox_x = int(float(tracktor_bboxes[j][1]))
				tracktor_bbox_y = int(float(tracktor_bboxes[j][2]))
				tracktor_bbox_w = int(float(tracktor_bboxes[j][3]))
				tracktor_bbox_h = int(float(tracktor_bboxes[j][4]))
				tracktor_center_coord = np.array([tracktor_bbox_x + tracktor_bbox_w/2, tracktor_bbox_y + tracktor_bbox_h/2])
				print(tracktor_bbox_id, tracktor_center_coord)

				### draw bounding box and id
				img = cv2.rectangle(img, (int(tracktor_bbox_x), int(tracktor_bbox_y)), (int(tracktor_bbox_x)+int(tracktor_bbox_w)-2, int(tracktor_bbox_y)+int(tracktor_bbox_h)-2), color=(0, 255, 255)) 
				img = cv2.putText(img, str(int(tracktor_bbox_id)), ((int(tracktor_bbox_x)+int(tracktor_bbox_x)+int(tracktor_bbox_w)-2)//2, (int(tracktor_bbox_y)+int(tracktor_bbox_y)+int(tracktor_bbox_h)-2)//2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 255), thickness=3)

				d = np.linalg.norm((tracktor_center_coord - center_coord))
				aff_mat[j][i] = d
			# 	if d < min_d:
			# 		min_d = d
			# 		min_idx = j
			# print(min_idx)
			# openpose_data['people'][min_idx]['person_id'] = bbox_id
		print(aff_mat)
		row_ind, col_ind = linear_sum_assignment(aff_mat)
		print(row_ind, col_ind)
		
		for row_id in range(len(aff_mat)):
			tracktor_bbox_x = int(float(tracktor_bboxes[row_id][1]))
			tracktor_bbox_y = int(float(tracktor_bboxes[row_id][2]))
			tracktor_bbox_w = int(float(tracktor_bboxes[row_id][3]))
			tracktor_bbox_h = int(float(tracktor_bboxes[row_id][4]))
			tracktor_center_coord = np.array([tracktor_bbox_x + tracktor_bbox_w/2, tracktor_bbox_y + tracktor_bbox_h/2])

			if row_id not in row_ind:
				label = 'None'
			else:
				# print(col_ind[row_ind.tolist().index(row_id)])
				label = labeled_frames[t]['regions'][col_ind[row_ind.tolist().index(row_id)]]['tags'][0]
			line_to_write.append(tracktor_bboxes[row_id][0])
			line_to_write.append(label)

			img = cv2.putText(img, str(label), ((int(tracktor_bbox_x)+int(tracktor_bbox_x)+int(tracktor_bbox_w)-2)//2, (int(tracktor_bbox_y)+int(tracktor_bbox_y)+int(tracktor_bbox_h)-2)//2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0), thickness=1)


		# for row_id, col_id in zip(row_ind, col_ind):
		# 	# print(col_id)
		# 	# openpose_data['people'][col_id]['person_id'] = labeled_frames[t]['regions'][row_id]['tags'][0]

		# 	tracktor_bbox_x = int(float(tracktor_bboxes[col_id][1]))
		# 	tracktor_bbox_y = int(float(tracktor_bboxes[col_id][2]))
		# 	tracktor_bbox_w = int(float(tracktor_bboxes[col_id][3]))
		# 	tracktor_bbox_h = int(float(tracktor_bboxes[col_id][4]))
		# 	tracktor_center_coord = np.array([tracktor_bbox_x + tracktor_bbox_w/2, tracktor_bbox_y + tracktor_bbox_h/2])

		# 	### draw bounding box and id
		# 	img = cv2.putText(img, str(labeled_frames[t]['regions'][row_id]['tags'][0]), ((int(tracktor_bbox_x)+int(tracktor_bbox_x)+int(tracktor_bbox_w)-2)//2, (int(tracktor_bbox_y)+int(tracktor_bbox_y)+int(tracktor_bbox_h)-2)//2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0), thickness=1)

		# 	print(tracktor_bboxes[col_id][0], labeled_frames[t]['regions'][row_id]['tags'][0])
		# 	line_to_write.append(tracktor_bboxes[col_id][0])
		# 	line_to_write.append(labeled_frames[t]['regions'][row_id]['tags'][0])

		outfile.write(",".join(line_to_write)+'\n')
			# print(identity)
		# for i in range(len(tracktor_bboxes)):
		# 	print(openpose_data['people'][i]['person_id'])


			# write the (identity, i) pair to the copy of the original json file
		# with open(os.path.join(openpose_labeled_dir, openpose_json_name), 'w') as outfile: 
			# json.dump(openpose_data, outfile)
		# cv2.imshow('frame', img)
		# cv2.waitKey(1)

	outfile.close()

