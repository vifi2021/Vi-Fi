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
import pandas as pd
import random
import copy
import itertools

### for every ts in that list, find the nearest ts in the other list
def nearest_ts(ts_list, target_ts):
	return min(ts_list, key=lambda x: abs(x - target_ts))

### return the first k timestamp that is closest to target
def k_nearest_ts(ts_list, target_ts, k, offset):
	res = sorted(ts_list, key=lambda x: abs(x + offset - target_ts))[:k]
	return sorted(res)

project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'
sequence_names = []
sequence_names += [d for d in os.listdir(project_dir) if '202012' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20210907' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211004' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211006_14' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211006_15' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211007' in d and os.path.isdir(os.path.join(project_dir,d))]


sequence_names = sorted(sequence_names)[:]
print(sequence_names)

WINDOW_SIZE = 10 

for seq_id, sequence_name in enumerate(sequence_names):
	df_d = {}
	idx_df = 0
	# if not sequence_name in ['20211007_143810']:
	# 	continue
	# if not '20201228' in sequence_name :
	# 	continue
	# if sequence_name == '20211007_143810': continue # this sequence has more than 15 people

	print(seq_id, sequence_name)
	sequence_dir = os.path.join(project_dir, sequence_name)
	depth_dir = os.path.join(sequence_dir, 'Depth/')
	dist_dir = os.path.join(sequence_dir, 'Dist/')
	ftm_csv_dir = os.path.join(sequence_dir, 'WiFi/')
	imu_csv_dir = os.path.join(sequence_dir, 'IMU/')
	vott_label_dir = os.path.join(sequence_dir, 'GND/vott-json-export/')

	if not os.path.isdir(os.path.join(vott_label_dir)):
		print('GND folder for %s is not ready yet, skip' % sequence_name)
		continue

	gnd_json_files = [f for f in os.listdir(vott_label_dir) if f and f.endswith('.json')]
	if len(gnd_json_files) == 0:
		print('GND for %s is not ready yet, skip' % sequence_name)
		continue
	else:
		gnd_json_file_name = gnd_json_files[0]

	# read in the txt that records valid frame range
	if os.path.isfile(os.path.join(sequence_dir, 'valid_frame_range.txt')):
		with open(os.path.join(sequence_dir, 'valid_frame_range.txt'), 'r') as frame_range:
			valid_range = frame_range.readlines()[0].replace('\n', '').split(", ")
	else:
		print('did not use valid range')
		valid_range = None

	# read in the txt that logs the time offset for each participants
	if os.path.isfile(os.path.join(sequence_dir, 'time_offsets.txt')):
		with open(os.path.join(sequence_dir, 'time_offsets.txt'), 'r') as offset_txt:
			offset_lines = offset_txt.readlines()
		offset = {offset_line.split(': ')[0]: float(offset_line.split(': ')[1].replace('\n', '')) for offset_line in offset_lines}
	elif '20201228' in sequence_name:
		offset = {'Subject1': 2219.273,
				'Subject3': 2962.273,
				'Subject2': 2764.273,
				'Subject4': 3461.273,
				'Subject5': 2948.273}
	else:
		print('no time offset txt found, set as 0')
		offset = collections.defaultdict(lambda: 0)
		# continue

	# read ftm csv from all participants
	ftm_files = sorted([f for f in os.listdir(ftm_csv_dir) if os.path.isfile(os.path.join(ftm_csv_dir,f))])
	ftm_csv_dicts = {}
	for i in range(len(ftm_files)):
		name = ftm_files[i].split('_')[1].replace('.csv', '')
		with open(os.path.join(ftm_csv_dir, ftm_files[i])) as f:
			reader = csv.reader(f)
			ftm_csv_data = list(reader)
		ftm_csv_dicts[name] = {int(ftm_csv_data[_i][0]): ftm_csv_data[_i][1:] for _i in range(len(ftm_csv_data))}

	# read IMU csv from all participants
	imu_files = sorted([f for f in os.listdir(imu_csv_dir) if os.path.isfile(os.path.join(imu_csv_dir,f))])
	accel_dicts = {}
	gyro_dicts = {}
	mag_dicts = {}
	for i in range(len(imu_files)):
		name = imu_files[i].split('_')[1].replace('.csv', '')
		accel_dicts[name] = {}
		gyro_dicts[name] = {}
		mag_dicts[name] = {}
		with open(os.path.join(imu_csv_dir, imu_files[i])) as f:
			reader = csv.reader(f)
			imu_csv_data = list(reader)
		for i in range(len(imu_csv_data)):
			if 'ACCEL' in imu_csv_data[i]:
				accel_dicts[name].update({int(imu_csv_data[i][0]): imu_csv_data[i][1:]})
			if 'GYRO' in imu_csv_data[i]:
				gyro_dicts[name].update({int(imu_csv_data[i][0]): imu_csv_data[i][1:]})
			if 'MAG' in imu_csv_data[i]:
				mag_dicts[name].update({int(imu_csv_data[i][0]): imu_csv_data[i][1:]})


	### read vott label json file
	with open(os.path.join(vott_label_dir, gnd_json_file_name)) as json_file: 
		vott_data = json.load(json_file, object_pairs_hook=collections.OrderedDict)

	# sort the labeled frames according to time
	labeled_frames = sorted(list(vott_data['assets'].values()), key=lambda x: datetime.strptime(x["asset"]["name"].replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'))
	
	if valid_range:
		print('valid_range exists')
		print([datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in valid_range])
		labeled_frames = [x for x in labeled_frames if datetime.strptime(valid_range[0], '%Y-%m-%d %H:%M:%S.%f') <= datetime.strptime(x["asset"]["name"].replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f') <= datetime.strptime(valid_range[1], '%Y-%m-%d %H:%M:%S.%f')]

	# down sample the frames (original is 10fps)
	if '2021' in sequence_name:
		labeled_frames = labeled_frames[::3]

	### prepare a blank pandas data frame to hold information
	df = pd.DataFrame(columns = ['DepthFrameName', 'dist_vectors', 'ftm_vectors', 'AffinityMat']) 

	### iterate all the rgb frames
	for t in range(len(labeled_frames)):

		if not labeled_frames[t]['regions']: continue
		depth_pkl_name = labeled_frames[t]['asset']['name'].replace('_', ':').replace('%20', ' ').replace('.png', '.pkl')
		consecutive_dist_map = collections.defaultdict(partial(np.empty, (0), float))
		consecutive_ftm_map = collections.defaultdict(partial(np.empty, (0), float))

		bbox_id_pool = set()
		for i in range(len(labeled_frames[t]['regions'])):
			bbox_id = labeled_frames[t]['regions'][i]['tags'][0]
			if bbox_id == 'Others':
				bbox_id = bbox_id + '_' + labeled_frames[t]['regions'][i]['id']
			bbox_id_pool.add(bbox_id)

		### create consecutive dist map for bounding boxes for a window of frames
		for j in range(WINDOW_SIZE):
			new_t = t-j
			if new_t >= 0:
				if '2021' in sequence_name:
					pkl_name = labeled_frames[new_t]['asset']['name'].replace('_', ':').replace('%20', ' ').replace('.png', '.npy')
					depth_pkl_path = os.path.join(dist_dir, pkl_name)
					if not os.path.isfile(depth_pkl_path):
						print("cannot find:", depth_pkl_path)
						continue
					depth_pkl = np.load(depth_pkl_path)
				else:
					pkl_name = labeled_frames[new_t]['asset']['name'].replace('_', ':').replace('%20', ' ').replace('.png', '.pkl')
					depth_pkl_path = os.path.join(dist_dir, pkl_name)
					# print(depth_pkl_path)
					if not os.path.isfile(depth_pkl_path):
						print("cannot find:", depth_pkl_path)
						continue
					f = open(depth_pkl_path, 'rb')
					depth_pkl = pickle.load(f, encoding='bytes')
					f.close()
				### infer the corresponding ftm timestamp
				rgbd_linux_time = int(float(datetime.strptime(pkl_name[:-4], '%Y-%m-%d %H:%M:%S.%f').strftime('%s') + str(datetime.strptime(pkl_name[:-4], '%Y-%m-%d %H:%M:%S.%f').microsecond/1e6)[1:]) * 1e3)
				ftm_ts = {}
				for name in ftm_csv_dicts.keys():
					ftm_ts[name] = k_nearest_ts(ts_list=ftm_csv_dicts[name].keys(), target_ts=rgbd_linux_time, k=1, offset=offset[name])

				### infer the corresponding imu timestamp
				accel_ts = {}
				gyro_ts = {}
				mag_ts = {}
				for name in accel_dicts.keys():
					accel_ts[name] = k_nearest_ts(ts_list=accel_dicts[name].keys(), target_ts=rgbd_linux_time, k=1, offset=offset[name])
					gyro_ts[name] = k_nearest_ts(ts_list=gyro_dicts[name].keys(), target_ts=rgbd_linux_time, k=1, offset=offset[name])
					mag_ts[name] = k_nearest_ts(ts_list=mag_dicts[name].keys(), target_ts=rgbd_linux_time, k=1, offset=offset[name])
					if consecutive_ftm_map[name].shape[0] < 11 * WINDOW_SIZE:
						consecutive_ftm_map[name] = np.append(consecutive_ftm_map[name], [float(ftm_csv_dicts[name][ftm_ts[name][0]][2])/1000, float(ftm_csv_dicts[name][ftm_ts[name][0]][3])/1000]+accel_dicts[name][accel_ts[name][0]][1:]+gyro_dicts[name][gyro_ts[name][0]][1:]+mag_dicts[name][mag_ts[name][0]][1:])
						consecutive_ftm_map[name] = np.array(consecutive_ftm_map[name], dtype='float')	

				for i in range(len(labeled_frames[new_t]['regions'])):
					bbox_x = int(float(labeled_frames[new_t]['regions'][i]['boundingBox']['left']))
					bbox_y = int(float(labeled_frames[new_t]['regions'][i]['boundingBox']['top']))
					bbox_h = int(float(labeled_frames[new_t]['regions'][i]['boundingBox']['height']))
					bbox_w = int(float(labeled_frames[new_t]['regions'][i]['boundingBox']['width']))
					bbox_id = labeled_frames[new_t]['regions'][i]['tags'][0]
					if bbox_id == 'Others':
						bbox_id = bbox_id + '_' + labeled_frames[new_t]['regions'][i]['id']

					depth_roi = depth_pkl[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
					depth_median = np.median(depth_roi)

					if bbox_id in bbox_id_pool and consecutive_dist_map[bbox_id].shape[0] < 5 * WINDOW_SIZE:
						consecutive_dist_map[bbox_id] = np.append(consecutive_dist_map[bbox_id], [depth_median, bbox_x, bbox_y, bbox_h, bbox_w])

				if all(len(consecutive_dist_map[key]) == 5 * WINDOW_SIZE for key in consecutive_dist_map) and all(len(consecutive_ftm_map[key]) == 11 * WINDOW_SIZE for key in consecutive_ftm_map):
					break


		### prepare the affinity matrix gnd
		N = 5 # maximum 5 phone holder identities existing at the same time
		M = 15 # maximum bounding boxes existing at the same time
		
		affinity_mat = np.zeros((N+1, M+1)) # last column and last row handles incoming or disappearing identities

		for i, ftm_id in enumerate(consecutive_ftm_map):
			for j, bbox_id in enumerate(consecutive_dist_map):
				# print(i, ftm_id, j, bbox_id)
				if ftm_id == bbox_id:
					affinity_mat[i,j] = 1
		for i in range(len(consecutive_ftm_map)):
			if np.sum(affinity_mat[i]) == 0:
				affinity_mat[i, -1] = 1
		for i in range(len(consecutive_dist_map)):
			if np.sum(affinity_mat[:, i]) == 0:
				affinity_mat[-1, i] = 1



		df_d[idx_df] = {'DepthFrameName': os.path.join(sequence_name, depth_pkl_name), 'dist_vectors': consecutive_dist_map, 'ftm_vectors': consecutive_ftm_map, 'AffinityMat': affinity_mat}
		idx_df += 1

	df = pd.DataFrame.from_dict(df_d, 'index')

	df.to_csv ('affinity_gnd_'+sequence_name+'.csv', index = False, header=True)
	df.to_pickle('affinity_gnd_'+sequence_name+'.pkl')






