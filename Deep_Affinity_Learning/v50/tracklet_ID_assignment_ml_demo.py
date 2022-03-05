# assign phone ID to the tracktor++ tracklets based on bipartite matching (linear assignment)
import os
import sys
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
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.spatial import distance
import torch
sys.path.insert(1, '/home/hans/Documents/Reality_Aware_Networks/multimodal_learning/v50/')
import myModel_v50
from copy import deepcopy
register_matplotlib_converters()


class TrackletNode(object):
	"""docstring for TrackletNode"""
	def __init__(self, arg):
		super(TrackletNode, self).__init__()
		self.arg = arg
		
### return the first k timestamp that is closest to target
def k_nearest_ts(ts_list, target_ts, k, offset):
	res = sorted(ts_list, key=lambda x: abs(x + offset - target_ts))[:k]
	return sorted(res)

plt.rcParams['font.size'] = '20'

# For ablation study
MODE = 'FTM+IMU' # milestone 54, 57, 60, 63(all train)
# MODE = 'FTM' # milestone 55, 58, 61
# MODE = 'IMU' # milestone 56, 59, 62

### load the pre-trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("detected device: ", device)

criterion = myModel_v50.AffinityLoss_test().cuda()
# criterion = myModel_v50.AffinityLoss_test_bipartite().cuda()
# exit()



Nm_phone = 5
Nm_camera = 15
acc_results = []


project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'
sequence_names = []
# sequence_names += [d for d in os.listdir(project_dir) if '202012' in d and 'pocket' not in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20210907' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211004' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211006' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names += [d for d in os.listdir(project_dir) if '20211007' in d and os.path.isdir(os.path.join(project_dir,d))]
sequence_names = sorted(sequence_names)

OPENCV_VISUALIZATION = 0
print(sequence_names)
df_index = []

final_correct_count = 0
final_total_count = 0
final_correct_count_post_processed = collections.defaultdict(lambda : 0, {})
final_total_count_post_processed = collections.defaultdict(lambda : 0, {})
overall_acc_post_processed = {}
acc_record = []
vote_acc_record = collections.defaultdict(list)
total_seqlen_acc_dict = collections.defaultdict(partial(np.empty, (0), float)) # record accuracy for every timestamp
numPeople_cnts_dict = collections.defaultdict(lambda: [0, 0, 0, 0]) # the map of {number of people: [frame count, correct_prediction_count, total_prediction_count, overall_acc]}

sequence_names_in_use = []



for seq_id, sequence_name in enumerate(sequence_names):
	
	# if not sequence_name in ['20210907_150313', '20211004_153212', '20211006_142625', '20211007_110306']:
	# if not sequence_name in ['20210907_150313', '20211004_143157', '20211006_142225', '20211007_110306', '20211007_141942']: # maybe use this to report
	if not sequence_name in ['20210907_145202', '20211004_142306', '20211006_152208', '20211007_105924', '20211007_134632']: # split 2
		continue

	# test on indoor
	# if not sequence_name in ['20201223_140951']:
	# 	continue
	

	print('\n')
	print(seq_id, sequence_name)

	sequence_names_in_use.append(sequence_name)

	if '202012' in sequence_name:
		phoneID_mapping = {'Subject1': 'A', 'Subject2': 'B', 'Subject3': 'C', 'Subject4': 'D', 'Subject5': 'E', 'None': 'N/A'}
	else:
		phoneID_mapping = {'Subject1': 'A', 'Subject6': 'B', 'Subject7': 'C', 'Subject4': 'D', 'Subject5': 'E', 'Others': 'Others', 'None': 'N/A'}

	# milestone_dir = os.path.join('/media/hans/SamsungDisk/multimodal_learning_milestones/', 'milestone90')
	# milestone_file = 'epoch_60.pth'

	# milestone_dir = os.path.join('/media/hans/SamsungDisk/multimodal_learning_milestones/', 'milestone90_split_v1')
	# milestone_file = 'epoch_60.pth'
	# milestone_file = 'best_so_far_epoch_32_acc_0.8937042.pth' # maybe use this to report in the paper

	milestone_dir = os.path.join('/media/hans/SamsungDisk/multimodal_learning_milestones/', 'milestone91_split_v2')
	milestone_file = 'epoch_60.pth'
	
	
	# print(milestone_file)
	model = torch.load(os.path.join(milestone_dir, milestone_file))
	model.eval()

	sequence_dir = os.path.join(project_dir, sequence_name)
	tracker_output_dir = os.path.join(sequence_dir, 'zedBox_'+sequence_name+'.txt')
	rgb_dir = os.path.join(sequence_dir, 'RGB/')
	depth_dir = os.path.join(sequence_dir, 'Dist/')
	ftm_csv_dir = os.path.join(sequence_dir, 'WiFi/')
	imu_csv_dir = os.path.join(sequence_dir, 'IMU/')
	vott_label_dir = os.path.join(sequence_dir, 'GND/vott-json-export/')

	# if '202012' in sequence_name:
	# 	if not os.path.isfile(os.path.join(vott_label_dir, 'GND_'+sequence_name+'-export.json')):
	# 		print('GND for %s is not ready yet, skip' % sequence_name)
	# 		continue
	# else:	
	# 	if not os.path.isfile(os.path.join(vott_label_dir, 'RAN_'+sequence_name+'-export.json')):
	# 		print('GND for %s is not ready yet, skip' % sequence_name)
	# 		continue
	
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
		print('no time offset txt found, use 0 offset')
		offset = collections.defaultdict(lambda: 0)

	tracker_gnd_filename = 'zedBox_gnd_match.txt'
	with open(os.path.join(sequence_dir, tracker_gnd_filename), 'r') as gnd_file:
		gnd_lines = gnd_file.readlines()
	# print(gnd_lines)

	gnd_ts_match_dict = {gnd_line.replace('\n', '').split(',')[0][:-4]: gnd_line.replace('\n', '').split(',')[1:] for gnd_line in gnd_lines}
	# print(gnd_ts_match_dict)
	# exit()

	df_index.append(sequence_name)

	# read ftm csv from all participants
	ftm_files = sorted([f for f in os.listdir(ftm_csv_dir) if os.path.isfile(os.path.join(ftm_csv_dir,f))])
	# col_id_dict = {0: 'Subject3', 1: 'Subject1', 2: 'Subject2'}
	# col_id_dict = {}
	ftm_csv_dicts = {}
	for i in range(len(ftm_files)):
		name = ftm_files[i].split('_')[1].replace('.csv', '')
		# col_id_dict[int(ftm_files[i].split('_')[0].replace('pixel', ''))-1] = name
		# print(name)
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
		name = ftm_files[i].split('_')[1].replace('.csv', '')
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

	# read tracktor++ output txt file and parse the output txt
	with open(tracker_output_dir, 'r') as tracktor_output_file:
		tracktor_output = tracktor_output_file.readlines()


	# read the RGB files and sort the file name according to timestamp
	rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))], key=lambda x: datetime.strptime(x.replace('20%', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'))


	# tracktor_ts_bbox_dict = collections.OrderedDict()
	tracktor_ts_bbox_dict = {}
	for line in tracktor_output:
		line_array = line.replace('\n', '').split(',')

		''' downsample if needed '''
		# if (int(line_array[0]) - 1) % 10 != 0: continue
		# print(int(line_array[0])-1)

		# frame_id = int(line_array[0])-1
		if '202012' in sequence_name:
			frame_id = (int(line_array[0])-1)//10
		else:
			frame_id = (int(line_array[0])-1)//3 # uncomment this line for 2021 dataset, frame_id is the rgb frame id at 10hz
		
		if frame_id >= len(rgb_files):
			break
		# if frame_id % 3 != 0:
		# 	continue
		# print(frame_id)

		key = rgb_files[frame_id]
		if valid_range:
			if datetime.strptime(valid_range[0], '%Y-%m-%d %H:%M:%S.%f') <= datetime.strptime(key.replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f') <= datetime.strptime(valid_range[1], '%Y-%m-%d %H:%M:%S.%f'):
				if key in tracktor_ts_bbox_dict:
					tracktor_ts_bbox_dict[key].append(line_array[1:])
				else:
					tracktor_ts_bbox_dict[key] = [line_array[1:]]
		else:
			print('no valid frame range, use all the frames as is')
			if key in tracktor_ts_bbox_dict:
				tracktor_ts_bbox_dict[key].append(line_array[1:])
			else:
				tracktor_ts_bbox_dict[key] = [line_array[1:]]


	if '2021' in sequence_name:
		tracktor_ts_bbox_dict = collections.OrderedDict(sorted(tracktor_ts_bbox_dict.items())[::3])
	print(len(tracktor_ts_bbox_dict.keys()))

	### construct tracklet dictionary
	# format:
	# tracklet_ID: [[ts, bbox_coord, position], [ts, bbox_coord, position], ...]
	tracklet_dict = collections.OrderedDict()
	for openpose_json_name, bboxes in tracktor_ts_bbox_dict.items():
		# read openpose json file
		# with open(os.path.join(pose_label_dir, openpose_json_name)) as openpose_json_file: 
		# 	openpose_data = json.load(openpose_json_file, object_pairs_hook=collections.OrderedDict)
		for bbox in bboxes:
			# print(bbox)
			bbox_id = bbox[0]
			bbox_x = int(float(bbox[1]))
			bbox_y = int(float(bbox[2]))
			bbox_w = int(float(bbox[3]))
			bbox_h = int(float(bbox[4]))

			# find the 3D position for this bbox_id
			position = np.array(list(map(float, bbox[-1][1:-1].split())))
			# print(position)

			bbox_color = (0, 255, 255)

			if bbox_id in tracklet_dict:
				tracklet_dict[bbox_id].append([datetime.strptime(openpose_json_name.replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'), [bbox_x, bbox_y, bbox_w, bbox_h], position, bbox_color])
			else:
				tracklet_dict[bbox_id] = [[datetime.strptime(openpose_json_name.replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'), [bbox_x, bbox_y, bbox_w, bbox_h], position, bbox_color]]

	# print('there are in total %d tracklets' % len(tracklet_dict))

	for track_id in tracklet_dict:
		tracklet_dict[track_id] = sorted(tracklet_dict[track_id], key=lambda x: x[0])
	# print(tracklet_dict)

	tracklet_assignment = collections.defaultdict(list)
	# perform tracklet id assignment for every timestamp from tracktor++ output
	seqlen_acc_dict = collections.defaultdict(partial(np.empty, (0), float)) # record accuracy for every timestamp
	seqlen_acc_dict_total = {} # record the total count of correct predicted association and total count of association needed
	accuracy_list = []
	len_list = []
	ts_list = []	

	# fig, ax = plt.subplots(1, 1)
	# plt.tight_layout()
	# ax_twin=ax.twinx()
	start_time = time.time()
	frame_count = 0

	### traverse all the frames
	for f_i, openpose_json_name in enumerate(tracktor_ts_bbox_dict):
		# print(openpose_json_name)
		frame_count += 1
		
		img = cv2.imread(os.path.join(rgb_dir, openpose_json_name))

		depth_pkl_name = openpose_json_name.replace('%20', ' ').replace('_keypoints.json', '.png').replace('_', ':').replace('.png', '.npy')

		if '202012' in sequence_name:
			depth_pkl_name = depth_pkl_name.replace('.npy', '.pkl')

		depth_pkl_path = os.path.join(depth_dir, depth_pkl_name)
		if not os.path.isfile(depth_pkl_path):
			print("cannot find:", depth_pkl_path)
			continue
		
		if '202012' in sequence_name:
			f = open(depth_pkl_path, 'rb')
			depth_pkl = pickle.load(f, encoding='bytes')
			f.close()
		else:
			depth_pkl = np.load(depth_pkl_path)

		ts = datetime.strptime(openpose_json_name.replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f')
		
		tracklet_id_candidates = [tracklet_id for tracklet_id, tracklet in tracklet_dict.items() if  ts in [node[0] for node in tracklet]][:]
		# print('camera side participants:', tracklet_id_candidates)
		numPeople_cnts_dict[len(tracklet_id_candidates)][0] += 1
		

		### construct affinity matrix gnd
		### get tracktor box gnd label
		gnd_match = gnd_ts_match_dict[openpose_json_name[:-4]]
		# print(gnd_match)
		if not gnd_match: continue # skip the empty frame or invalid prediction
		# print(gnd_match)
		gnd_match_dict = {gnd_match[i]: gnd_match[i+1] for i in range(0, len(gnd_match), 2)}
		# print('tracktor gnd:', gnd_match_dict)

		### find the shortest/longest sequence, and use that length to construct both depth and ftm vectors
		longest = float('-inf')
		shortest = float('inf')
		longest_id = None
		for row, tracklet_id in enumerate(tracklet_id_candidates):
			# depth_vec = []
			# ftm_vec = collections.defaultdict(list)
			tracklet_nodes = [_ for _ in tracklet_dict[tracklet_id] if _[0] <= ts][::-1]
			if len(tracklet_nodes) > longest:
				longest_id = tracklet_id
				longest = len(tracklet_nodes)
			if len(tracklet_nodes) < shortest:
				shortest = len(tracklet_nodes)
		# print('longest length', longest)
		longest = 10
		# if openpose_json_name == '2020-12-28 16:45:11.368360.png':
		# 	exit()

		### construct depth vector for every tracklet candidate
		depth_vec = collections.defaultdict(partial(np.empty, (0), float))
		for row, tracklet_id in enumerate(tracklet_id_candidates):
			# depth_vec = []
			# ftm_vec = collections.defaultdict(list)
			tracklet_nodes = [_ for _ in tracklet_dict[tracklet_id] if _[0] <= ts][::-1]
			if len(tracklet_nodes) >= 10:
				tracklet_nodes = tracklet_nodes[:10]
			length = 0

			for node in tracklet_nodes:
				if length >= longest:
					break
				t = node[0].strftime('%Y-%m-%d %H:%M:%S.%f')
				bbox_x = node[1][0]
				bbox_y = node[1][1]
				bbox_w = node[1][2]
				bbox_h = node[1][3]
				
				# print(t)
				# construct depth vector according to timestamps of the tracklet
				depth_pkl_name = (t+'.png').replace('.png', '.npy')
				if '202012' in sequence_name:
					depth_pkl_name = depth_pkl_name.replace('.npy', '.pkl')

				depth_pkl_path = os.path.join(depth_dir, depth_pkl_name)
				if not os.path.isfile(depth_pkl_path):
					print("cannot find:", depth_pkl_path)
					exit()
					continue
				if '202012' in sequence_name:
					f = open(depth_pkl_path, 'rb')
					depth_pkl = pickle.load(f, encoding='bytes')
					f.close()
				else:
					depth_pkl = np.load(depth_pkl_path)

				depth_roi = depth_pkl[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
				depth_median = np.median(depth_roi)

				depth_vec[tracklet_id] = np.append(depth_vec[tracklet_id], [depth_median, bbox_x, bbox_y, bbox_h, bbox_w])
				length += 1
		# print("number of camera participants:", len(depth_vec))
		depth_vec_list = [[key, len(v)//5] for key, v in depth_vec.items()]
		depth_vec_list.append(["None", -1])
		# print('length of sequences for camera participants', depth_vec_list)

		### construct ftm vectors using the timestamps from the longest depth vector
		ftm_vec = collections.defaultdict(partial(np.empty, (0), float))
		# print("longest_id", longest_id)
		# print("length of sequence for longest_id",len([_ for _ in tracklet_dict[longest_id] if _[0] <= ts]))
		tracklet_nodes = [_ for _ in tracklet_dict[longest_id] if _[0] <= ts][::-1]
		if len(tracklet_nodes) >= 10:
			tracklet_nodes = tracklet_nodes[:10]
		for node in tracklet_nodes:
			t = node[0].strftime('%Y-%m-%d %H:%M:%S.%f')

			# construct ftm vectors based on the timestamps of the trackes
			linux_time = int(float(datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f').strftime('%s') + str(datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f').microsecond/1e6)[1:]) * 1e3)
			
			ftm_ts = {}
			accel_ts = {}
			gyro_ts = {}
			mag_ts = {}
			for name in ftm_csv_dicts.keys():
				ftm_ts[name] = k_nearest_ts(ts_list=ftm_csv_dicts[name].keys(), target_ts=linux_time, k=1, offset=offset[name])
				accel_ts[name] = k_nearest_ts(ts_list=accel_dicts[name].keys(), target_ts=linux_time, k=1, offset=offset[name])
				gyro_ts[name] = k_nearest_ts(ts_list=gyro_dicts[name].keys(), target_ts=linux_time, k=1, offset=offset[name])
				mag_ts[name] = k_nearest_ts(ts_list=mag_dicts[name].keys(), target_ts=linux_time, k=1, offset=offset[name])
				# ftm_vec[name].append(float(ftm_csv_dicts[name][ftm_ts[name][0]][2])/1000)
				if MODE == 'FTM+IMU':
					ftm_vector_dim = 11
					ftm_vec[name] = np.append(ftm_vec[name], [float(ftm_csv_dicts[name][ftm_ts[name][0]][2])/1000, float(ftm_csv_dicts[name][ftm_ts[name][0]][3])/1000]+accel_dicts[name][accel_ts[name][0]][1:]+gyro_dicts[name][gyro_ts[name][0]][1:]+mag_dicts[name][mag_ts[name][0]][1:])
				elif MODE == 'FTM':
					ftm_vector_dim = 2
					ftm_vec[name] = np.append(ftm_vec[name], [float(ftm_csv_dicts[name][ftm_ts[name][0]][2])/1000, float(ftm_csv_dicts[name][ftm_ts[name][0]][3])/1000])
				elif MODE == 'IMU':
					ftm_vector_dim = 9
					ftm_vec[name] = np.append(ftm_vec[name], accel_dicts[name][accel_ts[name][0]][1:]+gyro_dicts[name][gyro_ts[name][0]][1:]+mag_dicts[name][mag_ts[name][0]][1:])
				else:
					print('Wrong MODE')
					exit()
				ftm_vec[name] = np.array(ftm_vec[name], dtype='float')	
		# print("number of ftm participants:", len(ftm_vec))
		ftm_vec_list = [[key, len(v)//ftm_vector_dim] for key, v in ftm_vec.items()]

		if '2021' in sequence_name:
			for e in range(Nm_phone-len(ftm_vec)):
				ftm_vec_list.append(['None', -1])
			ftm_vec_list.append(["Others", -1])
		else:
			ftm_vec_list.append(["None", -1])
		# print('length of sequences for ftm participants', ftm_vec_list)
		# print(depth_vec)
		# print(gnd_match_dict)
		### prepare the affinity matrix gnd
		affinity_mat = np.zeros((Nm_phone+1, Nm_camera+1)) # last column and last row handles incoming or disappearing identities
		for i, ftm_id in enumerate(ftm_vec):
			for j, bbox_id in enumerate(depth_vec):
				# print(j, bbox_id)
				bbox_id = gnd_match_dict[bbox_id]
				# print(i, ftm_id, j, bbox_id)
				if ftm_id == bbox_id:
					affinity_mat[i,j] = 1
		for i in range(len(ftm_vec)):
			if np.sum(affinity_mat[i]) == 0:
				affinity_mat[i, -1] = 1
		for i in range(len(depth_vec)):
			if np.sum(affinity_mat[:, i]) == 0:
				affinity_mat[-1, i] = 1
		# print('affinity_mat\n', affinity_mat)

		### construct x_v and x_f and their masks
		depth_vectors = np.array(list(depth_vec.values()))
		depth_vectors_lengths = np.array([len(x) for x in depth_vectors]).reshape(-1, 1)/5
		# print(depth_vectors_lengths)
		if depth_vectors.dtype == 'object':
			### pad the depth vectors to uniform seq length
			for i in range(len(depth_vectors)):
				depth_vectors[i] = np.pad(depth_vectors[i], (0, 5*10 - len(depth_vectors[i])),'constant')
			depth_vectors = np.array([v for v in depth_vectors])
		else:
			depth_vectors = np.pad(depth_vectors, ((0, 0), (0, 5*10 - depth_vectors.shape[1])), 'constant')
		# append the valid length to the end of each vector
		depth_vectors = np.append(depth_vectors, depth_vectors_lengths, axis=1)

		depth_vectors_mask = np.array([True] * depth_vectors.shape[0])
		# print('depth_vectors shape', depth_vectors.shape)
		# print(depth_vectors)
		# print(depth_vectors_mask)
		ftm_vectors = np.array(list(ftm_vec.values()))
		ftm_vectors_lengths = np.array([len(x)for x in ftm_vectors]).reshape(-1, 1)/ftm_vector_dim
		if ftm_vectors.dtype == 'object':
			### pad the depth vectors to uniform seq length
			for i in range(len(ftm_vectors)):
				ftm_vectors[i] = np.pad(ftm_vectors[i], (0, ftm_vector_dim*10 - len(ftm_vectors[i])),'constant')
			ftm_vectors = np.array([v for v in ftm_vectors])
		else:
			ftm_vectors = np.pad(ftm_vectors, ((0, 0), (0, ftm_vector_dim*10 - ftm_vectors.shape[1])), 'constant')
		# append the valid length to the end of each vector
		ftm_vectors = np.append(ftm_vectors, ftm_vectors_lengths, axis=1)
		ftm_vectors_mask = np.array([True] * ftm_vectors.shape[0])
		# print('ftm_vectors shape', ftm_vectors.shape)
		# print(ftm_vectors_mask)

		### expand the number of depth/ftm vectors to Nm
		depth_vectors_mask = np.concatenate((depth_vectors_mask, np.array([False]*(Nm_camera-depth_vectors.shape[0]))))
		depth_vectors = np.vstack((depth_vectors, np.ones((Nm_camera - depth_vectors.shape[0], depth_vectors.shape[1]))))
		ftm_vectors_mask = np.concatenate((ftm_vectors_mask, np.array([False]*(Nm_phone-ftm_vectors.shape[0]))))
		ftm_vectors = np.vstack((ftm_vectors, np.ones((Nm_phone - ftm_vectors.shape[0], ftm_vectors.shape[1]))))
		### append extra cell for masks
		ftm_vectors_mask = np.concatenate((ftm_vectors_mask, np.array([True])))
		depth_vectors_mask = np.concatenate((depth_vectors_mask, np.array([True])))

		# print('after expanding')
		# print('depth_vectors shape',depth_vectors.shape)
		# print(depth_vectors)
		# print(depth_vectors_mask)
		# print('ftm_vectors shape', ftm_vectors.shape)
		# print(ftm_vectors_mask)

		x_v = torch.from_numpy(depth_vectors)
		x_f = torch.from_numpy(ftm_vectors)
		x_v_mask = torch.from_numpy(depth_vectors_mask)
		x_f_mask = torch.from_numpy(ftm_vectors_mask)
		affinity_mat = torch.from_numpy(affinity_mat)

		# add batch dimension
		x_v = x_v.view(1, x_v.size(0), x_v.size(1))
		x_f = x_f.view(1, x_f.size(0), x_f.size(1))
		x_v_mask = x_v_mask.view(1, x_v_mask.size(0))
		x_f_mask = x_f_mask.view(1, x_f_mask.size(0))
		affinity_mat = affinity_mat.view(1, affinity_mat.size(0), affinity_mat.size(1))

		# to cuda
		x_v = x_v.to(device, dtype=torch.float)
		x_f = x_f.to(device, dtype=torch.float)
		x_v_mask = x_v_mask.cuda()
		x_f_mask = x_f_mask.cuda()
		affinity_mat = affinity_mat.to(device, dtype=torch.float)
		# print(affinity_mat)

		# print('torch cuda tensor')
		# print(x_v.size())
		# print(x_f.size())
		# print(x_v_mask.size())
		# print(x_f_mask.size())

		# hv = (torch.zeros(2*2, x_v.size(0), 32).cuda(), torch.zeros(2*2, x_v.size(0), 32).cuda())
		# hf = (torch.zeros(2*2, x_v.size(0), 32).cuda(), torch.zeros(2*2, x_v.size(0), 32).cuda())
		output = model(x_v, x_f, x_v_mask, x_f_mask)

		# print(output)

		# loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = criterion(output, affinity_mat, x_f_mask, x_v_mask)
		# ftm_to_camera, ftm_to_camera_gnd, camera_to_ftm, camera_to_ftm_gnd, correct_count, total_count, accuracy = criterion(output, affinity_mat, x_f_mask, x_v_mask)
		[fc_association, fc_association_gnd, 
		cf_association, cf_association_gnd, 
		correct_count_fc, total_count_fc, accuracy_fc, 
		correct_count_cf, total_count_cf, accuracy_cf,
		accuracy] = criterion(output, affinity_mat, x_f_mask, x_v_mask)

		# print(cf_association)
		# print(cf_association_gnd)
		# # use both cf and fc acc
		# correct_count = correct_count_cf + correct_count_fc
		# total_count = total_count_cf + total_count_fc
		# accuracy_list.append(accuracy)

		# use only cf acc
		correct_count = correct_count_cf 
		total_count = total_count_cf 
		ts_list.append(ts)
		accuracy_list.append(accuracy_cf)	
		numPeople_cnts_dict[len(tracklet_id_candidates)][1] += correct_count
		numPeople_cnts_dict[len(tracklet_id_candidates)][2] += total_count

		# # use only fc acc
		# correct_count = correct_count_fc 
		# total_count = total_count_fc 
		# ts_list.append(ts)
		# accuracy_list.append(accuracy_fc)		

		# print(accuracy)
		# print('ftm_to_camera:', fc_association)
		# print('ftm_to_camera_gnd:', fc_association_gnd)
		# print('camera_to_ftm:', cf_association)
		# print('camera_to_ftm_gnd:', cf_association_gnd)

		### opencv visualization
		for row, tracklet_id in enumerate(tracklet_id_candidates):
			# print(row, tracklet_id)
			tracklet_nodes = [_ for _ in tracklet_dict[tracklet_id] if _[0] <= ts][::-1]
			node = tracklet_nodes[0]
			bbox_x = node[1][0]
			bbox_y = node[1][1]
			bbox_w = node[1][2]
			bbox_h = node[1][3]
			bbox_pos = node[-2]

			# print(ftm_vec_list)
			# print(cf_association[row])
			# print(ftm_vec_list[cf_association[row]][0])
			tracklet_assignment[tracklet_id].append([ts, ftm_vec_list[cf_association[row]][0], gnd_match_dict[tracklet_id]])

			# for calculate pairwise distance
			for row2, tracklet_id2 in enumerate(tracklet_id_candidates):
				if row >= row2: continue
				tracklet_nodes2 = [_ for _ in tracklet_dict[tracklet_id2] if _[0] <= ts][::-1]
				node2 = tracklet_nodes2[0]
				bbox2_x = node2[1][0]
				bbox2_y = node2[1][1]
				bbox2_w = node2[1][2]
				bbox2_h = node2[1][3]
				bbox_pos2 = node2[-2]
				# print(bbox_pos, bbox_pos2)
				social_d = np.linalg.norm(bbox_pos-bbox_pos2)
				# print('social_d between %s and %s: %f' % (tracklet_id, tracklet_id2, social_d))
				social_d_thresh = 1.8288 # 6ft
				if social_d < social_d_thresh:
					img = cv2.rectangle(img, (int(bbox_x), int(bbox_y)), (int(bbox_x)+int(bbox_w)-2, int(bbox_y)+int(bbox_h)-2), color=(0, 0, 255), thickness=3) 
					img = cv2.rectangle(img, (int(bbox2_x), int(bbox2_y)), (int(bbox2_x)+int(bbox2_w)-2, int(bbox2_y)+int(bbox2_h)-2), color=(0, 0, 255), thickness=3) 

			img = cv2.rectangle(img, (int(bbox_x), int(bbox_y)), (int(bbox_x)+int(bbox_w)-2, int(bbox_y)+int(bbox_h)-2), color=(170,150,30), thickness=2) 
			img = cv2.putText(img, tracklet_id, ((int(bbox_x)+int(bbox_x)+int(bbox_w)-2)//2, (int(bbox_y)+int(bbox_y)+int(bbox_h)-2)//2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 255), thickness=2)
			img = cv2.putText(img, phoneID_mapping[gnd_match_dict[tracklet_id]],  (int(bbox_x), (int(bbox_y)+int(bbox_h))), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
			if gnd_match_dict[tracklet_id] == ftm_vec_list[cf_association[row]][0]:
				id_color = (30, 255, 255)
			else:
				id_color = (0, 0, 255)
			img = cv2.putText(img, phoneID_mapping[ftm_vec_list[cf_association[row]][0]], (int(bbox_x), int(bbox_y)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=id_color, thickness=3) # 255, 128, 0 orange

		# print(tracklet_assignment)

		seqlen_acc_dict[min(10, shortest)] = np.append(seqlen_acc_dict[min(10, shortest)], accuracy.cpu())
		total_seqlen_acc_dict[min(10, shortest)] = np.append(total_seqlen_acc_dict[min(10, shortest)], accuracy.cpu())

		if min(10, shortest) not in seqlen_acc_dict_total:
			seqlen_acc_dict_total[min(10, shortest)] = [correct_count.cpu(), total_count.cpu()]
		else:
			seqlen_acc_dict_total[min(10, shortest)][0] += correct_count.cpu().long()
			seqlen_acc_dict_total[min(10, shortest)][1] += total_count.cpu().long()

		# len_list.append(min(10, shortest))
		# ax.clear()
		# ax.plot(ts_list, accuracy_list, '-*')
		# # ax.plot(ts_list, len_list, )
		# ax.set_xlabel('Timestamp')
		# ax.set_ylabel("Association accuracy per frame")
		# ax_twin.clear()		
		# ax_twin.plot(ts_list, len_list, 'g')
		# ax_twin.set_ylabel('minimum sequence length per frame')

		# # uncomment to enable opencv debug
		# plt.pause(0.01)
		if OPENCV_VISUALIZATION:
			cv2.imshow('demo', img)
			cv2.waitKey(0)
	elapsed = time.time() - start_time
	print(elapsed)
	print('Elapsed: %s, %d frames' % (elapsed, frame_count))
	print("AVG process time per frame: %f" % (float(elapsed)/frame_count))
	print("avg frame rate: %s" % (frame_count/float(elapsed)))


	seqlen_acc_dict = sorted([[key, value] for key, value in seqlen_acc_dict.items()], key=lambda x: x[0])
	# seqlen_acc_dict_total = sorted([[key, value] for key, value in seqlen_acc_dict_total.items()], key=lambda x : x[0])

	### calc acc for k >= 10
	# acc_for_k_is_10 = seqlen_acc_dict_total[10][0] / seqlen_acc_dict_total[10][1]
	# acc_record.append(acc_for_k_is_10)
	# final_correct_count += seqlen_acc_dict_total[10][0]
	# final_total_count += seqlen_acc_dict_total[10][1]
	# print('avg accuracy for seq %s when seq len=10 is: %f' % (sequence_name, acc_for_k_is_10))

	### calc acc with no k limit
	acc_no_k_limit = sum([seqlen_acc_dict_total[i][0] for i in range(1, 11)]) / sum([seqlen_acc_dict_total[i][1] for i in range(1, 11)])
	acc_record.append(acc_no_k_limit)
	final_correct_count += sum([seqlen_acc_dict_total[i][0] for i in range(1, 11)])
	final_total_count += sum([seqlen_acc_dict_total[i][1] for i in range(1, 11)])
	# print('avg accuracy for seq %s when seq len=10 is: %f' % (sequence_name, acc_no_k_limit))	
	
	# print(seqlen_acc_dict)
	# print(sum([len(x[1]) for x in seqlen_acc_dict]))
	# print(len(tracktor_ts_bbox_dict))


	# # apply voting on the assignments for each trackletID
	# # majority vote
	# vote_acc_denorm = 0
	# vote_acc_numer = 0
	# for trackletID in tracklet_assignment:
	# 	assignment = tracklet_assignment[trackletID]
	# 	vote_map = collections.Counter([x[1] for x in assignment])
	# 	# print(trackletID, vote_map.most_common(1)[0][0])
	# 	for _l in range(len(tracklet_assignment[trackletID])):
	# 		tracklet_assignment[trackletID][_l][1] = vote_map.most_common(1)[0][0]
	# 		vote_acc_denorm += 1
	# 		print(trackletID, tracklet_assignment[trackletID][_l][1], tracklet_assignment[trackletID][_l][2])
	# 		if tracklet_assignment[trackletID][_l][1] == tracklet_assignment[trackletID][_l][2]:
	# 			vote_acc_numer += 1	

	# apply voting on the assignments for each trackletID
	# voting according to a future sliding window...
	# sliding windoe vote
	for history in range(10, 110, 10):
		updated_tracklet_assignment = deepcopy(tracklet_assignment)
		vote_acc_denorm = 0
		vote_acc_numer = 0
		for trackletID in tracklet_assignment:
			for _l in range(len(tracklet_assignment[trackletID])):
				if _l > history:
					assignment = tracklet_assignment[trackletID][_l - history:_l]
					# print(trackletID, winner)
				else:
					assignment = tracklet_assignment[trackletID][:history]
				vote_map = collections.Counter([x[1] for x in assignment])
				winner = vote_map.most_common(1)[0][0]
				if vote_map[winner] > vote_map[tracklet_assignment[trackletID][_l][1]]:
					updated_tracklet_assignment[trackletID][_l][1] = winner
				vote_acc_denorm += 1
				# print(trackletID, updated_tracklet_assignment[trackletID][_l][1], updated_tracklet_assignment[trackletID][_l][2])
				if updated_tracklet_assignment[trackletID][_l][1] == tracklet_assignment[trackletID][_l][2]:
					vote_acc_numer += 1	

	# # apply voting on the assignments for each trackletID
	# # voting according to a past sliding window...
	# # sliding windoe vote
	# for history in range(10, 110, 10):
	# 	updated_tracklet_assignment = deepcopy(tracklet_assignment)
	# 	vote_acc_denorm = 0
	# 	vote_acc_numer = 0
	# 	for trackletID in tracklet_assignment:
	# 		for _l in range(len(tracklet_assignment[trackletID])):
	# 			assignment = tracklet_assignment[trackletID][max(0, _l-history+1):_l+1] # sliding window of the past
	# 			vote_map = collections.Counter([x[1] for x in assignment])
	# 			# vote_map = collections.Counter()
	# 			# for assign_idx, assign in enumerate(assignment):
	# 			# 	vote_map[assign[1]] += assign_idx+1 # weighted sliding window
	# 			# print(vote_map)
	# 			winner = vote_map.most_common(1)[0][0]
	# 			if vote_map[winner] > vote_map[tracklet_assignment[trackletID][_l][1]]:
	# 				updated_tracklet_assignment[trackletID][_l][1] = winner
	# 			vote_acc_denorm += 1
	# 			# print(trackletID, updated_tracklet_assignment[trackletID][_l][1], updated_tracklet_assignment[trackletID][_l][2])
	# 			if updated_tracklet_assignment[trackletID][_l][1] == tracklet_assignment[trackletID][_l][2]:
	# 				vote_acc_numer += 1	

		# print(tracklet_assignment)
		vote_acc = vote_acc_numer / vote_acc_denorm
		final_correct_count_post_processed[history] += vote_acc_numer
		final_total_count_post_processed[history] += vote_acc_denorm
		# print(vote_acc)
		vote_acc_record[history].append(vote_acc)


	# fig1, ax1 = plt.subplots()
	# ax1.set_title('Sequences length vs Accuracy for seq%d: %s \nAcc=%f \nAcc_post_processed=%f' % (seq_id+1, sequence_name, acc_no_k_limit, vote_acc))
	# ax1.boxplot([x[1] for x in seqlen_acc_dict])
	# ax1.set_xticklabels([x[0] for x in seqlen_acc_dict])
	# plt.tight_layout()

	# histogram = [[x[0], len(x[1])] for x in seqlen_acc_dict]
	# fig2, ax2 = plt.subplots()
	# ax2.set_title('Histogram of sequence length for seq%d: %s' % (seq_id+1, sequence_name))
	# ax2.bar([x[0] for x in histogram], [x[1] for x in histogram])
	# plt.tight_layout()

overall_acc = final_correct_count/final_total_count
for history in range(10, 110, 10):
	overall_acc_post_processed[history] = final_correct_count_post_processed[history] / final_total_count_post_processed[history]


total_seqlen_acc_dict = sorted([[key, value] for key, value in total_seqlen_acc_dict.items()], key=lambda x: x[0])
fig3, ax3 = plt.subplots()
# ax3.set_title('Per-frame Acc vs measurement length')
ax3.boxplot([x[1] for x in total_seqlen_acc_dict])
ax3.set_xticklabels([x[0] for x in total_seqlen_acc_dict]) 
ax3.set_ylabel('Per-frame accuracy')
ax3.set_xlabel('Measurement length')
total_histogram = [[x[0], len(x[1])] for x in total_seqlen_acc_dict]
histogram_sum = sum([x[1] for x in total_histogram])
# total_histogram = []
plt.tight_layout()

fig4, ax4 = plt.subplots()
# ax4.set_title('Histogram of sequence length')
ax4.bar([x[0] for x in total_histogram], [x[1]/histogram_sum * 100 for x in total_histogram])
ax4.set_ylabel('Percentage')
ax4.set_xlabel('Measurement length')
plt.tight_layout()

print(acc_record)
print(vote_acc_record[history])
print('avg accuracy for all the sequence is:', overall_acc)
history = 30
print('overall voted accuracy:', overall_acc_post_processed[history])
plt.figure()
# plt.title('Accuracy for different sequences')
plt.xlabel('Sequence ID')
plt.ylabel('Accuracy')
plt.ylim((0, 1))
plt.plot([i+1 for i in range(len(sequence_names_in_use))], acc_record, '*-', label='no voting')
# for history in range(10, 110, 10):
# 	plt.plot([i+1 for i in range(len(sequence_names_in_use))], vote_acc_record[history], 'o-', label='voting history=%d' % history)
plt.plot([i+1 for i in range(len(sequence_names_in_use))], vote_acc_record[history], 'o-', label='voting window length=%d' % history)
plt.xticks([i+1 for i in range(len(sequence_names_in_use))])
# plt.savefig('accuracy_no_k_limit.png')
plt.legend()
plt.tight_layout()

plt.figure()
# plt.title('Post-processed accuracy vs different voting window size')
plt.xlabel('Voting window size')
plt.ylabel('Accuracy after voting')
overall_acc_post_processed = collections.OrderedDict(sorted(overall_acc_post_processed.items()))
plt.plot(list(overall_acc_post_processed.keys()), list(overall_acc_post_processed.values()), '*-')
plt.tight_layout()


numPeople_cnts_dict = collections.OrderedDict(sorted(numPeople_cnts_dict.items()))
for numPeople in numPeople_cnts_dict:
	numPeople_cnts_dict[numPeople][-1] = numPeople_cnts_dict[numPeople][1] / numPeople_cnts_dict[numPeople][2]

plt.figure()
plt.xlabel('Number of pedestrians present at the same time')
plt.ylabel('Accuracy')
plt.plot(list(numPeople_cnts_dict.keys()), [x[-1] for x in list(numPeople_cnts_dict.values())])
plt.tight_layout()

plt.figure()
plt.xlabel('Number of pedestrians present at the same time')
plt.ylabel('Frame count')
plt.plot(list(numPeople_cnts_dict.keys()), [x[0] for x in list(numPeople_cnts_dict.values())])
plt.tight_layout()


plt.show()