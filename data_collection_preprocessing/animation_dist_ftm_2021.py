### this scripts visualize the depth and ftm measurements in synchronization with RGB frames

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
# from flow_utils import readFlow, flow2img
import random
from matplotlib import pyplot as plt

### return the first k timestamp that is closest to target
def k_nearest_ts(ts_list, target_ts, k, offset):
	res = sorted(ts_list, key=lambda x: abs(x + offset- target_ts))[:k]
	return sorted(res)

opencv_color_mapping = {'Subject1': (0, 0, 255),
				 'Subject6': (255, 0, 0),
				 'Others': (0, 255, 0),
				 'Subject7': (255, 255, 0),
				 'Subject5': (0, 255, 255)}

plt_color_mapping = {'Subject1': (1, 0, 0),
				 'Subject6': (0, 0, 1),
				 'Others': (0, 1, 0),
				 'Subject7': (0, 1, 1),
				 'Subject5': (1, 1, 0)}

project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'


sequence_name = '20211006_144053'

date = sequence_name.split('_')[0]


sequence_names = sorted([d for d in os.listdir(project_dir) if date in d and os.path.isdir(os.path.join(project_dir,d))])
print(sequence_names)

print(sequence_name)


sequence_dir = os.path.join(project_dir,  sequence_name)
rgb_dir = os.path.join(sequence_dir, 'RGB/')
depth_dir = os.path.join(sequence_dir, 'Dist/')
ftm_csv_dir = os.path.join(sequence_dir, 'WiFi/')
vott_label_dir = os.path.join(sequence_dir, 'GND/vott-json-export/')
if not os.path.isfile(os.path.join(vott_label_dir, 'RAN_'+sequence_name+'-export.json')):
	print('GND for %s is not ready yet, skip' % sequence_name)
	exit()

# read in the txt that records valid frame range
if os.path.isfile(os.path.join(sequence_dir, 'valid_frame_range.txt')):
	with open(os.path.join(sequence_dir, 'valid_frame_range.txt'), 'r') as frame_range:
		valid_range = frame_range.readlines()[0].replace('\n', '').split(", ")
else:
	valid_range = None
valid_range = None # uncomment this line if don't use valid range

# read in the txt that logs the time offset for each participants
if os.path.isfile(os.path.join(sequence_dir, 'time_offsets.txt')):
	with open(os.path.join(sequence_dir, 'time_offsets.txt'), 'r') as offset_txt:
		offset_lines = offset_txt.readlines()
	offset = {offset_line.split(': ')[0]: float(offset_line.split(': ')[1].replace('\n', '')) for offset_line in offset_lines}
	# print(offset)
	# exit()
else:
	print('no offset is used')
	offset = None
	exit()
if date == '20201228':
	offset = {'Subject1': 2219.273,
			'Subject3': 2962.273,
			'Subject2': 2764.273,
			'Subject4': 3461.273,
			'Subject5': 2948.273}
# offset = collections.defaultdict(int) # uncomment this line if don't use valid range

### read vott label json file
with open(os.path.join(vott_label_dir, 'RAN_'+sequence_name+'-export.json')) as json_file: 
	vott_data = json.load(json_file, object_pairs_hook=collections.OrderedDict) 

sorted_vott_data = sorted(list(vott_data['assets'].values()), key=lambda x: datetime.strptime(x["asset"]["name"].replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'))
if valid_range:
	sorted_vott_data = [x for x in sorted_vott_data if datetime.strptime(valid_range[0], '%Y-%m-%d %H:%M:%S.%f') <= datetime.strptime(x["asset"]["name"].replace('%20', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f') <= datetime.strptime(valid_range[1], '%Y-%m-%d %H:%M:%S.%f')]

depth_medians = collections.defaultdict(lambda: ([], []))

ftm_measurements = collections.defaultdict(lambda: [[], []])

ftm_files = sorted([f for f in os.listdir(ftm_csv_dir) if os.path.isfile(os.path.join(ftm_csv_dir,f))])
# print(ftm_files)


# read ftm csv from all participants
ftm_csv_dicts = {}
for i in range(len(ftm_files)):
	with open(os.path.join(ftm_csv_dir, ftm_files[i])) as f:
		participant_name = ftm_files[i].split('_')[1]
		print(participant_name)
		reader = csv.reader(f)
		ftm_csv_data = list(reader)
	ftm_csv_dicts[participant_name] = {int(ftm_csv_data[_i][0]): ftm_csv_data[_i][1:] for _i in range(len(ftm_csv_data))}

# print(ftm_csv_dicts.keys())
# exit()





# for labeled_frame in list(vott_data['assets'].values())[::-1]:
for labeled_frame in sorted_vott_data[::3]:
	# print(labeled_frame)
	# if not labeled_frame['regions']: continue
	frame_name = labeled_frame['asset']['name'].replace('_', ':').replace('%20', ' ')
	img_name = os.path.join(rgb_dir, frame_name )
	img = cv2.imread(img_name)
	print(img_name)


	# depth_pkl_name = labeled_frame['asset']['name'].replace('%20', ' ').replace('_', ':').replace('.png', '.pkl')
	depth_pkl_name = labeled_frame['asset']['name'].replace('%20', ' ').replace('_', ':').replace('.png', '.npy')
	depth_pkl_path = os.path.join(depth_dir, depth_pkl_name)
	if not os.path.isfile(depth_pkl_path):
		print("cannot find:", depth_pkl_path)
		continue
	# f = open(depth_pkl_path, 'rb')
	# depth_pkl = pickle.load(f, encoding='bytes')
	# f.close()
	depth_pkl = np.load(depth_pkl_path)

	### get synchronized ftm range

	### infer the corresponding ftm timestamp
	rgbd_linux_time = int(float(datetime.strptime(depth_pkl_name[:-4], '%Y-%m-%d %H:%M:%S.%f').strftime('%s') + str(datetime.strptime(depth_pkl_name[:-4], '%Y-%m-%d %H:%M:%S.%f').microsecond/1e6)[1:]) * 1e3)


	ftm_ts = {}

	for name in ftm_csv_dicts.keys():
		ftm_ts[name] = k_nearest_ts(ts_list=ftm_csv_dicts[name].keys(), target_ts=rgbd_linux_time, k=1, offset=offset[name])

	# print(ftm_csv_dicts[name].keys())


	depth_pkl = np.reshape(depth_pkl, (depth_pkl.shape[0], depth_pkl.shape[1], 1))
	bbox_id_set = set()
	for i in range(len(labeled_frame['regions'])):
		bbox_x = int(float(labeled_frame['regions'][i]['boundingBox']['left']))
		bbox_y = int(float(labeled_frame['regions'][i]['boundingBox']['top']))
		bbox_h = int(float(labeled_frame['regions'][i]['boundingBox']['height']))
		bbox_w = int(float(labeled_frame['regions'][i]['boundingBox']['width']))

		name = labeled_frame['regions'][i]['tags'][0]
		bbox_id_set.add(name)
		cv2.rectangle(img, (int(bbox_x),int(bbox_y)), (int(bbox_x+bbox_w),int(bbox_y+bbox_h)), opencv_color_mapping[name], 3)

		depth_roi = depth_pkl[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
		# print(labeled_frame['regions'][i]['tags'])
		depth_medians[labeled_frame['regions'][i]['tags'][0]][1].append(np.median(depth_roi))
		depth_medians[labeled_frame['regions'][i]['tags'][0]][0].append(datetime.strptime(depth_pkl_name[:-4], '%Y-%m-%d %H:%M:%S.%f'))

	cv2.imshow("frame", img)
	# if len(bbox_id_set) != len(labeled_frame['regions']):
		# cv2.waitKey(0)
	cv2.waitKey(0)


	for name in ftm_csv_dicts.keys():
		ftm_measurements[name][1].append(float(ftm_csv_dicts[name][ftm_ts[name][0]][2])/1000)
		ftm_measurements[name][0].append(datetime.strptime(depth_pkl_name[:-4], '%Y-%m-%d %H:%M:%S.%f'))


	fig4 = plt.figure(1)
	plt.cla()
	for i, name in enumerate(ftm_csv_dicts.keys()):
		plt.plot(ftm_measurements[name][0], ftm_measurements[name][1], color=plt_color_mapping[name], linestyle='--', label="FTM"+str(i+1))
		plt.plot(depth_medians[name][0], depth_medians[name][1], color=plt_color_mapping[name], linestyle='-', label="CameraDepth"+str(i+1))

	
	plt.xlabel('Timestamp')
	plt.ylabel("Depth and FTM in meter")
	# plt.ylim((0, 20))
	# plt.title('AUC over training epoch', fontsize='xx-large')
	plt.legend(loc='upper left')
	# plt.tick_params(axis='both', labelsize=16)
	# fig4.tight_layout()
	plt.pause(0.01)
