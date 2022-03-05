'''
This script reads in the initialized vott json file, then modify it based on the zed detections

'''

import json
import os
import collections
from datetime import datetime
import cv2
import hashlib
from matplotlib import pyplot as plt
import numpy as np


OPENCV_DEBUG = True
WRITE_JSON = True

# OPENCV_DEBUG = False
# WRITE_JSON = False


project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'
sequence_names = []

date = '20211004'
sequence_names += [d for d in os.listdir(project_dir) if date in d and os.path.isdir(os.path.join(project_dir,d))]

date = '20210907'
sequence_names += [d for d in os.listdir(project_dir) if date in d and os.path.isdir(os.path.join(project_dir,d))]

date = '20211006'
sequence_names += [d for d in os.listdir(project_dir) if date in d and os.path.isdir(os.path.join(project_dir,d))]
date = '20211007'
sequence_names += [d for d in os.listdir(project_dir) if date in d and os.path.isdir(os.path.join(project_dir,d))]

sequence_names = sorted(sequence_names)


max_num_of_peds = 0

seq_numOfPeds_mapping = collections.defaultdict(list)
for seq_id, sequence_name in enumerate(sequence_names):
	print(seq_id, sequence_name)
	if sequence_name != '20211007_143810':
		continue

	sequence_dir = os.path.join(project_dir, sequence_name)
	vott_dir = os.path.join(sequence_dir, 'GND/')
	tracker_output_dir = os.path.join(sequence_dir, 'zedBox_'+sequence_name+'.txt')
	rgb_dir = os.path.join(sequence_dir, 'RGB/')
	if not os.path.isfile(os.path.join(vott_dir, 'RAN_'+sequence_name+'.vott')):
		print('json template from sequence %s doesn\'t exist, exit.' % sequence_name)
		vott_data = None
	else:
		with open(os.path.join(vott_dir, 'RAN_'+sequence_name+'.vott')) as json_file: 
			vott_data = json.load(json_file, object_pairs_hook=collections.OrderedDict)

	with open(tracker_output_dir, 'r') as tracktor_output_file:
		tracktor_output = tracktor_output_file.readlines()

	# read the RGB files and sort the file name according to timestamp
	rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))], key=lambda x: datetime.strptime(x.replace('20%', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'))
	# tracktor_ts_bbox_dict = collections.OrderedDict()
	tracktor_ts_bbox_dict = collections.OrderedDict()
	for line in tracktor_output:
		line_array = line.replace('\n', '').split(',')

		''' downsample if needed '''
		# if (int(line_array[0]) - 1) % 10 != 0: continue
		# print(line_array)
		# print(int(line_array[0])-1)

		frame_id = (int(line_array[0])-1)//3
		if frame_id >= len(rgb_files):
			break
		key = rgb_files[frame_id]
		if key in tracktor_ts_bbox_dict:
			tracktor_ts_bbox_dict[rgb_files[frame_id]].append(line_array[1:])
		else:
			tracktor_ts_bbox_dict[rgb_files[frame_id]] = [line_array[1:]]


	zedIDs_not_in_mapping = []
	zedIDs_in_mapping = []
	check_count = 0
	for f_i, frame_name in enumerate(tracktor_ts_bbox_dict):		
		check_count += 1
		
		# generate a asset json file for every frame
		# print(frame_name)

		bboxes = tracktor_ts_bbox_dict[frame_name]
		num_of_peds = len(bboxes)
		seq_numOfPeds_mapping[sequence_name].append(num_of_peds)
		max_num_of_peds = max(max_num_of_peds, num_of_peds)



# plot box plot of number of peds for each sequence
seq_numOfPeds_mapping = collections.OrderedDict(seq_numOfPeds_mapping)
print(seq_numOfPeds_mapping.keys())
bp_data = [seq_numOfPeds_mapping[k] for k in seq_numOfPeds_mapping.keys()]
# print(bp_data)
fig, ax = plt.subplots()
ax.boxplot(bp_data)
ax.plot(range(1, len(seq_numOfPeds_mapping)+1), [np.mean(seq_numOfPeds_mapping[k]) for k in seq_numOfPeds_mapping.keys()], '*-', label='avg.')
ax.set_xticklabels(list(seq_numOfPeds_mapping.keys()),
                    rotation=90, fontsize=8)
ax.set_ylabel('# of detected people at a time')


plt.tight_layout()
plt.legend()
plt.show()




print('The maximum number of pedestrians: ', max_num_of_peds)


