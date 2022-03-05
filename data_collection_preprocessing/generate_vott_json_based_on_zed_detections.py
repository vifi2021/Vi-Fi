'''
This script reads in the initialized vott json file, then modify it based on the zed detections

'''

import json
import os
import collections
from datetime import datetime
import cv2
import hashlib


OPENCV_DEBUG = True
WRITE_JSON = True

# OPENCV_DEBUG = False
# WRITE_JSON = False


project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'
# date = '20211004'
# date = '20211006'
date = '20211007'

sequence_names = sorted([d for d in os.listdir(project_dir) if date in d and os.path.isdir(os.path.join(project_dir,d))])

realnames = ['Subject1', 'Subject6']
if date == '20211006' or date == '20211007':
	realnames = ['Subject1', 'Subject6', 'Subject7']

max_num_of_peds = 0
for seq_id, sequence_name in enumerate(sequence_names):
	print(seq_id, sequence_name)

	if sequence_name != '20211007_144153':
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


	zedID_realname_mapping = collections.defaultdict(lambda: 'Others')
	realname_zedID_mapping = {realname: "" for realname in realnames}
	zedIDs_not_in_mapping = []
	zedIDs_in_mapping = []
	check_count = 0
	for f_i, frame_name in enumerate(tracktor_ts_bbox_dict):	
		# if frame_name != '2021-10-06 15:18:43.228291.png':
		# 	continue
		check_count += 1
		img = cv2.imread(os.path.join(rgb_dir, frame_name))
		
		# generate a asset json file for every frame
		print(frame_name)

		asset_id = hashlib.md5(str('file:'+os.path.join(rgb_dir, frame_name).replace(' ', '%20')).encode('utf-8')).hexdigest()
		asset_json_content = collections.OrderedDict()

		assetInfo = collections.OrderedDict()
		assetInfo['format'] = 'png'
		assetInfo['id'] = asset_id
		assetInfo['name'] = frame_name.replace(' ', '%20')
		assetInfo['path'] = 'file:'+os.path.join(rgb_dir, frame_name).replace(' ', '%20')
		assetInfo['size'] = {'width': img.shape[1], 'height': img.shape[0]}

		asset_json_name = asset_id+ '-asset.json'

		bboxes = tracktor_ts_bbox_dict[frame_name]
		flag = {realname: False for realname in realnames}
		max_num_of_peds = max(max_num_of_peds, len(bboxes))

		for bbox in bboxes:
			
			bbox_id = bbox[0]
			bbox_x = int(float(bbox[1]))
			bbox_y = int(float(bbox[2]))
			bbox_w = int(float(bbox[3]))
			bbox_h = int(float(bbox[4]))

			img = cv2.rectangle(img, (int(bbox_x), int(bbox_y)), (int(bbox_x)+int(bbox_w)-2, int(bbox_y)+int(bbox_h)-2), color=(170,150,30), thickness=2) 
			img = cv2.putText(img, bbox_id, (int(bbox_x), int(bbox_y)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(170,150,30), thickness=3) # 255, 128, 0 orange

			if bbox_id in zedID_realname_mapping:
				# do things
				realname = zedID_realname_mapping[bbox_id]
				flag[realname] = True
			if '' in zedID_realname_mapping:
				flag[zedID_realname_mapping['']] = True
		if OPENCV_DEBUG:
			cv2.imshow('demo', img)
			if all(flag.values()) != True or check_count % 10 == 0:
				check_count = 1
				cv2.waitKey()
				print('Manual help needed. The most updated mapping is the following:')
				print(realname_zedID_mapping)
				print("if it still looks good on the current frame, just input '.' ")
				flag = {realname: False for realname in realnames}
				for realname in realnames:
					if flag[realname] == False:
						# print(realname)
						input_bbox_id = input("Input bbox ID for "+realname+" (don't enter if invisible/occluded) ")
						if input_bbox_id == '.':
							break
						zedID_realname_mapping.pop(realname_zedID_mapping[realname], None)

						zedID_realname_mapping[input_bbox_id] = realname
						realname_zedID_mapping[realname] = input_bbox_id


		regions = []
		for bbox in bboxes:

			bbox_id = bbox[0]
			bbox_x = int(float(bbox[1]))
			bbox_y = int(float(bbox[2]))
			bbox_w = int(float(bbox[3]))
			bbox_h = int(float(bbox[4]))

			regionInfo = collections.OrderedDict()
			regionInfo['id'] = bbox_id
			regionInfo['type'] = 'RECTANGLE'
			regionInfo['tags'] = [zedID_realname_mapping[bbox_id]]
			regionInfo['boundingBox'] = {'height': bbox_h, 'width': bbox_w, 'left': bbox_x, 'top': bbox_y}
			regionInfo['points'] = [{'x': bbox_x, 'y': bbox_y}, {'x': bbox_x+bbox_w, 'y': bbox_y}, {'x': bbox_x+bbox_w, 'y': bbox_y+bbox_h}, {'x': bbox_x, 'y': bbox_y+bbox_h}]
			regions.append(regionInfo)


		assetInfo['state'] = 2
		assetInfo['type'] = 1

		asset_json_content['asset'] = assetInfo
		asset_json_content['regions'] = regions
		asset_json_content['version'] = '2.1.0'

		if WRITE_JSON:
			with open(sequence_dir+'/GND/'+asset_json_name, 'w') as outfile:
			    json.dump(asset_json_content, outfile, indent=4)


		# update vott project file
		if not vott_data is None:
			vott_data['assets'][asset_id] = collections.OrderedDict()
			vott_data['assets'][asset_id]['format'] = 'png'
			vott_data['assets'][asset_id]['id'] = asset_id
			vott_data['assets'][asset_id]['name'] = assetInfo['name']
			vott_data['assets'][asset_id]['path'] = assetInfo['path']
			vott_data['assets'][asset_id]['size'] = assetInfo['size']
			vott_data['assets'][asset_id]['state'] = 2
			vott_data['assets'][asset_id]['type'] = 1

	if WRITE_JSON and not vott_data is None:
		vott_data['assets'] = collections.OrderedDict(sorted(vott_data['assets'].items(), key=lambda x: x[1]['name']))
		with open(sequence_dir+'/GND/RAN_'+sequence_name+'.vott', 'w') as outfile:
		    json.dump(vott_data, outfile, indent=4)


print('The maximum number of pedestrians: ', max_num_of_peds)


