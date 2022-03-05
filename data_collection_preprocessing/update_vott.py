# this script modifies the file locations in the .vott project file and every individual json file for VOTT labeling

# workflow before running this script:
# 1. rename the GND folder that contains the .vott file from other co-workers to GND_coworker
# 2. create another GND folder
# 3. create a vott project with the same project name, to generate a blank .vott template

# pipeline of this script
# 1. read in the vott template
# 2. correct the .vott from other co-workers using the info from vott template
# 3. correct individual json file and and correc the file name


import json
import os
import collections
from datetime import datetime
import cv2
import hashlib

project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'

# date = '20210907'
# date = '20211004'
# date = '20211006'
date = '20211007'

# read in the vott template
sequence_names = sorted([d for d in os.listdir(project_dir) if date in d and os.path.isdir(os.path.join(project_dir,d))])

for seq_id, sequence_name in enumerate(sequence_names):
	print(seq_id, sequence_name)

	if sequence_name != '20211007_144525':
		continue

	sequence_dir = os.path.join(project_dir, sequence_name)
	vott_dir = os.path.join(sequence_dir, 'GND/')
	rgb_dir = os.path.join(sequence_dir, 'RGB/')

	# load the .vott file from myself
	if not os.path.isfile(os.path.join(vott_dir, 'RAN_'+sequence_name+'.vott')):
		print('json template from sequence %s doesn\'t exist, exit.' % sequence_name)
		vott_data = None
	else:
		with open(os.path.join(vott_dir, 'RAN_'+sequence_name+'.vott')) as json_file: 
			vott_data = json.load(json_file, object_pairs_hook=collections.OrderedDict)
			vott_data['assets'] = collections.OrderedDict()

	# load the .vott file from co-worker
	vott_coworker_dir = os.path.join(sequence_dir, 'GND_co-worker/')
	if not os.path.isfile(os.path.join(vott_coworker_dir, 'RAN_'+sequence_name+'.vott')):
		print('.vott file from co-worker for sequence %s doesn\'t exist, exit.' % sequence_name)
		vott_data_coworker = None
		continue

	else:
		with open(os.path.join(vott_coworker_dir, 'RAN_'+sequence_name+'.vott')) as json_file: 
			vott_data_coworker = json.load(json_file, object_pairs_hook=collections.OrderedDict)
		rgb_dir_coworker = list(vott_data_coworker['assets'].items())[0][1]['path'][:-32]
	# determine if co-worker is using WINDOWS ("_" in frame names)
	if '_' in list(vott_data_coworker['assets'].items())[0][1]['name']:
		WINDOWS_OS = True
	else:
		WINDOWS_OS = False
	
	frame_name_id_mapping = {}
	for asset in vott_data_coworker['assets'].values():
		frame_name_id_mapping[asset['name'].replace(' ', '%20')] = asset['id']
	id_frame_name_mapping = {x[1]:x[0] for x in list(frame_name_id_mapping.items())}

	# print((frame_name_id_mapping))
	rgb_files = sorted([f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))], key=lambda x: datetime.strptime(x.replace('20%', ' ').replace('_', ':')[:-4], '%Y-%m-%d %H:%M:%S.%f'))

	for rgb_file_name in rgb_files:

		frame_name = rgb_file_name.replace(' ', '%20').replace('_', ':')
		if WINDOWS_OS:
			frame_name_coworker = frame_name.replace(':', '_')
		else:
			frame_name_coworker = frame_name
		
		old_path = rgb_dir_coworker + frame_name_coworker
		# print(old_path)
		
		if frame_name_coworker not in frame_name_id_mapping:
			# print("!")
			# exit()
			old_asset_id = hashlib.md5(old_path.encode('utf-8')).hexdigest()
		else:
			old_asset_id = frame_name_id_mapping[frame_name_coworker]
			
		new_path = str('file:'+os.path.join(rgb_dir, frame_name))
		new_asset_id = hashlib.md5(new_path.encode('utf-8')).hexdigest()

		# if old_asset_id in vott_data_coworker['assets']:
		# 	# if the old asset id exists in the .vott file, just copy the info from their and update the asset id and path
		# 	vott_data['assets'][new_asset_id] = vott_data_coworker['assets'][old_asset_id]
		# 	vott_data['assets'][new_asset_id]['id'] = new_asset_id
		# 	vott_data['assets'][new_asset_id]['path'] = new_path
		# else:
		# 	# if the old asset_id doesn't exist in the .vott file, generate from scratch
		vott_data['assets'][new_asset_id] = collections.OrderedDict()
		vott_data['assets'][new_asset_id]['format'] = 'png'
		vott_data['assets'][new_asset_id]['id'] = new_asset_id
		vott_data['assets'][new_asset_id]['name'] = frame_name
		vott_data['assets'][new_asset_id]['path'] = new_path
		vott_data['assets'][new_asset_id]['size'] = list(vott_data_coworker['assets'].items())[0][1]['size']
		vott_data['assets'][new_asset_id]['type'] = 1

		asset_json_file_name = old_asset_id + '-asset.json'
		if not os.path.isfile(os.path.join(vott_coworker_dir, asset_json_file_name)):
			# if the individual asset json doen't exist, this means that asset is not labeled
			# print(asset_json_file_name)
			vott_data['assets'][new_asset_id]['state'] = 1
		else:
			with open(os.path.join(vott_coworker_dir, asset_json_file_name)) as asset_json: 
				asset_json_coworker = json.load(asset_json, object_pairs_hook=collections.OrderedDict)
			# to handle some duplicates that Subject7 creates during the labeling
			if 'anon' in asset_json_coworker['asset']['path']:
				continue
			asset_json_coworker['asset']['name'] = frame_name
			asset_json_coworker['asset']['path'] = new_path
			asset_json_coworker['asset']['id'] = new_asset_id
			# to handle some co-worker confused Subject6 with Subject7
			if sequence_name == '20211006_143503' or sequence_name == '20211006_144053' or sequence_name == '20211006_144439' or sequence_name == '20211006_144842':
				for region in asset_json_coworker['regions']:
					if region['tags'][0] == 'Subject6':
						region['tags'] = ['Subject7']
					elif region['tags'][0] == 'Subject7':
						region['tags'] = ['Subject6']
			asset_json_name = new_asset_id + '-asset.json'
			vott_data['assets'][new_asset_id]['state'] = 2
			# store the individual json file to GND folder
			with open(sequence_dir+'/GND/'+asset_json_name, 'w') as outfile:
			    json.dump(asset_json_coworker, outfile, indent=4)

		
	# store the overall .vott project file to GND folder
	vott_data['assets'] = collections.OrderedDict(sorted(vott_data['assets'].items(), key=lambda x: x[1]['name']))
	with open(sequence_dir+'/GND/RAN_'+sequence_name+'.vott', 'w') as outfile:
		    json.dump(vott_data, outfile, indent=4)













