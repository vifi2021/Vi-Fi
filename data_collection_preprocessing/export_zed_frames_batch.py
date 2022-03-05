# run the 'export.py' in batch

import os
import subprocess

project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'
# to_be_organized_dir = '2021_10_06_to_be_organized'
to_be_organized_dir = '2021_10_07_to_be_organized'
svo_file_names = sorted([d for d in os.listdir(os.path.join(project_dir, to_be_organized_dir)) if '.svo' in d and os.path.isfile(os.path.join(project_dir, to_be_organized_dir, d))])
sequence_names = sorted([d for d in os.listdir(project_dir) if ''.join(to_be_organized_dir.split('_')[:3]) in d and os.path.isdir(os.path.join(project_dir,d))])


for svo_file_name, sequence_name in zip(svo_file_names, sequence_names):
	subprocess.run(['python3', 'export_svo.py', os.path.join(project_dir, to_be_organized_dir, svo_file_name), os.path.join(project_dir, sequence_name, 'RGB/'), '4'])