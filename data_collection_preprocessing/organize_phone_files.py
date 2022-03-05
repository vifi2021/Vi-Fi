# this script make copies of the IMU.csv and FTM.csv from the *to_be_organized/ folder and put them into the corresponding sequence folder

import os
import subprocess

project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'
# to_be_organized_dir = '2021_10_04_to_be_organized/'
# to_be_organized_dir = '2021_10_06_to_be_organized/'
to_be_organized_dir = '2021_10_07_to_be_organized/'


sequence_names = sorted([d for d in os.listdir(project_dir) if ''.join(to_be_organized_dir.split('_')[:3]) in d and os.path.isdir(os.path.join(project_dir,d))])
print(sequence_names)

participants_names = sorted([d for d in os.listdir(os.path.join(project_dir, to_be_organized_dir)) if os.path.isdir(os.path.join(project_dir, to_be_organized_dir, d))])
for participant in participants_names:
	print(participant)

	IMU_files = sorted([d for d in os.listdir(os.path.join(project_dir, to_be_organized_dir, participant)) if 'Phone' in d and 'AM' in d]) + sorted([d for d in os.listdir(os.path.join(project_dir, to_be_organized_dir, participant)) if 'Phone' in d and 'PM' in d])

	FTM_files = sorted([d for d in os.listdir(os.path.join(project_dir, to_be_organized_dir, participant)) if 'WiFi' in d and 'AM' in d]) + sorted([d for d in os.listdir(os.path.join(project_dir, to_be_organized_dir, participant)) if 'WiFi' in d and 'PM' in d])

	print(IMU_files)
	print(FTM_files)

	for i in range(len(IMU_files)):
		subprocess.run(['cp', os.path.join(project_dir, to_be_organized_dir, participant, IMU_files[i]), os.path.join(project_dir, sequence_names[i], 'IMU/')])
		subprocess.run(['cp', os.path.join(project_dir, to_be_organized_dir, participant, FTM_files[i]), os.path.join(project_dir, sequence_names[i], 'WiFi/')])



