# this scripts create directores for the collected data
'''
sequence_name/
	RGB/
	Depth/
	Dist
	WiFi/
	IMU/
	GND/
'''

import os
from pathlib import Path

# to_be_organized_dir = '2021_10_04_to_be_organized/'
# to_be_organized_dir = '2021_10_06_to_be_organized/'
to_be_organized_dir = '2021_10_07_to_be_organized/'

svo_files = sorted([d for d in os.listdir(os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', to_be_organized_dir)) if '.svo' in d])

for svo_file_name in svo_files:
	sequence_name = '%s_%s' % (''.join(to_be_organized_dir.split('_')[:3]), svo_file_name.replace('.svo', '').split('_')[-1].replace('-', ''))

	RGB_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', sequence_name, 'RGB/')
	if not Path(RGB_dir).is_dir():
		print("creating", RGB_dir)
		Path(RGB_dir).mkdir(parents=True, exist_ok=True)

	Depth_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', sequence_name, 'Depth/')
	if not Path(Depth_dir).is_dir():
		print("creating", Depth_dir)
		Path(Depth_dir).mkdir(parents=True, exist_ok=True)

	Dist_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', sequence_name, 'Dist/')
	if not Path(Dist_dir).is_dir():
		print("creating", Dist_dir)
		Path(Dist_dir).mkdir(parents=True, exist_ok=True)

	WiFi_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', sequence_name, 'WiFi/')
	if not Path(WiFi_dir).is_dir():
		print("creating", WiFi_dir)
		Path(WiFi_dir).mkdir(parents=True, exist_ok=True)

	IMU_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', sequence_name, 'IMU/')
	if not Path(IMU_dir).is_dir():
		print("creating", IMU_dir)
		Path(IMU_dir).mkdir(parents=True, exist_ok=True)

	GND_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', sequence_name, 'GND/')
	if not Path(GND_dir).is_dir():
		print("creating", GND_dir)
		Path(GND_dir).mkdir(parents=True, exist_ok=True)

	GND_coworker_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/', sequence_name, 'GND_co-worker/')
	if not Path(GND_coworker_dir).is_dir():
		print("creating", GND_coworker_dir)
		Path(GND_coworker_dir).mkdir(parents=True, exist_ok=True)