# this scripts plot GPS measurements for RAN data collected in 2021

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
import random
from matplotlib import pyplot as plt

sequence_names = sorted([d for d in os.listdir('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/') if '20210907' in d and os.path.isdir(os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/',d))])
print(sequence_names)

seq_id = 1 - 1
sequence_name = sequence_names[seq_id]
print(sequence_name)

sequence_dir = os.path.join('/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/',  sequence_name)
imu_csv_dir = os.path.join(sequence_dir, 'IMU/')


# read IMU csv from all participants
imu_files = sorted([f for f in os.listdir(imu_csv_dir) if os.path.isfile(os.path.join(imu_csv_dir,f))])
print(imu_files)
accel_dicts = {}
gyro_dicts = {}
mag_dicts = {}
gps_dicts = {}
for i in range(len(imu_files)):
	name = "_".join(imu_files[i].split('_')[:2])
	accel_dicts[name] = {}
	gyro_dicts[name] = {}
	mag_dicts[name] = {}
	gps_dicts[name] = {}
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
		if 'GPS' in imu_csv_data[i]:
			gps_dicts[name].update({int(imu_csv_data[i][0]): imu_csv_data[i][1:]})

plt.figure()
plt.title(f'GPS for {sequence_name}')
plt.xlabel('longitude')
plt.ylabel('latitude')
for name in gps_dicts.keys():
	print("GPS for", name)
	gps_dicts[name] = collections.OrderedDict(sorted(gps_dicts[name].items(), key=lambda x: float(x[0])))
	lat = [float(x[1]) for x in list(gps_dicts[name].values())]
	lon = [float(x[2]) for x in list(gps_dicts[name].values())]
	ts = [int(x) for x in list(gps_dicts[name].keys())]
	print('starting point (lat, long): ', lat[0], lon[0])
	print(len(ts))
	print(ts[0], ts[-1])
	print('start time', datetime.fromtimestamp(int(str(ts[0])[:-3])))
	print('end time', datetime.fromtimestamp(int(str(ts[-1])[:-3])))
	# exit()
	plt.plot(lon[0], lat[0], 'ro')
	plt.plot(lon, lat)
plt.show()


