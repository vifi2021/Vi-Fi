# this script check the sample rate for:
# FTM.csv files
# IMU.csv files

import os 
import csv
from matplotlib import pyplot as plt
import collections

project_dir = '/media/hans/WINLAB-MOBILE-GROUP/RAN_Dec_data/'

sequence_names = []

sequence_names += [d for d in os.listdir(project_dir) if '20210907' in d and os.path.isdir(os.path.join(project_dir,d))]

sequence_names += [d for d in os.listdir(project_dir) if '20211004' in d and os.path.isdir(os.path.join(project_dir,d))]

# sequence_names += [d for d in os.listdir(project_dir) if '20211006' in d and os.path.isdir(os.path.join(project_dir,d))]

# sequence_names += [d for d in os.listdir(project_dir) if '20211007' in d and os.path.isdir(os.path.join(project_dir,d))]


sequence_names = sorted(sequence_names)


ftm_sample_rates = collections.defaultdict(list)

# iterate through all the WiFi csv files
for sequence_name in sequence_names:
	ftm_csv_dir = os.path.join(project_dir, sequence_name, 'WiFi/')
	imu_csv_dir = os.path.join(project_dir, sequence_name, 'IMU')

	print('\n', sequence_name)
	# read ftm csv from all participants
	ftm_files = sorted([f for f in os.listdir(ftm_csv_dir) if os.path.isfile(os.path.join(ftm_csv_dir,f))])
	ftm_csv_dicts = {}
	for i in range(len(ftm_files)):
		name = ftm_files[i].split('_')[1].replace('.csv', '')
		# print(name)
		# print(ftm_files[i])
		with open(os.path.join(ftm_csv_dir, ftm_files[i])) as f:
			reader = csv.reader(f)
			ftm_csv_data = list(reader)
		# ftm_csv_dicts[name] = {int(ftm_csv_data[_i][0]): ftm_csv_data[_i][1:] for _i in range(len(ftm_csv_data))}

		num_samples = len(ftm_csv_data)
		# print('number of samples:', num_samples)


		# find start time and end time and calculate time duration in seconds
		start_time = int(ftm_csv_data[0][0])
		end_time = int(ftm_csv_data[-1][0])
		delta_time = (end_time - start_time) / 1000 # in seconds

		# calculate the average sample rate
		sample_rate_ftm = num_samples / delta_time
		ftm_sample_rates[name].append(sample_rate_ftm)
		# print('ftm sample rate for %s is: %s' % (name, sample_rate_ftm))



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

		# calculate the average sample rate for ACCEL
		accel_timestamps = sorted(accel_dicts[name].keys())
		start_time_accel = accel_timestamps[0]
		end_time_accel = accel_timestamps[-1]
		delta_time_accel = (end_time_accel - start_time_accel) / 1000
		sample_rate_accel = len(accel_timestamps) / delta_time_accel
		# print('accel sample rate for %s is: %s' % (name, sample_rate_accel))

		# calculate the average sample rate for GYRO
		gyro_timestamps = sorted(gyro_dicts[name].keys())
		start_time_gyro = gyro_timestamps[0]
		end_time_gyro = gyro_timestamps[-1]
		delta_time_gyro = (end_time_gyro - start_time_gyro) / 1000
		sample_rate_gyro = len(gyro_timestamps) / delta_time_gyro
		# print('gyro sample rate for %s is: %s' % (name, sample_rate_gyro))

		# calculate the average sample rate for ACCEL
		mag_timestamps = sorted(mag_dicts[name].keys())
		start_time_mag = mag_timestamps[0]
		end_time_mag = mag_timestamps[-1]
		delta_time_mag = (end_time_mag - start_time_mag) / 1000
		sample_rate_mag = len(mag_timestamps) / delta_time_mag
		# print('mag sample rate for %s is: %s' % (name, sample_rate_mag))


print(sequence_names)
fig, ax = plt.subplots()

for name in ['Subject1', 'Subject6', 'Subject7']:
	ax.plot(range(len(ftm_sample_rates[name])), ftm_sample_rates[name], label=name)
ax.set_xticks(range(len(ftm_sample_rates['Subject1'])))
ax.set_xticklabels(sequence_names, rotation=90, fontsize=8)
ax.set_ylabel('FTM sample rate (Hz)')

# plt.xticks(range(len(ftm_sample_rates[name])), sequence_names, rotation=90)
plt.axhline(y=3, color='r', linestyle='-', label='frame rate')
plt.tight_layout()
plt.legend()
plt.show()