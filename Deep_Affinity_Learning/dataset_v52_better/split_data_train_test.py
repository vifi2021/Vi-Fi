import os
import csv
import json
import pandas as pd
import numpy as np
import random
import copy
import itertools
import gc


project_dir = './'
sequence_names = []

# indoor_sequence_names = [d for d in os.listdir(project_dir) if '2020' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))]
# sequence_names += indoor_sequence_names

cac_spot1_sequence_names = [d for d in os.listdir(project_dir) if '20210907' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))]
sequence_names += cac_spot1_sequence_names

cac_spot2_sequence_names = [d for d in os.listdir(project_dir) if '20211004' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))]
sequence_names += cac_spot2_sequence_names

busch_sequence_names = [d for d in os.listdir(project_dir) if '20211006_14' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))]+[d for d in os.listdir(project_dir) if '20211006_15' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))]
sequence_names += busch_sequence_names

liv_spot1_sequence_names = [d for d in os.listdir(project_dir) if '20211007_10' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))] + [d for d in os.listdir(project_dir) if '20211007_11' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))]
sequence_names += liv_spot1_sequence_names

liv_spot2_sequence_names = [d for d in os.listdir(project_dir) if '20211007_13' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))] + [d for d in os.listdir(project_dir) if '20211007_14' in d and '.pkl' in d and os.path.isfile(os.path.join(project_dir,d))]
sequence_names += liv_spot2_sequence_names

sequence_names = ['_'.join(sequence_name[:-4].split('_')[2:4]) for sequence_name in sequence_names if '.pkl' in sequence_name]

sequence_names = sorted(sequence_names)
print(len(sequence_names))

random.shuffle(sequence_names)

N = 1
fold_size = len(sequence_names) // N

shuffle = True
# shuffle = False

WINDOW_SIZE = 10

csv_dir = "train_outdoor_test_split_v2/"

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

fold_info_txt = open(os.path.join(csv_dir, 'fold_info.txt'), 'w')

for i in range(N):
	# val_sequence_names = random.sample(indoor_sequence_names, 1) + random.sample(cac_spot1_sequence_names, 1) + random.sample(cac_spot2_sequence_names, 1) + random.sample(busch_sequence_names, 1) + random.sample(liv_spot1_sequence_names, 1) + random.sample(liv_spot2_sequence_names, 1)
	# val_sequence_names = ['_'.join(sequence_name[:-4].split('_')[2:4]) for sequence_name in val_sequence_names if '.pkl' in sequence_name]
	
	# val_sequence_names = ['20201223_140951', '20210907_145202', '20211004_142306', '20211006_152208', '20211007_105924', '20211007_134632']
	val_sequence_names = ['20210907_145202', '20211004_142306', '20211006_152208', '20211007_105924', '20211007_134632']

	print(val_sequence_names)

	fold_info_txt.write(', '.join(val_sequence_names))
	fold_info_txt.write('\n')

	train_csv_name = "train_" + "fold" + str(i+1) + ".csv"
	test_csv_name = "test_" + "fold" + str(i+1) + ".csv"

	train_d = {}
	idx_train = 0
	test_d = {}
	idx_test = 0

	train_df = pd.DataFrame(columns = ['DepthFrameName', 'dist_vectors', 'ftm_vectors', 'AffinityMat']) 
	test_df = pd.DataFrame(columns = ['DepthFrameName', 'dist_vectors', 'ftm_vectors', 'AffinityMat']) 

	for sequence in (sequence_names):
		if not os.path.isfile('affinity_gnd_'+sequence+'.pkl'): continue
		matching_gnd = pd.read_pickle('affinity_gnd_'+sequence+'.pkl')
		print(sequence, matching_gnd.shape)
		for idx in range(matching_gnd.shape[0]):
			### handle nan
			depth_vectors_dict = matching_gnd.iloc[idx, 1]
			depth_vectors = np.array(list(depth_vectors_dict.values()))
			depth_vectors_lengths = np.array([len(x) for x in depth_vectors]).reshape(-1, 1)
			if depth_vectors.dtype == 'object':
				### pad the depth vectors to uniform seq length
				for ii in range(len(depth_vectors)):
					depth_vectors[ii] = np.pad(depth_vectors[ii], (0, 5*WINDOW_SIZE - len(depth_vectors[ii])),'constant')
			else:
				depth_vectors = np.pad(depth_vectors, ((0, 0), (0, 5*WINDOW_SIZE - depth_vectors.shape[1])), 'constant')
			depth_vectors = np.array([v for v in depth_vectors], dtype='float32')
			# append the valid length to the end of each vector
			depth_vectors = np.append(depth_vectors, depth_vectors_lengths, axis=1)

			ftm_vectors_dict = matching_gnd.iloc[idx, 2]
			ftm_vectors = np.array(list(ftm_vectors_dict.values()))
			ftm_vectors_lengths = np.array([len(x) for x in ftm_vectors]).reshape(-1, 1)
			if ftm_vectors.dtype == 'object':
				### pad the depth vectors to uniform seq length
				for ii in range(len(ftm_vectors)):
					ftm_vectors[ii] = np.pad(ftm_vectors[ii], (0, 11*WINDOW_SIZE - len(ftm_vectors[ii])),'constant')
				# ftm_vectors = np.array([v for v in ftm_vectors])
			else:
				ftm_vectors = np.pad(ftm_vectors, ((0, 0), (0, 11*WINDOW_SIZE - ftm_vectors.shape[1])), 'constant')
			ftm_vectors = np.array([v for v in ftm_vectors], dtype='float32')
			# append the valid length to the end of each vector
			ftm_vectors = np.append(ftm_vectors, ftm_vectors_lengths, axis=1)

			affinity_mat = matching_gnd.iloc[idx, 3]


			### split the data 
			if matching_gnd.iloc[idx, 0].split('/')[0] in val_sequence_names:
				if not np.sum(np.isnan(depth_vectors)) and not np.sum(np.isinf(depth_vectors)) and not np.sum(np.isnan(ftm_vectors)) and not np.sum(np.isinf(ftm_vectors)):
					test_d[idx_test] = matching_gnd.iloc[idx, :]
					idx_test += 1
			else:
				if not np.sum(np.isnan(depth_vectors)) and not np.sum(np.isinf(depth_vectors)) and not np.sum(np.isnan(ftm_vectors)) and not np.sum(np.isinf(ftm_vectors)):
					if shuffle and len(depth_vectors) <=10:
						## shuffle the affinity matrix to cover all the permutations
						## randomly choose k ftm shuffle x k dist shuffle
						idx_ftm = np.arange(len(ftm_vectors_dict))
						permutations_idx_ftm = list(itertools.permutations(idx_ftm)) # if len(idx_ftm) is too large, may cause memory problem

						idx_dist = np.arange(len(depth_vectors_dict))
						permutations_idx_dist = list(itertools.permutations(idx_dist)) # if len(idx_dist) is too large, may cause memory problem

						if len(permutations_idx_ftm) >= 6:
							permutations_idx_ftm = random.sample(permutations_idx_ftm, 6)
						if len(permutations_idx_dist) >= 6:
							permutations_idx_dist = random.sample(permutations_idx_dist, 6)
						for idx_ftm_perm in permutations_idx_ftm:
							for idx_dist_perm in permutations_idx_dist:
								# print(count)
								affinity_mat_shuf = copy.deepcopy(affinity_mat)
								idx_ftm_perm = np.array(idx_ftm_perm)
								idx_dist_perm = np.array(idx_dist_perm)

								l = list(ftm_vectors_dict.items())
								l_shuf = []
								for ii in idx_ftm_perm:
									l_shuf.append(l[ii])
								consecutive_ftm_map_perm = dict(l_shuf)
								affinity_mat_shuf[0:len(consecutive_ftm_map_perm)] = affinity_mat_shuf[idx_ftm_perm]

								l = list(depth_vectors_dict.items())
								l_shuf = []
								for ii in idx_dist_perm:
									l_shuf.append(l[ii])
								consecutive_dist_map_perm = dict(l_shuf)
								affinity_mat_shuf[:, 0:len(consecutive_dist_map_perm)] = affinity_mat_shuf[:, idx_dist_perm]

								train_d[idx_train] = {'DepthFrameName': matching_gnd.iloc[idx, 0], 'dist_vectors': consecutive_dist_map_perm, 'ftm_vectors': consecutive_ftm_map_perm, 'AffinityMat': affinity_mat_shuf}
								idx_train += 1
					else:
						train_d[idx_train] = matching_gnd.iloc[idx, :]
						idx_train += 1

		
	train_df = pd.DataFrame.from_dict(train_d, 'index')
	test_df = pd.DataFrame.from_dict(test_d, 'index')

	print('saving to pickle')
	train_df.to_pickle(csv_dir+train_csv_name.replace('.csv', '.pkl'))
	test_df.to_pickle(csv_dir+test_csv_name.replace('.csv', '.pkl'))

	test_df.to_csv (csv_dir+test_csv_name, index = False, header=True)
	
	print(train_df.shape, test_df.shape)

fold_info_txt.close()
	

