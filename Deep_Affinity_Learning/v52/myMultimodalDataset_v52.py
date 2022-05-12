import torch
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
import pandas as pd
import os
import numpy as np
import math
from PIL import Image
import pickle
import random
import cv2
import collections
import json
import ast



class myMutimodalDataset(Dataset):

	def __init__(self, gndPKLPath, Nm_phone=5, Nm_camera=15, phase="train", transform=None):
		self.matching_gnd = pd.read_pickle(gndPKLPath)
		self.transform = transform
		self.phase=phase
		self.Nm_phone = Nm_phone
		self.Nm_camera = Nm_camera
		self.window_size = 10

	def __getitem__(self, idx):
		frame_name = self.matching_gnd.iloc[idx, 0]

		### depth vector and mask
		depth_vectors_dict = self.matching_gnd.iloc[idx, 1]
		depth_vectors = np.array(list(depth_vectors_dict.values()))
		depth_vectors_lengths = np.array([len(x) for x in depth_vectors]).reshape(-1, 1)/5
		if depth_vectors.dtype == 'object':
			### pad the depth vectors to uniform seq length
			for i in range(len(depth_vectors)):
				depth_vectors[i] = np.pad(depth_vectors[i], (0, 5*self.window_size - len(depth_vectors[i])),'constant')
			depth_vectors = np.array([v for v in depth_vectors])
		else:
			depth_vectors = np.pad(depth_vectors, ((0, 0), (0, 5*self.window_size - depth_vectors.shape[1])), 'constant')
		# append the valid length to the end of each vector
		depth_vectors = np.append(depth_vectors, depth_vectors_lengths, axis=1)
		depth_vectors_mask = np.array([True] * depth_vectors.shape[0])
		depth_num = depth_vectors.shape[0]
		
		### ftm vector and mask
		ftm_vectors_dict = self.matching_gnd.iloc[idx, 2]
		ftm_vectors = np.array(list(ftm_vectors_dict.values()))
		ftm_vectors_lengths = np.array([len(x)for x in ftm_vectors]).reshape(-1, 1)/11
		if ftm_vectors.dtype == 'object':
			### pad the depth vectors to uniform seq length
			for i in range(len(ftm_vectors)):
				ftm_vectors[i] = np.pad(ftm_vectors[i], (0, 11*self.window_size - len(ftm_vectors[i])),'constant')
			ftm_vectors = np.array([v for v in ftm_vectors])
		else:
			ftm_vectors = np.pad(ftm_vectors, ((0, 0), (0, 11*self.window_size - ftm_vectors.shape[1])), 'constant')
		# append the valid length to the end of each vector
		ftm_vectors = np.append(ftm_vectors, ftm_vectors_lengths, axis=1)
		ftm_vectors_mask = np.array([True] * ftm_vectors.shape[0])
		ftm_num = ftm_vectors.shape[0]

		### affinity matrix dict
		aff_mat = self.matching_gnd.iloc[idx, -1] 
		
		### expand the number of depth/ftm vectors to Nm
		depth_vectors_mask = np.concatenate((depth_vectors_mask, np.array([False]*(self.Nm_camera - depth_vectors.shape[0]))))
		depth_vectors = np.vstack((depth_vectors, np.ones((self.Nm_camera - depth_vectors.shape[0], depth_vectors.shape[1]))))
		ftm_vectors_mask = np.concatenate((ftm_vectors_mask, np.array([False]*(self.Nm_phone - ftm_vectors.shape[0]))))
		ftm_vectors = np.vstack((ftm_vectors, np.ones((self.Nm_phone - ftm_vectors.shape[0], ftm_vectors.shape[1]))))

		### shuffle the order of the vectors
		r = random.uniform(0, 1)
		if r < 0:

			### shuffle the cols and rows that have depth and ftm info
			ftm_index = np.arange(ftm_num)
			# print(ftm_index)
			np.random.shuffle(ftm_index)
			ftm_index = np.concatenate((ftm_index, np.arange(ftm_num, self.Nm_phone)))
			depth_index = np.arange(depth_num)
			np.random.shuffle(depth_index)
			depth_index = np.concatenate((depth_index, np.arange(depth_num, self.Nm_camera)))

			depth_vectors = depth_vectors[depth_index]
			depth_vectors_mask = depth_vectors_mask[depth_index]
			ftm_vectors = ftm_vectors[ftm_index]
			ftm_vectors_mask = ftm_vectors_mask[ftm_index]
			aff_mat[:,:-1] = aff_mat[:,:-1][:, depth_index] # depth is on axis 1
			aff_mat[:-1] = aff_mat[:-1][ftm_index] # ftm is on axis 0
			
		### append extra cell for masks
		ftm_vectors_mask = np.concatenate((ftm_vectors_mask, np.array([True])))
		depth_vectors_mask = np.concatenate((depth_vectors_mask, np.array([True])))
		
		if np.sum(np.isnan(ftm_vectors)): 
			print("ftm_vectors contains nan")
			exit()
		if np.sum(np.isnan(depth_vectors)):
			print("depth_vectors contains nan")
			print(depth_vectors)
			print(depth_index, np.sum(np.isnan(depth_index)))
			exit()

		sample = {'DepthFrameName': frame_name,
				"depth_vectors": depth_vectors,
				"depth_vectors_mask": depth_vectors_mask,
				"ftm_vectors": ftm_vectors,
				"ftm_vectors_mask": ftm_vectors_mask,
				"aff_mat": aff_mat}

		if self.transform:
			sample = self.transform(sample)

		return sample



	def __len__(self):
		return len(self.matching_gnd)