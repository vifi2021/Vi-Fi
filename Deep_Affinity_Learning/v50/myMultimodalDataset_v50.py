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

class myResize(object):
	def __init__(self):
		pass
	def __call__(self, sample):
		uvd_mat = sample['uvd_mat']
		uvd_mat = cv2.resize(uvd_mat, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
		sample["uvd_mat"] = uvd_mat
		return sample


class myRandomHorizontalFlip():
	def __init__(self):
		pass
	def __call__(self, sample, p=0.5):
		uvd_mat = sample['uvd_mat']
		if random.uniform(0, 1) <= p:
			cv2.flip(uvd_mat, 1)
			sample["uvd_mat"] = uvd_mat
		return sample

class myRandomVerticalFlip():
	def __init__(self):
		pass
	def __call__(self, sample, p=0.5):
		uvd_mat = sample['uvd_mat']
		if random.uniform(0, 1) <= p:
			cv2.flip(uvd_mat, 0)
			sample["uvd_mat"] = uvd_mat
		return sample

class myToTensor():
	def __init__(self):
		pass
	def __call__(self, sample):
		uvd_mat, ftm_vector = sample['uvd_mat'], sample['ftm_vector']
		tsfm = transforms.ToTensor()
		uvd_mat = tsfm(uvd_mat)
		# print(uvd_mat)
		ftm_vector = tsfm(ftm_vector)
		sample["uvd_mat"] = uvd_mat
		sample["ftm_vector"] = ftm_vector
		return sample



class myMutimodalDataset(Dataset):

	def __init__(self, gndPKLPath, Nm_phone=5, Nm_camera=15, phase="train", transform=None):
		self.matching_gnd = pd.read_pickle(gndPKLPath)
		self.transform = transform
		self.phase=phase
		self.Nm_phone = Nm_phone
		self.Nm_camera = Nm_camera
		self.window_size = 10
		# print(self.matching_gnd)

	def __getitem__(self, idx):
		# print(self.matching_gnd.iloc[idx, 0])
		frame_name = self.matching_gnd.iloc[idx, 0]
		### depth vector and mask
		depth_vectors_dict = self.matching_gnd.iloc[idx, 1]
		# print(depth_vectors_dict)
		depth_vectors = np.array(list(depth_vectors_dict.values()))
		# print(depth_vectors)
		depth_vectors_lengths = np.array([len(x) for x in depth_vectors]).reshape(-1, 1)/5
		# print(depth_vectors_lengths.shape)
		# print(depth_vectors_lengths)
		# print(depth_vectors_lengths)
		if depth_vectors.dtype == 'object':
			### pad the depth vectors to uniform seq length
			for i in range(len(depth_vectors)):
				depth_vectors[i] = np.pad(depth_vectors[i], (0, 5*self.window_size - len(depth_vectors[i])),'constant')
			depth_vectors = np.array([v for v in depth_vectors])
		else:
			depth_vectors = np.pad(depth_vectors, ((0, 0), (0, 5*self.window_size - depth_vectors.shape[1])), 'constant')
		# append the valid length to the end of each vector
		depth_vectors = np.append(depth_vectors, depth_vectors_lengths, axis=1)
		# print(depth_vectors.shape[1])
		# depth_vectors = np.append(depth_vectors, np.ones((depth_vectors.shape[0], 1)), axis=1)
		# print(depth_vectors)

		# depth_vectors = depth_vectors / np.linalg.norm(depth_vectors, axis=1)[:, np.newaxis]
		depth_vectors_mask = np.array([True] * depth_vectors.shape[0])
		depth_num = depth_vectors.shape[0]
		
		### ftm vector and mask
		ftm_vectors_dict = self.matching_gnd.iloc[idx, 2]
		# print(ftm_vectors_dict)
		ftm_vectors = np.array(list(ftm_vectors_dict.values()))
		# ftm_vectors = ftm_vectors / np.linalg.norm(ftm_vectors, axis=1)[:, np.newaxis]
		ftm_vectors_lengths = np.array([len(x)for x in ftm_vectors]).reshape(-1, 1)/11
		# print(ftm_vectors_lengths)
		if ftm_vectors.dtype == 'object':
			### pad the depth vectors to uniform seq length
			for i in range(len(ftm_vectors)):
				ftm_vectors[i] = np.pad(ftm_vectors[i], (0, 11*self.window_size - len(ftm_vectors[i])),'constant')
			ftm_vectors = np.array([v for v in ftm_vectors])
		else:
			ftm_vectors = np.pad(ftm_vectors, ((0, 0), (0, 11*self.window_size - ftm_vectors.shape[1])), 'constant')
		# append the valid length to the end of each vector
		ftm_vectors = np.append(ftm_vectors, ftm_vectors_lengths, axis=1)
		# ftm_vectors = np.append(ftm_vectors, np.ones((ftm_vectors.shape[0], 1)), axis=1)


		ftm_vectors_mask = np.array([True] * ftm_vectors.shape[0])
		ftm_num = ftm_vectors.shape[0]

		### affinity matrix dict
		aff_mat = self.matching_gnd.iloc[idx, -1] 

		# print(depth_vectors.shape) # depth map dim: N(people) x 64 x 64 x k(consecutive)
		# print(ftm_vectors.shape) # ftm map: N(people) x 20 
		
		### expand the number of depth/ftm vectors to Nm
		depth_vectors_mask = np.concatenate((depth_vectors_mask, np.array([False]*(self.Nm_camera - depth_vectors.shape[0]))))
		depth_vectors = np.vstack((depth_vectors, np.ones((self.Nm_camera - depth_vectors.shape[0], depth_vectors.shape[1]))))
		# depth_vectors = np.vstack((depth_vectors, np.zeros((self.Nm_camera - depth_vectors.shape[0], depth_vectors.shape[1]))))
		# depth_vectors = np.vstack((depth_vectors, np.append(np.random.rand(self.Nm_camera - depth_vectors.shape[0], depth_vectors.shape[1]-1), np.ones((self.Nm_camera - depth_vectors.shape[0], 1)), axis=1)))
		ftm_vectors_mask = np.concatenate((ftm_vectors_mask, np.array([False]*(self.Nm_phone - ftm_vectors.shape[0]))))
		ftm_vectors = np.vstack((ftm_vectors, np.ones((self.Nm_phone - ftm_vectors.shape[0], ftm_vectors.shape[1]))))
		# ftm_vectors = np.vstack((ftm_vectors, np.zeros((self.Nm_phone - ftm_vectors.shape[0], ftm_vectors.shape[1]))))
		# ftm_vectors = np.vstack((ftm_vectors, np.append(np.random.rand(self.Nm_phone - ftm_vectors.shape[0], ftm_vectors.shape[1]-1), np.ones((self.Nm_phone - ftm_vectors.shape[0], 1)), axis=1)))
		# print(depth_vectors.shape)
		# print(depth_vectors)
		# print(depth_vectors_mask)
		# print(ftm_vectors.shape)
		# print(ftm_vectors_mask)
		# print(aff_mat)

		### shuffle the order of the vectors
		r = random.uniform(0, 1)
		if r < 0:
			# print(r)
			# ### shuffle all Nm cols and Nm rows
			# ftm_index = np.arange(self.Nm)
			# np.random.shuffle(ftm_index)
			# depth_index = np.arange(self.Nm)
			# np.random.shuffle(depth_index)
			# # print(ftm_index)
			# # print(depth_index)

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
			
			# print(ftm_index)
			# print(ftm_vectors_mask)
			# print(depth_index)
			# print(aff_mat)

		### append extra cell for masks
		ftm_vectors_mask = np.concatenate((ftm_vectors_mask, np.array([True])))
		depth_vectors_mask = np.concatenate((depth_vectors_mask, np.array([True])))
		
		# print(depth_vectors)
		# print(depth_vectors_mask)
		# print(ftm_vectors)
		# print(ftm_vectors_mask)
		# print(aff_mat)

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