import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
import os
from skimage import io
from skimage import img_as_float
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import myModel_v50
import myMultimodalDataset_v50
from sklearn import metrics
import argparse
import datetime
from torchviz import make_dot
from matplotlib import rc
from torch.utils.tensorboard import SummaryWriter

rc('font',family='serif')
rc('text', usetex=True)



""" ********************************** main ********************************** """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', help='directory to the dataset csv file, default is data_10_11_aug')
parser.add_argument('milestone_dir', help='directory to save partial trained models')
parser.add_argument('--lr', type=float, help='learning rate, default is 0.0001')
parser.add_argument('--epoch', type=int, help='number of epoch you want to train, default is 100')
parser.add_argument('--milestone', help='to extract starting epoch number, if it is 0, will read the ImageNet pretrained model, otherwise read accordingly from the milestones')
parser.add_argument('--batchSize', type=int, help='batchsize, default is 32')
parser.add_argument('--fold', type=int, help='which fold in N-fold cross validation, default is 1')
args = parser.parse_args()

if args.dataset:
    dataset_dir = args.dataset
else:
    dataset_dir = "/media/hans/SamsungDisk/multimodal_learning_datasets/dataset_v50/5-folds_csv_seq_shuf/"

if args.batchSize:
	train_batch_size = args.batchSize
else:
	train_batch_size = 32

if args.epoch:
	train_number_epochs = args.epoch
else:
	train_number_epochs = 200

if args.lr:
	learning_rate = args.lr
else:
	learning_rate = 0.001

if args.fold:
	fold = args.fold
else:
	fold = 1

milestone_dir = args.milestone_dir # end with "/"
if not os.path.exists(milestone_dir):
	os.makedirs(milestone_dir)


# writer = None
writer = SummaryWriter(comment="_"+milestone_dir.split('/')[-1])

torch.backends.cudnn.enabled = True
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("detected device: ", device)

if args.milestone:
	# print("Continue training from ", archs[archID], args.milestone)
	model = torch.load(args.milestone)
	starting_epoch = int(args.milestone.split("/")[-1].split(".")[0].split('_')[-1])
	# print(starting_epoch)
	# exit()
else:
	model = myModel_v50.MultimodalNetwork()
	starting_epoch = 0

model = model.cuda()
criterion = myModel_v50.AffinityLoss().cuda()
# print(model)


train_dataset = myMultimodalDataset_v50.myMutimodalDataset(gndPKLPath=dataset_dir + 'train_fold'+str(fold)+'.pkl', phase="train")
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=train_batch_size)

# train_dataset = myMultimodalDataset_v50.myMutimodalDataset(gndPKLPath=dataset_dir + 'test_fold'+str(fold)+'.pkl', phase="train")
# train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=train_batch_size)

val_dataset = myMultimodalDataset_v50.myMutimodalDataset(gndPKLPath=dataset_dir + 'test_fold'+str(fold)+'.pkl', phase="test")
val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=0, batch_size=train_batch_size)

optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[train_number_epochs//2], gamma=0.1)

accuracy_max = 0
best_so_far_pth = "_"
for epoch in range(starting_epoch+1, starting_epoch+1 + train_number_epochs):
	scheduler.step()
	# epoch_list.append(epoch)
	print("Now entering epoch: ", epoch)
	### train phase
	model.train()
	for i, train_data in enumerate(train_dataloader, 0):
		# print('batch', i)
		depth_vectors, depth_vectors_mask, ftm_vectors, ftm_vectors_mask, aff_mat = train_data['depth_vectors'], train_data['depth_vectors_mask'], train_data['ftm_vectors'], train_data['ftm_vectors_mask'], train_data['aff_mat']

		depth_vectors = depth_vectors.to(device, dtype=torch.float)
		depth_vectors_mask = depth_vectors_mask.to(device)
		ftm_vectors = ftm_vectors.to(device, dtype=torch.float)
		ftm_vectors_mask = ftm_vectors_mask.to(device)
		aff_mat = aff_mat.to(device, dtype=torch.float)

		# print(depth_vectors, torch.sum(torch.isnan(depth_vectors)))
		# print(ftm_vectors, torch.sum(torch.isnan(ftm_vectors)))
		# print(depth_vectors.size())
		# print(depth_vectors_mask.size())
		# print(ftm_vectors)
		# print(ftm_vectors_mask)
		# print(aff_mat)

		# print(ftm_vectors.size(), depth_vectors.size())
		# output = model(ftm_vectors, depth_vectors )

		
		output = model(depth_vectors, ftm_vectors, depth_vectors_mask, ftm_vectors_mask)
		# print("output", output, torch.sum(torch.isnan(output)))
		
		loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = criterion(output, aff_mat, ftm_vectors_mask, depth_vectors_mask)
		if writer:
			writer.add_scalar("Training/Loss", loss, epoch)
			writer.add_scalar("Training/Acc", accuracy, epoch)

		# print("loss", loss)
		# print("loss_pre", loss_pre)
		# print("loss_next", loss_next)
		# print("loss_similarity", loss_similarity)
		optimizer.zero_grad()
		loss.backward()
		# for p in model.parameters():
		# 	print("grad")
		# 	print(p)

		
		# torch.nn.utils.clip_grad_norm_(model.parameters(), 1.00, error_if_nonfinite=True)
		optimizer.step()
		# if i == 3:
		# 	exit()

		if (i+1) %10 == 0 :
			# print("gnd:", aff_mat)
			# print('prediction', output)

			print("Epoch: {}, Iteration: {}, bs: {}, lr: {}, Current loss {}, Current Acc {}".format(epoch, i+1, train_batch_size, optimizer.param_groups[0]['lr'], loss.item(), accuracy.item()))
			# exit()



	### validation phase: validate on training set
	model.eval()

	for i, train_data in enumerate(val_dataloader, 0):
		depth_vectors, depth_vectors_mask, ftm_vectors, ftm_vectors_mask, aff_mat = train_data['depth_vectors'], train_data['depth_vectors_mask'], train_data['ftm_vectors'], train_data['ftm_vectors_mask'], train_data['aff_mat']

		depth_vectors = depth_vectors.to(device, dtype=torch.float)
		depth_vectors_mask = depth_vectors_mask.to(device)
		ftm_vectors = ftm_vectors.to(device, dtype=torch.float)
		ftm_vectors_mask = ftm_vectors_mask.to(device)
		aff_mat = aff_mat.to(device, dtype=torch.float)

		# print(ftm_vectors, depth_vectors)
		# output = model(ftm_vectors, depth_vectors )
		output = model(depth_vectors, ftm_vectors, depth_vectors_mask, ftm_vectors_mask)
		# print("output", output, torch.sum(torch.isnan(output)))
		
		loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = criterion(output, aff_mat, ftm_vectors_mask, depth_vectors_mask)
		if writer:
			writer.add_scalar("Test/Loss", loss, epoch)
			writer.add_scalar("Test/Acc", accuracy, epoch)

	if (epoch) % 10 == 0:
		model_name = "epoch_"+str(epoch)+".pth"
		torch.save(model, os.path.join(milestone_dir, model_name)) # save model

	if accuracy != 1 and accuracy > accuracy_max:
		### delete the old pth
		os.system('rm '+os.path.join(milestone_dir, best_so_far_pth))

		### update the accuracy_max and best_so_far_pth
		accuracy_max = accuracy
		best_so_far_pth = "best_so_far_epoch_"+str(epoch)+"_acc_" + str(accuracy.data.cpu().numpy())+".pth"
		torch.save(model, os.path.join(milestone_dir, best_so_far_pth)) # save model		


