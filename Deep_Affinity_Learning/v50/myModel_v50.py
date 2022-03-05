import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MultimodalNetwork(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(MultimodalNetwork, self).__init__()
        self.false_constant = 1
        self.lstm_feature_dim = 32

        self.lstm_v = nn.LSTM(3, self.lstm_feature_dim, num_layers=2, batch_first=True, bidirectional=True) # only use bbox_depth, bbox_x, bbox_y to train
        self.lstm_f = nn.LSTM(11, self.lstm_feature_dim, num_layers=2, batch_first=True, bidirectional=True)

        ### compression network
        self.final= nn.Sequential(
            # nn.BatchNorm2d(self.lstm_feature_dim*2),
            nn.Conv2d(self.lstm_feature_dim * 2, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(),

            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),


            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 16, kernel_size=1, stride=1),
            nn.ReLU(True),

            nn.Conv2d(16, 1, kernel_size=1, stride=1),
            nn.ReLU(True),
        )
        self.Nm_phone = 5
        self.Nm_camera = 15

        

    def forward(self, x_v, x_f, depth_vectors_mask, ftm_vectors_mask):
        # x_f: B x Nm_phone x (11*k+1)
        # x_v: B x Nm_camera x (5*k+1)
        # permutation_cubic: B x 2k x Nm_phone x Nm_camera
        # output B x 1 x Nm_phone x Nm_camera (affinity matrix)

        # print(x_v.size())
        x_f_len = x_f[:, :, -1] # B x Nm (x 1)
        x_v_len = x_v[:, :, -1] # B x Nm (x 1)
        # print('x_v_len:', x_v_len)

        x_f = x_f[:, :, :-1] # B x Nm x (11*k)
        x_v = x_v[:, :, :-1] # B x Nm x (5*k)

        x_f = x_f.view(x_f.size()[0], x_f.size()[1], -1, 11)
        x_v = x_v.view(x_v.size()[0], x_v.size()[1], -1, 5)
        x_v = x_v[:, :, :, 0:3]#  only use bbox_depth, bbox_x, bbox_y to train
        ### x_v: B x Nm x k(10) x 3
        ### x_f: B x Nm x k(10) x 11
        # print(x_v.shape, x_f.shape)
        output_v_mat = torch.zeros(x_v.size(0), self.Nm_camera, self.lstm_feature_dim).cuda()
        output_f_mat = torch.zeros(x_f.size(0), self.Nm_phone, self.lstm_feature_dim).cuda()

        # ### train the lstm with fixed sequence length
        # for i in range(x_v.size()[1]): # for every bounding box
        #     for j in range(x_v.size()[-1]): # for every timestamp
        #         output_v = x_v[:, i, :, j]
        #         output_v, hidden_v = self.lstm_v(output_v.view(output_v.size(0), 1, output_v.size(1)), hidden_v)
        #     output_v_mat[:, i, :] = output_v.view(output_v.size(0), -1)
            
        # for i in range(x_f.size(1)):
        #     for j in range(x_f.size()[-1]):
        #         output_f = x_f[:, i, :, j]
        #         output_f, hidden_f = self.lstm_f(output_f.view(output_f.size(0), 1, output_f.size(1)), hidden_f)
        #         # print(output_v.size(), output_f.size())
        #     output_f_mat[:, i, :] = output_f.view(output_f.size(0), -1)

        # print(x_v)
        # print(x_f)

        ### train the lstm with varying sequence length
        for i in range(x_v.size(1)): # for every bounding box
            hidden_v = (torch.zeros(2*2, x_v.size(0), self.lstm_feature_dim).cuda(), torch.zeros(2*2, x_v.size(0), self.lstm_feature_dim).cuda())
            seq_len = x_v_len[:, i].cpu().long()
            # print(seq_len)
            packed_x_v = pack_padded_sequence(x_v[:, i, :], seq_len, batch_first=True, enforce_sorted=False)
            output_v, hidden_v = self.lstm_v(packed_x_v, hidden_v)
            # print(hidden_v[0][-1].size())
            # output_v, lens = pad_packed_sequence(output_v, batch_first=True)
            output_v_mat[:, i, :] = hidden_v[0][-1]


            # for j in range(x_v.size()[-2]): # for every timestamp
            #     output_v = x_v[:, i, j, :]
            #     output_v, hidden_v = self.lstm_v(output_v.view(output_v.size(0), 1, output_v.size(1)), hidden_v)
            # output_v_mat[:, i, :] = output_v.view(output_v.size(0), -1)
            
        for i in range(x_f.size(1)):
            hidden_f = (torch.zeros(2*2, x_f.size(0), self.lstm_feature_dim).cuda(), torch.zeros(2*2, x_f.size(0), self.lstm_feature_dim).cuda())
            seq_len = x_f_len[:, i].cpu().long()
            packed_x_f = pack_padded_sequence(x_f[:, i, :], seq_len, batch_first=True, enforce_sorted=False)
            output_f, hidden_f = self.lstm_f(packed_x_f, hidden_f)
            # print(hidden_f[0][-1].size())
            # output_v, lens = pad_packed_sequence(output_v, batch_first=True)
            output_f_mat[:, i, :] = hidden_f[0][-1]
            # print(output_f_mat[:,i,:])

            # for j in range(x_f.size()[-2]):
            #     output_f = x_f[:, i, j, :]
            #     output_f, hidden_f = self.lstm_f(output_f.view(output_f.size(0), 1, output_f.size(1)), hidden_f)
            #     # print(output_v.size(), output_f.size())
            # output_f_mat[:, i, :] = output_f.view(output_f.size(0), -1)
        # for p in self.lstm_f.parameters():
        #     print(p)
        #     print(p.grad)
        # exit()

        # print(output_f_mat.size(), output_v_mat.size())
        # print(output_v_mat)
        # exit()
        # print(output_f_mat)

        ### according to mask, reset the dummy object as 0 vector
        # print(depth_vectors_mask[:, :-1].size())
        # print(depth_vectors_mask)
        valid_depth_mask = depth_vectors_mask[:, :-1].unsqueeze(2).repeat(1, 1, output_v_mat.size(-1)).float()
        # print(valid_depth_mask)
        # print(valid_depth_mask[-1])
        output_v_mat = output_v_mat * valid_depth_mask
        # print(output_v_mat.size())
        # print(depth_vectors_mask.size())
        # output_v_mat[depth_vectors_mask[:,:-1]] = torch.rand(self.lstm_feature_dim).cuda()
        # print(output_v_mat[depth_vectors_mask[:,:-1]])
        # print(torch.rand(self.lstm_feature_dim).cuda())
        # print(output_v_mat)

        valid_ftm_mask = ftm_vectors_mask[:, :-1].unsqueeze(2).repeat(1, 1, output_f_mat.size(-1)).float()
        output_f_mat = output_f_mat * valid_ftm_mask
        # output_f_mat[ftm_vectors_mask[:, :-1]] = torch.rand(self.lstm_feature_dim).cuda()

        # print(depth_vectors_mask[-1])
        # print(output_f_mat[-1])
        # print(ftm_vectors_mask[-1])
        # print(output_v_mat[-1])
        ### forward the compression network
        output_f_mat = output_f_mat.unsqueeze(2).repeat(1, 1, self.Nm_camera, 1).permute(0, 3, 1, 2)
        output_v_mat = output_v_mat.unsqueeze(1).repeat(1, self.Nm_phone, 1, 1).permute(0, 3, 1, 2)
        # print(output_f_mat.size(), output_v_mat.size())
        permutation_cubic = torch.cat([output_f_mat, output_v_mat],1)
        # print(x_f.size())
        # print(x_v[-1, :, :, :])
        # print(x_v)
        # print(permutation_cubic.size())

        ### feed the permutation_cubic to the conv net for dimension reduction
        # output = self.final(permutation_cubic)
        x = permutation_cubic
        x = x.contiguous()
        for f in self.final:
            x = f(x)
        # print(f.weight.grad, x)
        output = x
        
        ### add extra row and col for the Nm x Nm affinity mat to be (Nm+1) x (Nm+1)
        # print("output", output)
        output = self.add_unmatched_dim(output)
        # print("output\n", output[-1, -1, :, :])
        # print(output)

        return output

    def add_unmatched_dim(self, x):
        # if self.false_objects_column is None:
        #     self.false_objects_column = Variable(torch.ones(x.shape[0], x.shape[1], x.shape[2], 1)) * self.false_constant
        #     if self.use_gpu:
        #         self.false_objects_column = self.false_objects_column.cuda()
        # x = torch.cat([x, self.false_objects_column], 3)

        # if self.false_objects_row is None:
        #     self.false_objects_row = Variable(torch.ones(x.shape[0], x.shape[1], 1, x.shape[3])) * self.false_constant
        #     if self.use_gpu:
        #         self.false_objects_row = self.false_objects_row.cuda()
        # x = torch.cat([x, self.false_objects_row], 2)
        false_objects_column = Variable(torch.ones(x.shape[0], x.shape[1], x.shape[2], 1)) * self.false_constant
        false_objects_column = false_objects_column.cuda()
        x = torch.cat([x, false_objects_column], 3)
        false_objects_row = Variable(torch.ones(x.shape[0], x.shape[1], 1, x.shape[3])) * self.false_constant
        false_objects_row = false_objects_row.cuda()
        x = torch.cat([x, false_objects_row], 2)
        return x


class AffinityLoss(nn.Module):
    def __init__(self, Nm_phone=5, Nm_camera=15):
        super(AffinityLoss, self).__init__()
        self.Nm_phone = Nm_phone
        self.Nm_camera = Nm_camera
        
    def forward(self, input, target, mask0, mask1):
        '''
        input: the predicted affinity matrix after network forward: (Nm_phone+1) x (Nm_camera+1)
        target: the gnd affinity matrix: (Nm_phone+1) x (Nm_camera+1)
        mask0: the valid ftm mask: (1, Nm_phone+1)
        mask1: the valid bbox mask: (1, Nm_camera+1)

        '''
        mask0 = torch.unsqueeze(mask0, 1) # (1, 1, Nm_phone)
        # print(mask0)
        mask1 = torch.unsqueeze(mask1, 1) # (1, 1, Nm_camera)
        # print(mask1)
        target = torch.unsqueeze(target, 1) # (1, Nm_phone, Nm_camera)
        # print("mask0", mask0[-1,-1,:])
        # print("target\n", target[-1, -1, :, :])

        mask_pre = mask0[:, :, :]
        mask_next = mask1[:, :, :]
        mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.Nm_camera+1)
        mask1 = mask1.unsqueeze(2).repeat(1, 1, self.Nm_phone+1, 1)
        mask0 = Variable(mask0.data)
        mask1 = Variable(mask1.data)
        target = Variable(target.byte().data)

        # mask0 = mask0.cuda()
        # mask1 = mask1.cuda()

        # print(mask0)
        # print(mask1)
        # print(mask0.size(), mask1.size())
        mask_region = (mask0 * mask1).float() # the valid position mask
        
        mask_region_pre = mask_region.clone() 
        mask_region_pre[:, :, self.Nm_phone, :] = 0
        # print("mask region pre", mask_region_pre[-1, -1, :, :], torch.sum(torch.isnan(mask_region_pre)))
        
        mask_region_next = mask_region.clone() 
        mask_region_next[:, :, :, self.Nm_camera] = 0
        
        mask_region_union = mask_region_pre*mask_region_next

        # print("input\n", input[-1, -1, :, :]),
        # print("mask", mask_region_next[-1, -1, :])
        # print(target[-1, -1, :])
        # exit()

        input_pre = mask_region_pre*input
        # print("input_pre before Softmax", input_pre[-1, -1, :], torch.sum(torch.isnan(input_pre)))
        # input_pre = input_pre / torch.norm(input_pre, dim=3, keepdim=True)
        input_pre = nn.Softmax(dim=3)(input_pre)
        # print('input pre after softmax\n', input_pre[-1, -1, :])
        # print(input_pre.size())

        input_next = mask_region_next*input
        # print("input_next before Softmax", input_next[-1, -1, :])
        # input_next = input_next / torch.norm(input_next, dim=2, keepdim=True)
        input_next = nn.Softmax(dim=2)(input_next)
        # print('input_next after softmax\n', input_next[-1, -1, :])        
        # print(input_next.size())

        # input_pre = nn.Softmax(dim=3)(mask_region_pre*input)
        # input_next = nn.Softmax(dim=2)(mask_region_next*input)

        # print("input_pre", input_pre)
        # print("input_next", input_next)

        # input_pre = mask_region_pre*input_pre
        # input_next = mask_region_next*input_next

        # print(input_pre)

        input_all = input_pre.clone()
        input_all[:, :, :self.Nm_phone, :self.Nm_camera] = torch.max(input_pre, input_next)[:, :, :self.Nm_phone, :self.Nm_camera]
        # input_all[:, :, :self.Nm_phone, :self.Nm_camera] = ((input_pre + input_next)/2.0)[:, :, :self.Nm_phone, :self.Nm_camera]
        target = target.float()
        target_pre = mask_region_pre * target
        target_next = mask_region_next * target
        target_union = mask_region_union * target
        
        target_num = target.sum()
        target_num_pre = target_pre.sum()
        target_num_next = target_next.sum()
        target_num_union = target_union.sum()
        
        # print('target_pre', target_pre[-1, -1, :])
        if int(target_num_pre):
            loss_pre = - (target_pre * torch.log(input_pre)).sum() / target_num_pre
        else:
            loss_pre = - (target_pre * torch.log(input_pre)).sum()
        # print(loss_pre)

        if int(target_num_next):
            loss_next = - (target_next * torch.log(input_next)).sum() / target_num_next
        else:
            loss_next = - (target_next * torch.log(input_next)).sum()

        # if int(target_num_pre) and int(target_num_next):
        #     loss = -(target_pre * torch.log(input_all)).sum() / target_num_pre
        # else:
        #     loss = -(target_pre * torch.log(input_all)).sum()
        if int(target_num_union):
            loss = -(target_union * torch.log(input_all)).sum() / target_num_union
        else:
            loss = -(target_union * torch.log(input_all)).sum()

        if int(target_num_union):
            loss_similarity = (target_union * (torch.abs((input_pre) - (input_next)))).sum() / target_num
        else:
            loss_similarity = (target_union * (torch.abs((input_pre) - (input_next)))).sum()
        # print(loss)

        # if int(target_num_union):
        #     loss_sparsity = (target_union * (torch.abs((input_all) ))).sum() / target_num
        # else:
        #     loss_sparsity = (target_union * (torch.abs((input_all) ))).sum()


        ### acc computation
        _, indexes_ = target_pre.max(3)
        # print(target_pre, _, indexes_)
        # print(mask_pre)
        indexes_ = indexes_[:, :, :-1]
        # print(target_pre.size())
        # print(indexes_.size())
        _, indexes_pre = input_pre.max(3)
        indexes_pre = indexes_pre[:, :, :-1]
        # print(indexes_.size())
        mask_pre_num = mask_pre[:, :, :-1].sum()
        # print(mask_pre_num)
        mask_pre = mask_pre.bool()
        # print(mask_pre)
        if mask_pre_num:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:,:, :-1]]).float().sum() / mask_pre_num
        else:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]).float().sum() + 1

        _, indexes_ = target_next.max(2)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = input_next.max(2)
        indexes_next = indexes_next[:, :, :-1]
        mask_next_num = mask_next[:, :, :-1].sum()
        mask_next = mask_next.bool()
        # print(mask_next)
        if mask_next_num:
            accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() / mask_next_num
        else:
            accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() + 1

        accuracy = (accuracy_pre + accuracy_next)/2.0
        # accuracy = max(accuracy_pre, accuracy_next)
        # print(accuracy)

        return loss_pre, loss_next, loss_similarity, (loss_pre + loss_next + loss + loss_similarity)/4.0, accuracy_pre, accuracy_next, accuracy, indexes_pre



class AffinityLoss_test(nn.Module):
    def __init__(self, Nm_phone=5, Nm_camera=15):
        super(AffinityLoss_test, self).__init__()
        self.Nm_phone = Nm_phone
        self.Nm_camera = Nm_camera
        
    def forward(self, input, target, mask0, mask1):
        '''
        input: the predicted affinity matrix after network forward: (Nm_phone+1) x (Nm_camera+1)
        target: the gnd affinity matrix: (Nm_phone+1) x (Nm_camera+1)
        mask0: the valid ftm mask: (1, Nm_phone+1)
        mask1: the valid bbox mask: (1, Nm_camera+1)

        '''
        correct_count_fc = 0
        total_count_fc = 0
        correct_count_cf = 0
        total_count_cf = 0

        mask0 = torch.unsqueeze(mask0, 1) # (1, 1, Nm_phone)
        mask1 = torch.unsqueeze(mask1, 1) # (1, 1, Nm_camera)
        target = torch.unsqueeze(target, 1) # (1, Nm_phone, Nm_camera)
        # print("mask0", mask0[-1,-1,:])
        # print("target", target[-1, -1, :, :])

        mask_pre = mask0[:, :, :]
        mask_next = mask1[:, :, :]
        mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.Nm_camera+1)
        mask1 = mask1.unsqueeze(2).repeat(1, 1, self.Nm_phone+1, 1)
        mask0 = Variable(mask0.data)
        mask1 = Variable(mask1.data)
        target = Variable(target.byte().data)

        # mask0 = mask0.cuda()
        # mask1 = mask1.cuda()

        # print(mask0)
        # print(mask1)
        # print(mask0.size(), mask1.size())
        mask_region = (mask0 * mask1).float() # the valid position mask
        
        mask_region_pre = mask_region.clone() 
        mask_region_pre[:, :, self.Nm_phone, :] = 0
        # print("mask region pre", mask_region_pre[-1, -1, :, :], torch.sum(torch.isnan(mask_region_pre)))
        
        mask_region_next = mask_region.clone() 
        mask_region_next[:, :, :, self.Nm_camera] = 0
        
        mask_region_union = mask_region_pre*mask_region_next

        # print("input", input[-1, -1, :, :]),
        # print("mask", mask_region_next[-1, -1, :])
        # print(target[-1, -1, :])

        input_pre = mask_region_pre*input
        # print("input_pre before Softmax", input_pre[-1, -1, :], torch.sum(torch.isnan(input_pre)))
        # input_pre = input_pre / torch.norm(input_pre, dim=3, keepdim=True)
        input_pre = nn.Softmax(dim=3)(input_pre)
        # print('input pre after softmax\n', input_pre[-1, -1, :])

        input_next = mask_region_next*input
        # print("input_next before Softmax", input_next[-1, -1, :])
        # input_next = input_next / torch.norm(input_next, dim=2, keepdim=True)
        input_next = nn.Softmax(dim=2)(input_next)
        # print('input_next after softmax\n', input_next)        

        # input_pre = nn.Softmax(dim=3)(mask_region_pre*input)
        # input_next = nn.Softmax(dim=2)(mask_region_next*input)

        # print("input_pre", input_pre)
        # print("input_next", input_next)

        # input_pre = mask_region_pre*input_pre
        # input_next = mask_region_next*input_next

        # print(input_pre)

        input_all = input_pre.clone()
        input_all[:, :, :self.Nm_phone, :self.Nm_camera] = torch.max(input_pre, input_next)[:, :, :self.Nm_phone, :self.Nm_camera]
        # input_all[:, :, :self.Nm, :self.Nm] = ((input_pre + input_next)/2.0)[:, :, :self.Nm, :self.Nm]
        target = target.float()
        target_pre = mask_region_pre * target
        target_next = mask_region_next * target
        target_union = mask_region_union * target
        
        target_num = target.sum()
        target_num_pre = target_pre.sum()
        target_num_next = target_next.sum()
        target_num_union = target_union.sum()
        
        # print('target_pre', target_pre[-1, -1, :])
        if int(target_num_pre):
            loss_pre = - (target_pre * torch.log(input_pre)).sum() / target_num_pre
        else:
            loss_pre = - (target_pre * torch.log(input_pre)).sum()
        # print(loss_pre)

        if int(target_num_next):
            loss_next = - (target_next * torch.log(input_next)).sum() / target_num_next
        else:
            loss_next = - (target_next * torch.log(input_next)).sum()

        # if int(target_num_pre) and int(target_num_next):
        #     loss = -(target_pre * torch.log(input_all)).sum() / target_num_pre
        # else:
        #     loss = -(target_pre * torch.log(input_all)).sum()
        if int(target_num_union):
            loss = -(target_union * torch.log(input_all)).sum() / target_num_union
        else:
            loss = -(target_union * torch.log(input_all)).sum()

        if int(target_num_union):
            loss_similarity = (target_union * (torch.abs((input_pre) - (input_next)))).sum() / target_num
        else:
            loss_similarity = (target_union * (torch.abs((input_pre) - (input_next)))).sum()
        # print(loss)

        # if int(target_num_union):
        #     loss_sparsity = (target_union * (torch.abs((input_all) ))).sum() / target_num
        # else:
        #     loss_sparsity = (target_union * (torch.abs((input_all) ))).sum()


        ### acc computation
        _, indexes_ = target_pre.max(3)
        # print(target_pre, _, indexes_)
        # print(mask_pre)
        indexes_ = indexes_[:, :, :-1]
        # print(target_pre.size())
        # print(indexes_.size())
        _, indexes_pre = input_pre.max(3)
        indexes_pre = indexes_pre[:, :, :-1]
        # print(indexes_.size())
        mask_pre_num = mask_pre[:, :, :-1].sum()
        # print(mask_pre_num)
        mask_pre = mask_pre.bool()
        # print(input_pre)
        # print(mask_pre)
        if mask_pre_num:
            accuracy_fc = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:,:, :-1]]).float().sum() / mask_pre_num
            correct_count_fc += (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:,:, :-1]]).float().sum()
            total_count_fc += mask_pre_num
            fc_association = indexes_pre[mask_pre[:, :, :-1]]
            fc_association_gnd = indexes_[mask_pre[:,:, :-1]]
        else:
            accuracy_fc = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]).float().sum() + 1

        _, indexes_ = target_next.max(2)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = input_next.max(2)
        indexes_next = indexes_next[:, :, :-1]
        mask_next_num = mask_next[:, :, :-1].sum()
        mask_next = mask_next.bool()
        # print(input_next)
        # print(mask_next)
        if mask_next_num:
            accuracy_cf = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() / mask_next_num
            correct_count_cf += (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum()
            total_count_cf += mask_next_num
            cf_association = indexes_next[mask_next[:, :, :-1]]
            cf_association_gnd = indexes_[mask_next[:, :, :-1]]
        else:
            accuracy_cf = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() + 1

        accuracy = (accuracy_fc + accuracy_cf)/2.0
        # accuracy = max(accuracy_fc, accuracy_cf)
        # print(accuracy)

        # return loss_pre, loss_next, loss_similarity, (loss_pre + loss_next + loss + loss_similarity)/4.0, accuracy_fc, accuracy_cf, accuracy, indexes_pre
        return [fc_association, fc_association_gnd, 
                cf_association, cf_association_gnd, 
                correct_count_fc, total_count_fc, accuracy_fc, 
                correct_count_cf, total_count_cf, accuracy_cf,
                accuracy ]



class AffinityLoss_test_bipartite(nn.Module):
    def __init__(self, Nm=5):
        super(AffinityLoss_test_bipartite, self).__init__()
        self.Nm = Nm
        
    def forward(self, input, target, mask0, mask1):
        '''
        input: the predicted affinity matrix after network forward: (Nm+1) x (Nm+1)
        target: the gnd affinity matrix: (Nm+1) x (Nm+1)
        mask0: the valid ftm mask: (1, Nm+1)
        mask1: the valid bbox mask: (1, Nm+1)

        '''
        correct_count_fc = 0
        total_count_fc = 0
        correct_count_cf = 0
        total_count_cf = 0

        mask0 = torch.unsqueeze(mask0, 1) # (1, 1, Nm)
        mask1 = torch.unsqueeze(mask1, 1) # (1, 1, Nm)
        target = torch.unsqueeze(target, 1) # (1, Nm, Nm)
        # print("mask0", mask0[-1,-1,:])
        # print("target", target[-1, -1, :, :])

        mask_pre = mask0[:, :, :]
        mask_next = mask1[:, :, :]
        mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.Nm+1)
        mask1 = mask1.unsqueeze(2).repeat(1, 1, self.Nm+1, 1)
        mask0 = Variable(mask0.data)
        mask1 = Variable(mask1.data)
        target = Variable(target.byte().data)

        # mask0 = mask0.cuda()
        # mask1 = mask1.cuda()

        # print(mask0)
        # print(mask1)
        # print(mask0.size(), mask1.size())
        mask_region = (mask0 * mask1).float() # the valid position mask
        
        mask_region_pre = mask_region.clone() 
        mask_region_pre[:, :, self.Nm, :] = 0
        # print("mask region pre", mask_region_pre[-1, -1, :, :], torch.sum(torch.isnan(mask_region_pre)))
        
        mask_region_next = mask_region.clone() 
        mask_region_next[:, :, :, self.Nm] = 0
        
        mask_region_union = mask_region_pre*mask_region_next

        # print("input", input[-1, -1, :, :]),
        # print("mask", mask_region_next[-1, -1, :])
        # print(target[-1, -1, :])

        input_pre = mask_region_pre*input
        # print("input_pre before Softmax", input_pre[-1, -1, :], torch.sum(torch.isnan(input_pre)))
        # input_pre = input_pre / torch.norm(input_pre, dim=3, keepdim=True)
        input_pre = nn.Softmax(dim=3)(input_pre)
        # print('input pre after softmax\n', input_pre[-1, -1, :])

        input_next = mask_region_next*input
        # print("input_next before Softmax", input_next[-1, -1, :])
        # input_next = input_next / torch.norm(input_next, dim=2, keepdim=True)
        input_next = nn.Softmax(dim=2)(input_next)
        # print('input_next after softmax\n', input_next)        

        # input_pre = nn.Softmax(dim=3)(mask_region_pre*input)
        # input_next = nn.Softmax(dim=2)(mask_region_next*input)

        # print("input_pre", input_pre)
        # print("input_next", input_next)

        # input_pre = mask_region_pre*input_pre
        # input_next = mask_region_next*input_next

        # print(input_pre)

        input_all = input_pre.clone()
        input_all[:, :, :self.Nm, :self.Nm] = torch.max(input_pre, input_next)[:, :, :self.Nm, :self.Nm]
        # input_all[:, :, :self.Nm, :self.Nm] = ((input_pre + input_next)/2.0)[:, :, :self.Nm, :self.Nm]
        target = target.float()
        target_pre = mask_region_pre * target
        target_next = mask_region_next * target
        target_union = mask_region_union * target
        
        target_num = target.sum()
        target_num_pre = target_pre.sum()
        target_num_next = target_next.sum()
        target_num_union = target_union.sum()
        
        # print('target_pre', target_pre[-1, -1, :])
        if int(target_num_pre):
            loss_pre = - (target_pre * torch.log(input_pre)).sum() / target_num_pre
        else:
            loss_pre = - (target_pre * torch.log(input_pre)).sum()
        # print(loss_pre)

        if int(target_num_next):
            loss_next = - (target_next * torch.log(input_next)).sum() / target_num_next
        else:
            loss_next = - (target_next * torch.log(input_next)).sum()

        # if int(target_num_pre) and int(target_num_next):
        #     loss = -(target_pre * torch.log(input_all)).sum() / target_num_pre
        # else:
        #     loss = -(target_pre * torch.log(input_all)).sum()
        if int(target_num_union):
            loss = -(target_union * torch.log(input_all)).sum() / target_num_union
        else:
            loss = -(target_union * torch.log(input_all)).sum()

        if int(target_num_union):
            loss_similarity = (target_union * (torch.abs((input_pre) - (input_next)))).sum() / target_num
        else:
            loss_similarity = (target_union * (torch.abs((input_pre) - (input_next)))).sum()
        # print(loss)

        # if int(target_num_union):
        #     loss_sparsity = (target_union * (torch.abs((input_all) ))).sum() / target_num
        # else:
        #     loss_sparsity = (target_union * (torch.abs((input_all) ))).sum()



        ### acc computation
        _, indexes_ = target_pre.max(3)
        # print(target_pre, _, indexes_)
        # print(mask_pre)
        indexes_ = indexes_[:, :, :-1]
        # print(target_pre.size())
        # print(indexes_.size())
        _, indexes_pre = input_pre.max(3)
        indexes_pre = indexes_pre[:, :, :-1]
        # print(indexes_.size())
        mask_pre_num = mask_pre[:, :, :-1].sum()
        # print(mask_pre_num)
        mask_pre = mask_pre.bool()
        # print(mask_pre)
        # print(input_pre)
        # print(input_pre[mask_pre])
        if mask_pre_num:
            accuracy_fc = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:,:, :-1]]).float().sum() / mask_pre_num
            # correct_count_fc += (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:,:, :-1]]).float().sum()
            total_count_fc += mask_pre_num
            fc_association = indexes_pre[mask_pre[:, :, :-1]]
            fc_association_gnd = indexes_[mask_pre[:,:, :-1]]
        else:
            accuracy_fc = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]).float().sum() + 1

        _, indexes_ = target_next.max(2)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = input_next.max(2)
        indexes_next = indexes_next[:, :, :-1]
        mask_next_num = mask_next[:, :, :-1].sum()
        mask_next = mask_next.bool()
        # print(mask_next)
        # print(input_next)
        # print(input_next[:, :, :, mask_next[0, 0, :]])
        if mask_next_num:
            accuracy_cf = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() / mask_next_num
            # correct_count_cf += (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum()
            total_count_cf += mask_next_num
            cf_association = indexes_next[mask_next[:, :, :-1]]
            cf_association_gnd = indexes_[mask_next[:, :, :-1]]

        else:
            accuracy_cf = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() + 1

        accuracy = (accuracy_fc + accuracy_cf)/2.0
        # accuracy = max(accuracy_fc, accuracy_cf)
        # print(accuracy)

        ### one to one association for fc and cf association
        aff_mat_fc = input_pre[mask_pre]
        aff_mat_cf = input_next[:, :, :, mask_next[0, 0, :]]
        aff_mat_fc = aff_mat_fc.view(aff_mat_fc.size(-2), aff_mat_fc.size(-1))
        aff_mat_cf = aff_mat_cf.view(aff_mat_cf.size(-2), aff_mat_cf.size(-1))
        aff_mat_fc = aff_mat_fc[:-1, :]
        aff_mat_cf = aff_mat_cf[:, :-1]
        # print(input_pre)
        # print(aff_mat_fc)
        # print(input_next)
        # print(aff_mat_cf)
        
        row_ind_fc, col_ind_fc = linear_sum_assignment(-aff_mat_fc.cpu().detach().numpy())
        fc_association = torch.Tensor(col_ind_fc).long()
        # print(fc_association)
        # print(fc_association_gnd)

        row_ind_cf, col_ind_cf = linear_sum_assignment(-aff_mat_cf.cpu().detach().numpy())
        cf_association = torch.Tensor(row_ind_cf).long()
        # print(cf_association)
        # print(cf_association_gnd)

        # print((torch.Tensor(fc_association) == fc_association_gnd.cpu()))
        accuracy_fc = (fc_association == fc_association_gnd.cpu()).float().sum() / len(fc_association_gnd)
        correct_count_fc += (fc_association == fc_association_gnd.cpu()).float().sum()
        # total_count_fc += len(fc_association_gnd)

        accuracy_cf = (cf_association == cf_association_gnd.cpu()).float().sum() / len(cf_association_gnd)
        correct_count_cf += (cf_association == cf_association_gnd.cpu()).float().sum()
        # total_count_cf += len(cf_association_gnd)

        # return loss_pre, loss_next, loss_similarity, (loss_pre + loss_next + loss + loss_similarity)/4.0, accuracy_fc, accuracy_cf, accuracy, indexes_pre
        # print([fc_association, fc_association_gnd, 
        #         cf_association, cf_association_gnd, 
        #         correct_count_fc, total_count_fc, accuracy_fc, 
        #         correct_count_cf, total_count_cf, accuracy_cf,
        #         accuracy ])
        return [fc_association, fc_association_gnd, 
                cf_association, cf_association_gnd, 
                correct_count_fc, total_count_fc, accuracy_fc, 
                correct_count_cf, total_count_cf, accuracy_cf,
                accuracy ]