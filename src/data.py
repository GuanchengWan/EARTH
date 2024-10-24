import sys
import torch
import numpy as np
from torch.autograd import Variable
import controldiffeq

            
import os
from fastdtw import fastdtw
from tqdm import tqdm

class DataBasicLoader(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window # 20
        self.h = args.horizon # 1
        self.d = 0
        self.dataset = args.dataset
        self.add_his_day = False
        self.rawdat = np.loadtxt(open("../data/{}.txt".format(args.dataset)), delimiter=',')
        print('data shape', self.rawdat.shape)
        if args.sim_mat:
            self.load_sim_mat(args)
            
        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape # n_sample, n_group
        # print(self.n, self.m)

        self.scale = np.ones(self.m)

        self._pre_train(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        print('size of train/val/test sets',len(self.train[0]),len(self.val[0]),len(self.test[0]))
    
        self.dtw_delta = self.num_nodes - 5
        
        if args.cuda:
            self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).cuda()
            sem_mask = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta]
        for i in range(self.sem_mask.shape[0]):
            self.sem_mask[i][sem_mask[i]] = 0
            
    
    
    
    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(open("../data/{}.txt".format(args.sim_mat)), delimiter=','))
        self.orig_adj = self.adj
        rowsum = 1. / torch.sqrt(self.adj.sum(dim=0))
        self.adj = rowsum[:, np.newaxis] * self.adj * rowsum[np.newaxis, :]
        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()
            self.orig_adj = self.orig_adj.cuda()

    def _pre_train(self, train, valid, test):
        self.train_set = train_set = range(self.P+self.h-1, train)
        self.valid_set = valid_set = range(train, valid)
        self.test_set = test_set = range(valid, self.n)
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy() #199, 47
        
        
        cache_path = './dtw_' + self.dataset + '.npy'
        if os.path.exists(cache_path):
            dtw_matrix = np.load(cache_path)
            self.num_nodes = self.m
            print('Loaded DTW matrix from {}'.format(cache_path))
        else:
            # 计算训练集中的平均时间序列
            data_mean = self.rawdat.reshape(self.rawdat.shape[0], self.rawdat.shape[1], 1)
            self.num_nodes = self.m
            # 计算节点之间的 DTW 距离矩阵
            dtw_matrix = np.zeros((self.num_nodes, self.num_nodes))
            for i in tqdm(range(self.num_nodes)):
                for j in range(i, self.num_nodes):
                    dtw_distance, _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
                    dtw_matrix[i][j] = dtw_distance

            # 对称化 DTW 矩阵
            for i in range(self.num_nodes):
                for j in range(i):
                    dtw_matrix[i][j] = dtw_matrix[j][i]

            # 保存 DTW 矩阵到缓存文件
            np.save(cache_path, dtw_matrix)
            print('Saved DTW matrix to {}'.format(cache_path))
        
        self.dtw_matrix = dtw_matrix

        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        self.peak_thold = np.mean(train_mx, 0)
        # normalize
        self.dat  = (self.rawdat  - self.min ) / (self.max  - self.min + 1e-12)
        # print(self.dat.shape)
        
        
    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        if (train == valid):
            self.val = self.test
 
    def _batchify(self, idx_set, horizon, useraw=False): ###tonights work

        n = len(idx_set)
        Y = torch.zeros((n, self.m))
        if self.add_his_day and not useraw:
            X = torch.zeros((n, self.P+1, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P

            if useraw: # for narmalization
                X[i,:self.P,:] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i,:] = torch.from_numpy(self.rawdat[idx_set[i], :])
            else:
                his_window = self.dat[start:end, :]
                if self.add_his_day:
                    if idx_set[i] > 51 : # at least 52
                        his_day = self.dat[idx_set[i]-52:idx_set[i]-51, :] #
                    else: # no history day data
                        his_day = np.zeros((1,self.m))

                    his_window = np.concatenate([his_day,his_window])
                    # print(his_window.shape,his_day.shape,idx_set[i],idx_set[i]-52,idx_set[i]-51)
                    X[i,:self.P+1,:] = torch.from_numpy(his_window) # size (window+1, m)
                else:
                    X[i,:self.P,:] = torch.from_numpy(his_window) # size (window, m)
                Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    # original
    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt,:]
            Y = targets[excerpt,:]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)

            data = [model_inputs, Variable(Y)]
            yield data
            start_idx += batch_size


class DataCDELoader(DataBasicLoader):
    def __init__(self, args):
        super().__init__(args)
        
        self.times = torch.linspace(0, self.train[0].size(1)-1, self.train[0].size(1))
        # self.train[0] = self.train[0].transpose(1,2)
        # self.val[0] = self.val[0].transpose(1,2)
        # self.test[0] = self.test[0].transpose(1,2)
        
        # Augment input data with times
        self.augment_data_with_times()

        # Interpolate time series data using natural cubic splines
        self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(self.times, self.train[0].transpose(1,2))
        self.val_coeffs = controldiffeq.natural_cubic_spline_coeffs(self.times, self.val[0].transpose(1,2))
        self.test_coeffs = controldiffeq.natural_cubic_spline_coeffs(self.times, self.test[0].transpose(1,2))

        self.train.append(self.train_coeffs)
        self.val.append(self.val_coeffs)
        self.test.append(self.test_coeffs)
        
        
        
    def augment_data_with_times(self):
        # Augment training data
        augmented_X_tra = [self.times.unsqueeze(0).unsqueeze(0).repeat(self.train[0].size(0), self.train[0].size(2), 1).unsqueeze(-1).transpose(1,2)]
        augmented_X_tra.append(torch.Tensor(self.train[0][..., :]).unsqueeze(-1))
        self.train[0] = torch.cat(augmented_X_tra, dim=3)

        # Augment validation data
        augmented_X_val = [self.times.unsqueeze(0).unsqueeze(0).repeat(self.val[0].size(0), self.val[0].size(2), 1).unsqueeze(-1).transpose(1,2)]
        augmented_X_val.append(torch.Tensor(self.val[0][..., :]).unsqueeze(-1))
        self.val[0] = torch.cat(augmented_X_val, dim=3)

        # Augment test data
        augmented_X_test = [self.times.unsqueeze(0).unsqueeze(0).repeat(self.test[0].size(0), self.test[0].size(2), 1).unsqueeze(-1).transpose(1,2)]
        augmented_X_test.append(torch.Tensor(self.test[0][..., :]).unsqueeze(-1))
        self.test[0] = torch.cat(augmented_X_test, dim=3)

        
        
    def get_batches(self, data, batch_size, shuffle=True):
        # Use coefficients instead of raw data
        inputs = data[0]
        targets = data[1]
        coeffs = data[2]  # Extract coefficients tuple from data
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt,:]
            Y = targets[excerpt,:]
            C = tuple(c[excerpt,:] for c in coeffs)  # Extract coefficients for this batch
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
                C = tuple(c.cuda() for c in C)
            model_inputs = Variable(X)

            data = [model_inputs, Variable(Y), C] # Include coefficients in the returned data
            yield data
            start_idx += batch_size
