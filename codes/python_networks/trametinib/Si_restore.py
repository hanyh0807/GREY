# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore
from scipy.spatial.distance import hamming
from multiprocessing import Pool, Process, Queue, Array, Value, Event
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import pickle
import gc
import itertools as it
import subprocess
import re
import json
import networkx as nx

#import attractor
from . import attractor

class simul():
    def __init__(self, num_initial, drug = 'drug', start_process = True, mb_size = 8, y_scaling=False, use_randsign=False, max_task_num=10000, name=None, scale=5, network_file=None, profile_file=None, targets=None):
        self.name = name
        if type(drug) is not list:
            self.drug = [drug]
        else:
            self.drug = drug
        self.num_initial = num_initial
        self.eps = 1e-3
        self.use_randsign = use_randsign
        self.scale = scale
        self.max_task_num = Value('d', 0)
        self.max_task_num.value = max_task_num
        
        self.adjmat = pd.read_csv(network_file, header=0,index_col=0)
        self.weightmat = self.adjmat.values
        input_link_num = np.abs(self.weightmat).sum(axis=1)
        input_link_num_order = np.argsort(input_link_num)
        self.weightmat = self.weightmat[input_link_num_order]
        self.weightmat = self.weightmat[:,input_link_num_order]
        self.basal = 0
        self.Nodes = self.adjmat.columns.ravel().astype(np.str)
        self.Nodes = self.Nodes[input_link_num_order]

        wmidx = np.where(self.weightmat)
        wmidx = np.concatenate([st.reshape([-1,1]) for st in wmidx], axis=1)
        self.weightmat_link_idx = Array('d', 1000) #maximum 500 indices
        self.weightmat_link_idx[:np.count_nonzero(self.weightmat)*2] = wmidx.reshape([-1])

        self.sign_mask = self.weightmat.copy()
        self.sign_mask = np.sign(self.sign_mask)

        self.use_v4 = True
        self.outmut_initial = False
        
        pert_data = pd.read_csv(profile_file, header=0, index_col=0)
        remove = ~pd.isna(pert_data.loc['AUC'])
        pert_data = pert_data.loc[:,remove]
        pdcols = list(pert_data.columns)
        pdcols = [str(x) for x in pdcols]
        pert_data.columns = pdcols

        status_base_temp = pert_data.iloc[1:,:]
        status_base = pd.DataFrame(data=np.zeros((self.Nodes.size,status_base_temp.shape[1])), index=self.Nodes, columns=status_base_temp.columns)
        status_base.loc[status_base_temp.index] = status_base_temp
        
        self.num_input_node = np.sum(input_link_num==0)
        #drugtarget = np.array([['MAP2K1', 'MAP2K2']])
        drugtarget = np.array([targets])
        total_pert_nodes = np.unique(np.concatenate(drugtarget))
        train_y_house = []
        perturbation_house = []
        status_house = []
        for di, d in enumerate(self.drug):
            #train_y_temp = np.log(pert_data.loc[d])
            #train_y_temp = np.log(pert_data.loc['IC50'])
            train_y_temp = pert_data.loc['AUC'] #GDSC, small AUC == small IC50
            valididx = ~pd.isna(train_y_temp)
            train_y_temp = train_y_temp.to_frame().transpose().loc[:,valididx]
            train_y_temp.index=[0]
            #train_y_temp = (train_y_temp - train_y_temp.values.min()) / (train_y_temp.values.max() - train_y_temp.values.min())
            train_y_house.append(train_y_temp)
            
            perttemplate = np.zeros((total_pert_nodes.size,1))
            for dt in drugtarget[di]:
                perttemplate[total_pert_nodes==dt] = 1
            perturbation_temp = pd.DataFrame(np.tile(perttemplate, (train_y_temp.shape[1])), index=total_pert_nodes, columns=train_y_temp.columns+'_'+d)
            perturbation_house.append(perturbation_temp)
            
            status_temp = status_base.loc[:,status_base.columns[valididx]]
            status_temp.columns = status_temp.columns + '_' + d
            status_house.append(status_temp)
            
        self.train_y_whole = pd.concat(train_y_house, axis=1, ignore_index=False)
        self.perturbation_whole = pd.concat(perturbation_house, axis=1, ignore_index=False)
        self.status_whole = pd.concat(status_house, axis=1, ignore_index=False)
    
        self.valsize = int(self.perturbation_whole.shape[1] * 0.2)
        self.testsize = int(self.perturbation_whole.shape[1] * 0.2)
        np.random.seed(0)
        tvsplit = np.random.permutation(self.perturbation_whole.shape[1])

        trcut = self.testsize
        self.perturbation = self.perturbation_whole.iloc[:,tvsplit[:-trcut]]
        self.status = self.status_whole.iloc[:,tvsplit[:-trcut]]
        self.train_y = self.train_y_whole.iloc[:,tvsplit[:-trcut]]
        #self.train_y_exp = exp_data.iloc[:,tvsplit[:-trcut]]

        start = self.valsize + self.testsize
        end = self.testsize
        self.perturbation_val = self.perturbation_whole.iloc[:,tvsplit[-start:-end]]
        self.status_val = self.status_whole.iloc[:,tvsplit[-start:-end]]
        self.train_y_val = self.train_y_whole.iloc[:,tvsplit[-start:-end]]
        #self.train_y_exp_val = exp_data.iloc[:,tvsplit[-start:-end]]

        self.perturbation_test = self.perturbation_whole.iloc[:,tvsplit[-self.testsize:]]
        self.status_test = self.status_whole.iloc[:,tvsplit[-self.testsize:]]
        self.train_y_test = self.train_y_whole.iloc[:,tvsplit[-self.testsize:]]
        #self.train_y_exp_test = exp_data.iloc[:,tvsplit[-self.testsize:]]

        yscaler = MinMaxScaler((-1,1))
        yscaler.fit(self.train_y.T)
        self.train_y.iloc[:] = yscaler.transform(self.train_y.T).T
        self.train_y_val.iloc[:] = yscaler.transform(self.train_y_val.T).T
        self.train_y_test.iloc[:] = yscaler.transform(self.train_y_test.T).T

        y_index_name = ['Proliferation'] #  positively effects on viability 
        self.y_index = [] # index of measured nodes at weightmat
        for i in y_index_name:
            self.y_index.append(np.where(i == self.Nodes)[0][0])

        self.d_pert_idx = []
        for i in self.perturbation.index:
            self.d_pert_idx.append(np.where(i == self.Nodes)[0][0])
        
        self.num_sample = self.perturbation.shape[1]
        #mini_batch_size = 8
        
#        self.initials = np.array(attractor.generate_initial_state_all(self.Nodes,self.num_sample))
        self.initials = []
        self.dummy_y = []
        self.dummy_exp = []
        self.dummy_status = []
        self.dummy_perturbation = []
        self.dummy_y_index = [] # index of measured nodes at weightmat
        self.dontidx = []

        self.zc = Queue()
        self.rq = Queue()
        self.pq = Queue()
        self.sq = Queue()
        self.eq = Queue()
        self.esq = Queue()
        self.emq = Queue()
        self.params = Array('d', 1000)
        self.dummy_params = Array('d', 1000)
        self.dummy_link_idx = Array('d', 2000) #maximum 500 indices
        self.dummy_params_num = Value('d', 0)
        self.dummy_nodes_num = Value('d', 0)
        self.dummy_pheno_idx = Array('d', 10)
        self.dummy_pert_idx = Array('d', 100)
        self.dummy_pert_num = Value('d', 0)
        self.averact = Array('f', 500)
        self.averact_one = Array('f', 500)
        self.averact_mone = Array('f', 500)
        self.tvt = Value('d', 0) # 0 - train, 1 - val, 2 - test
        self.seed = Value('d', 0)
        self.taskseed = Value('d', 0)
        self.mb_size = Value('d', mb_size)
        self.maxnode = Value('d', 41)
        self.do_reset_initials = Value('d',0)
        self.do_initialize_task = Value('d',0)
        self.START_CALC_EVENT = Event()
        self.START_CALC_EVENT.clear()

        if start_process:
            ptrain = Process(target=self.return_reward_2att)
            ptrain.start()
#            pval = Process(target=self.return_reward_val_notmut)
#            ptest = Process(target=self.return_reward_test_notmut)
            
#            self.plist = [ptrain, pval, ptest]
#            for p in self.plist:
#                p.start()
    
    def get_num_params(self):
        param = np.sum(self.weightmat!=0)
        node = len(self.Nodes) - self.num_input_node
        return param, node

    def get_link_scale(self, level=3, scale=4):
        matidx = self.weightmat!=0
        maxmat = np.zeros_like(self.weightmat)
        
        if self.use_randsign:
            maxmat[matidx] = scale * 2
        else:
            maxmat[matidx] = scale
        #maxmat[matidx] = scale
        
        action_width = maxmat[matidx].astype(np.float)
        action_center = action_width - scale
        
        return action_width, action_center
    
    def reset_initials(self):
        np.random.seed()
        self.initials = np.array(attractor.generate_initial_state_all(self.Nodes[self.num_input_node:],self.num_initial))
        self.initials = np.concatenate((np.zeros([self.num_initial, self.num_input_node], dtype='int32'), self.initials),axis=1)
        if self.use_v4:
            self.initials[self.initials==0] = -1

    def dummy_reset_initials(self, numnode):
        np.random.seed()
        self.initials = np.array(attractor.generate_initial_state_all(np.zeros(numnode),self.num_initial))
        if self.use_v4:
            self.initials[self.initials==0] = -1

    def return_reward_2att(self):
        pool = Pool(8)
        while True:
            self.START_CALC_EVENT.wait()
            self.START_CALC_EVENT.clear()
            self.reset_initials()
            if self.do_reset_initials.value == 1:
                self.reset_initials()
                self.do_reset_initials.value = 0
            a_pred = np.array(self.params[:].copy())
            seed = int(self.seed.value)
            
            current_sm = self.sign_mask
            current_origin = self.weightmat[self.weightmat!=0]
            cur_y_index = self.y_index
            #cur_y_exp_index = self.y_exp_index
            if self.tvt.value == 0:
                np.random.seed(seed)
                #mini_batch_size = int(self.mb_size.value)
                mini_batch_size = 300
                miniidx = np.random.permutation(self.num_sample)[:mini_batch_size]
                data_mini = self.train_y.iloc[:,miniidx].values
                pert_mini = self.perturbation.iloc[:,miniidx]!=0
                status_mini = self.status.iloc[:,miniidx]
            elif self.tvt.value == 1:
                data_mini = self.train_y_val.values
                pert_mini = self.perturbation_val!=0
                status_mini = self.status_val
                mini_batch_size = self.valsize
            elif self.tvt.value == 2:
                data_mini = self.train_y_test.values
                pert_mini = self.perturbation_test!=0
                status_mini = self.status_test
                mini_batch_size = self.testsize
            else:
                print('ERROR!!!!')
                exit(1)

            observation = []
            prediction = []
            first_att_prol_act = []
            newpa = np.array(current_sm.copy())
            newpa[newpa!=0] = np.round(a_pred[:np.count_nonzero(current_sm)]).astype(np.int)

            if not self.use_randsign:
                newpa = newpa * current_sm
            newba = np.zeros(self.Nodes.size)

            tvt = self.tvt.value
            results = pool.starmap(calcatt, zip(np.arange(mini_batch_size),
            it.repeat(data_mini), it.repeat(pert_mini), it.repeat(status_mini), it.repeat(newpa),
            it.repeat(self.Nodes), it.repeat(self.initials), it.repeat(self.outmut_initial),
            it.repeat(tvt)), chunksize=1)
            results = np.array(results, dtype=np.object)

            ob = results[:,0]
            yy_aver = np.concatenate([np.expand_dims(at,0) for at in results[:,1]], axis=0)
            yy_p_aver = np.concatenate([np.expand_dims(at,0) for at in results[:,2]], axis=0)

            viability = np.mean(yy_p_aver[:,cur_y_index], axis=1) - np.mean(yy_aver[:,cur_y_index], axis=1)
            total_change = np.mean(np.sqrt((yy_p_aver - yy_aver)**2))

            observation = data_mini.ravel()
            prediction = viability
            first_att_prol_act = np.mean(yy_aver[:,cur_y_index])

            #cur_y_exp_index = np.array(cur_y_exp_index)
            expcorr, expspear, expmi, exp_e = [], [], [], []
            fcc, fcs, fcm = [], [], []

            observation = np.array(observation).reshape([-1])
            prediction = np.array(prediction).reshape([-1])
            cor = np.corrcoef(observation, prediction)[0,1]
            cors = spearmanr(observation, prediction)[0]
            mi = mutual_info_regression(observation.reshape([-1,1]), prediction.reshape([-1]))
            if np.isnan(cor):
                cor = 0.0
            if np.isnan(cors):
                cors = 0.0
            if np.isnan(mi):
                mi = 0.0
               
            if cors < 0:
                cor = 0.0
            reward = cor
            pea = cor
            spear = cors

            self.rq.put(np.array(reward))
            self.sq.put(np.array(spear))
            self.zc.put(np.array(mi))


def calcatt(i, data_mini, pert_mini, status_mini, newpa, Nodes, initials, outmut_initial, tvt):
    status_pert_nodes = list(status_mini.index[status_mini.iloc[:,i]==-1])
    status_stim_nodes = list(status_mini.index[status_mini.iloc[:,i]==1])
    initial_status_pert_nodes = list(status_mini.index[status_mini.iloc[:,i]==-2])
    initial_status_stim_nodes = list(status_mini.index[status_mini.iloc[:,i]==2])
                
    stim_idx, pert_idx = [], []
    initial_stim_idx, initial_pert_idx = [], []
    if tvt != 3:
        for pn in status_pert_nodes:
            pert_idx.append(np.where(Nodes==pn)[0][0])
        for sn in status_stim_nodes:
            stim_idx.append(np.where(Nodes==sn)[0][0])

        for pn in initial_status_pert_nodes:
            initial_pert_idx.append(np.where(Nodes==pn)[0][0])
        for sn in initial_status_stim_nodes:
            initial_stim_idx.append(np.where(Nodes==sn)[0][0])

    else:
        stim_idx = status_stim_nodes
        pert_idx = status_pert_nodes
        initial_stim_idx = initial_status_stim_nodes
        initial_pert_idx = initial_status_pert_nodes

    stim_idx_np = np.array(stim_idx)
    pert_idx_np = np.array(pert_idx)
    initial_stim_idx_np = np.array(initial_stim_idx)
    initial_pert_idx_np = np.array(initial_pert_idx)

    cur_initials = initials.copy()
    if outmut_initial:
        if initial_stim_idx_np.size > 0:
            cur_initials[:,initial_stim_idx_np] = 1
        if initial_pert_idx_np.size > 0:
            cur_initials[:,initial_pert_idx_np] = -1
        cur_initials = np.unique(cur_initials, axis=0)

    yy, _ = np.array(attractor.find_attractor_pert_v4(newpa, cur_initials, pert_idx_np, stim_idx_np, outmut_initial, initial_pert_idx_np, initial_stim_idx_np), dtype=np.object)
    yy_unique = np.concatenate(yy,axis=0)
    yy_unique, counts = np.unique(yy_unique, axis=0, return_counts=True)
    yy_aver = np.average(yy_unique,axis=0,weights=counts)
    yy_aver += 1

    for iy in range(len(yy)):
        if len(yy[iy]) > 1:
            #yy[iy] = yy[iy][np.random.choice(len(yy[iy]))].reshape([1,-1])
            yy[iy] = yy[iy][0].reshape([1,-1])

    yy_unique_for_pert = np.concatenate(yy, axis=0)
    yy_unique_for_pert, counts = np.unique(yy_unique_for_pert, axis=0, return_counts=True)

    pert_nodes = [s for s in list(pert_mini.index[pert_mini.iloc[:,i]])]
    if tvt != 3:
        for pn in pert_nodes:
            pert_idx.append(np.where(Nodes==pn)[0][0])
    else:
        pert_idx.extend(pert_nodes)
   
    # Remove activated nodes if they are perturbed
    def check_and_remove_idx(remove_target, pert_idx):
        inters = np.intersect1d(remove_target, pert_idx)
        if inters.shape[0] > 0:
            for inin in inters:
                remove_target.remove(inin)
        return remove_target

    stim_idx = check_and_remove_idx(stim_idx, pert_idx)
    initial_stim_idx = check_and_remove_idx(initial_stim_idx, pert_idx)
    initial_pert_idx = check_and_remove_idx(initial_pert_idx, pert_idx)

    stim_idx_np = np.unique(np.array(stim_idx))
    pert_idx_np = np.unique(np.array(pert_idx))
    initial_stim_idx_np = np.array(initial_stim_idx)
    initial_pert_idx_np = np.array(initial_pert_idx)
    
    yy_p, _ = np.array(attractor.find_attractor_pert_v4(newpa, yy_unique_for_pert, pert_idx_np, stim_idx_np, outmut_initial, initial_pert_idx_np, initial_stim_idx_np), dtype=np.object)
    el = [x.shape[0] for x in yy_p]
    yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))
    yy_p_aver += 1
    
    #scale 0~1
    yy_aver = yy_aver / 2
    yy_p_aver = yy_p_aver / 2
    return data_mini[:,i], yy_aver, yy_p_aver
