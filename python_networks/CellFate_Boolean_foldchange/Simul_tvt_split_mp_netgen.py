# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import distance
from multiprocessing import Process, Queue, Array, Value, Event
from sklearn.feature_selection import mutual_info_regression
import networkx as nx


#import attractor
from . import attractor

class simul():
    def __init__(self, num_initial, start_process = True, mb_size = 8, y_scaling=False, max_task_num=10000, use_randsign=False, scale = 5):
        self.num_initial = num_initial
        self.y_scaling = y_scaling
        #self.max_task_num = max_task_num
        self.max_task_num = Value('d', 0)
        self.max_task_num.value = max_task_num
        self.use_randsign = use_randsign
        self.scale = scale
        
        self.weightmat = np.array([[0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0],
                     [1,1,0,0,-3,0,0,0,0,0,0],
                     [0,0,2,0,0,2,0,-2,0,0,0],
                     [1,1,0,-4,0,0,0,1,0,0,0],
                     [0,0,0,1,0,1,0,0,0,-3,0],
                     [0,0,0,0,0,0,0,0,0,0,-2],
                     [0,0,0,-3,0,0,2,0,0,2,0],
                     [0,0,1,-3,0,0,0,0,0,0,1],
                     [0,0,0,-2,2,0,0,0,0,0,4],
                     [0,0,0,-2,0,0,0,0,1,0,0]])
        self.basal = np.array([[0],[0],[0],[-3],[0],[0],[1],[-3],[0],[-1],[0]])
#        self.weightmat_backbone = self.adjmat.values
        self.Nodes = np.array(['TNF', 'FAS', 'RIP1', 'NFkB', 'C8', 'cIAP', 'ATP', 'C3', 'ROS', 'MOMP', 'MPT'])

        wmidx = np.where(self.weightmat)
        wmidx = np.concatenate([st.reshape([-1,1]) for st in wmidx], axis=1)
        self.weightmat_link_idx = Array('d', 1000) #maximum 500 indices
        self.weightmat_link_idx[:np.count_nonzero(self.weightmat)*2] = wmidx.reshape([-1])

        self.posratio = np.sum(self.weightmat>0) / np.count_nonzero(self.weightmat)

        self.num_input_node = 2

        self.sign_mask = self.weightmat.copy()
        self.sign_mask = np.sign(self.sign_mask)
            
        self.use_v4 = True
        if self.use_v4:
            self.pert_data = np.transpose(pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/simulated_fc_v4.csv', header=0, index_col=None))
        else:
            self.pert_data = np.transpose(pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/simulated_fc.csv', header=0, index_col=None))
        self.perturbation_whole = self.pert_data.iloc[:11,:]
        self.train_y_whole = self.pert_data.iloc[11:,:]

        self.valsize = int(self.perturbation_whole.shape[1] * 0.2)
        self.testsize = int(self.perturbation_whole.shape[1] * 0.2)
        np.random.seed(0)
        tvsplit = np.random.permutation(self.perturbation_whole.shape[1])

        trcut = self.testsize
        self.perturbation = self.perturbation_whole.iloc[:,tvsplit[:-trcut]]
        self.train_y = self.train_y_whole.iloc[:,tvsplit[:-trcut]]

        start = self.valsize + self.testsize
        end = self.testsize
        self.perturbation_val = self.perturbation_whole.iloc[:,tvsplit[-start:-end]]
        self.train_y_val = self.train_y_whole.iloc[:,tvsplit[-start:-end]]

        self.perturbation_test = self.perturbation_whole.iloc[:,tvsplit[-self.testsize:]]
        self.train_y_test = self.train_y_whole.iloc[:,tvsplit[-self.testsize:]]

        if self.y_scaling:
            #base = self.train_y.abs().max(axis=1).values
            #tra = self.train_y.values / (base.reshape([-1,1]) * 2) + 0.5
            #tea = self.train_y_test.values / (base.reshape([-1,1]) * 2) + 0.5

            base = self.train_y.abs().max().max()
            tra = self.train_y.values / (base * 2) + 0.5
            tva = self.train_y_val.values / (base * 2) + 0.5
            tea = self.train_y_test.values / (base * 2) + 0.5

            self.train_y = pd.DataFrame(data=tra, columns=self.train_y.columns, index = self.train_y.index)
            self.train_y_val = pd.DataFrame(data=tva, columns=self.train_y_val.columns, index = self.train_y_val.index)
            self.train_y_test = pd.DataFrame(data=tea, columns=self.train_y_test.columns, index = self.train_y_test.index)

        y_index_name = [s.split(':')[1] for s in list(self.train_y.index)]
        self.y_index = [] # index of measured nodes at weightmat
        for i in y_index_name:
            self.y_index.append(np.where(i == self.Nodes)[0][0])
        
        self.num_sample = self.perturbation.shape[1]
        #mini_batch_size = 8
        
#        self.initials = np.array(attractor.generate_initial_state_all(self.Nodes,self.num_sample))
        self.initials = []
        self.dummy_y = []
        self.dummy_perturbation = []
        self.dummy_y_index = [] # index of measured nodes at weightmat
        self.dontidx = []

        self.zc = Queue()
        self.rq = Queue()
        self.sq = Queue()
        self.params = Array('d', 1500)
        self.dummy_params = Array('d', 1500)
        self.dummy_link_idx = Array('d', 3000) #maximum 1500 indices
        self.dummy_params_num = Value('d', 0)
        self.dummy_nodes_num = Value('d', 0)
        self.averact = Array('f', 500)
        self.tvt = Value('d', 0) # 0 - train, 1 - val, 2 - test
        self.seed = Value('d', 0)
        self.taskseed = Value('d', 0)
        self.mb_size = Value('d', mb_size)
        self.do_reset_initials = Value('d',0)
        self.do_initialize_task = Value('d',0)
        self.START_CALC_EVENT = Event()
        self.START_VAL_CALC_EVENT = Event()
        self.START_TEST_CALC_EVENT = Event()
        self.START_CALC_EVENT.clear()
        self.START_VAL_CALC_EVENT.clear()
        self.START_TEST_CALC_EVENT.clear()

        if start_process:
            ptrain = Process(target=self.return_reward_notmut)
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
        #return 10, node

    def get_link_scale(self, level=3, scale=4):
        matidx = self.weightmat!=0
        maxmat = np.zeros_like(self.weightmat)
        
        if self.use_randsign:
            maxmat[matidx] = scale * 2
        else:
            maxmat[matidx] = scale
#        maxmat[self.probadjmat>=level] = scale
        
        action_width = maxmat[matidx].astype(np.float)
        action_center = action_width - scale
        
#        self.sign_mask[matidx] = 1
#        self.sign_mask[self.probadjmat>=level] = self.weightmat[self.probadjmat>=level]
#        self.sign_mask[self.probadjmat<level] = self.weightmat[self.probadjmat<level]
        
        return action_width, action_center
        #return np.array([scale, scale, scale]), np.array([0])
        #return np.array([scale]*10), np.array([0])
    
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

    def initialize_task(self, mb_size=16, pert=None):
        reset_dummy_weightmat = True
        while reset_dummy_weightmat:
            np.random.seed()
            newseed = np.random.choice(int(self.max_task_num.value))
            if newseed in self.dontidx:
                continue
            np.random.seed(newseed)
            reset_dummy_weightmat = False

            from_random = True
            if from_random:
                while True:
                    genseed = np.random.choice(int(1e10))
                    #Generating random network
                    nettype = np.random.choice(['sf'])
                    if nettype == 'sf':
                        numnode = np.random.randint(20,41)
                        G = nx.generators.directed.scale_free_graph(numnode, alpha=0.2,beta=0.6,gamma=0.2, delta_in=1, delta_out=1, seed=genseed)
                    elif nettype == 'gnp':
                        numnode = np.random.randint(13,31)
                        connectp = np.random.rand()*0.2 + 0.15
                        G = nx.generators.random_graphs.fast_gnp_random_graph(numnode, p=connectp, seed=genseed, directed=True)
                    elif nettype == 'rko':
                        numnode = np.random.randint(15,30)
                        outdeg = np.random.choice(2) + 2    #2~3
                        G = nx.generators.directed.random_k_out_graph(numnode, outdeg, 1, seed=genseed)
                    else:
                        raise KeyError
                    if not nx.is_weakly_connected(G):
                        continue
                    self.dummy_weightmat = np.array(nx.convert_matrix.to_numpy_matrix(G)).T
                    self.dummy_reset_initials(numnode)
                    self.dummy_y_index = np.arange(numnode)
                    '''
                    #use default network
                    self.dummy_weightmat = np.array(self.weightmat.copy())
                    self.reset_initials()
                    self.dummy_y_index = self.y_index
                    numnode = self.dummy_weightmat.shape[0]
                    '''
                    '''
                    # network permutation with preserving indegree of each node
                    self.dummy_weightmat = np.zeros_like(self.weightmat)
                    for row in range(self.Nodes.size):
                        nozero = np.count_nonzero(self.weightmat[row,:])
                        if nozero != 0:
                            self.dummy_weightmat[row,np.random.choice(self.Nodes.size, nozero, replace=False)] = 1
                    '''

                    np.random.seed(newseed)
                    scaleparam = (np.random.rand()*0.75) + 2.0
                    #scaleparam = 3
                    #ww = np.round(np.random.normal(loc=0.0,scale=scaleparam,size=np.count_nonzero(self.weightmat)))
                    ww = np.round(np.random.rand(np.count_nonzero(self.dummy_weightmat))*(self.scale*2)) - self.scale
                    while True:
                        zeroidx = ww==0
                        if np.sum(zeroidx) == 0:
                            break
                        #ww[zeroidx] = np.round(np.random.normal(loc=0.0,scale=scaleparam,size=np.sum(zeroidx)))
                        ww[zeroidx] = np.round(np.random.rand(np.sum(zeroidx))*(self.scale*2)) - self.scale

                    if self.use_randsign:
                        tempsign = np.ones_like(ww)
                        posratio = np.random.rand()*0.6 + 0.2
                        tempsign[np.random.rand(ww.size)>posratio] = -1
                        ww = np.abs(ww) * tempsign
                    else:
                        ww = np.abs(ww)

                    #self.dummy_weightmat[self.dummy_weightmat!=0] = np.abs(np.clip(ww, -scale, scale))
                    self.dummy_weightmat[self.dummy_weightmat!=0] = np.clip(ww, -self.scale, self.scale)
                    self.dummy_sign_mask = np.sign(self.dummy_weightmat)
                    if not self.use_randsign:
                        self.dummy_weightmat *= self.sign_mask
                        self.dummy_sign_mask = np.sign(self.dummy_weightmat)
                    #self.dummy_basal = np.clip(np.round(np.random.normal(loc=0.0,scale=1.3,size=len(self.Nodes))),-scale, scale).astype(np.int32).reshape([-1,1])
                    self.dummy_basal = self.basal.copy()

                    #if np.all(np.abs(np.sum(self.dummy_weightmat, axis=1)) <= 2):
                    #    break
                    break
            else:
                self.dummy_weightmat = np.array(self.weightmat.copy())
                ww = np.abs(self.dummy_weightmat[self.dummy_weightmat!=0])
                ww = ww[np.random.permutation(ww.size)]
                self.dummy_weightmat[self.dummy_weightmat!=0] = ww
                self.dummy_weightmat *= self.sign_mask
                self.dummy_sign_mask = np.sign(self.dummy_weightmat)
                self.dummy_basal = self.basal.copy()

            if self.use_v4:
                yy = np.array(attractor.find_attractor_v4(self.dummy_weightmat, self.initials))
            else:
                yy = np.array(attractor.find_attractor_v3(self.dummy_weightmat, self.dummy_basal, self.initials))
            yy_unique = np.concatenate(yy,axis=0)
            yy_unique, counts = np.unique(yy_unique, axis=0, return_counts=True)
            yy_aver = np.average(yy_unique,axis=0,weights=counts)
            if self.use_v4:
                yy_aver += 1
                yy_aver = yy_aver / 2
        
            np.random.seed()
            for i in range(len(yy)):
                if len(yy[i]) > 1:
                    #yy[i] = yy[i][np.random.choice(len(yy[i]))].reshape([1,-1])
                    yy[i] = yy[i][0].reshape([1,-1])
        
            yy_unique_for_pert = np.concatenate(yy, axis=0)
            yy_unique_for_pert, counts = np.unique(yy_unique_for_pert, axis=0, return_counts=True)

            self.dummy_perturbation = pd.DataFrame(data=np.zeros([numnode, mb_size], dtype=np.int), index=np.arange(numnode))
            self.dummy_y = pd.DataFrame(data=np.zeros([numnode, mb_size]), index=np.arange(numnode))
            i = 0
            trial = 0
            while i < mb_size:
                trial += 1
                if trial > 1000:
                    reset_dummy_weightmat = True
                    self.dontidx.append(newseed)
                    break
                if pert is not None:
                    self.dummy_perturbation = pert
                else:
                    pertnum = np.random.randint(1,int(numnode/2))
        #            self.dummy_perturbation[np.random.choice(np.arange(len(self.Nodes)),size=pertnum,replace=False), i] = np.random.choice([-1,1],size=pertnum,replace=True)#it's general version
                    self.dummy_perturbation.iloc[:,i] = 0.0
                    self.dummy_perturbation.iloc[np.random.choice(np.arange(numnode),size=pertnum,replace=False), i] = 1.0
                pert_nodes = [s for s in list(self.dummy_perturbation.index[(self.dummy_perturbation!=0).iloc[:,i]])]
            
                stim_idx = []
                pert_idx = []
                for pn in pert_nodes:
                    '''
                    if pn[-1] == 'i':
                        pert_idx.append(np.where(self.Nodes==pn[:-1])[0][0])
                    else:
                        stim_idx.append(np.where(self.Nodes==pn)[0][0])
                    '''
                    if np.random.rand() > 0.5:
                        pert_idx.append(pn)
                    else:
                        stim_idx.append(pn)

                stim_idx_np = np.unique(np.array(stim_idx))
                pert_idx_np = np.unique(np.array(pert_idx))
                self.dummy_perturbation.iloc[pert_idx_np.astype(np.int),i] = -1.0
            
                if self.use_v4:
                    yy_p = np.array(attractor.find_attractor_pert_v4(self.dummy_weightmat, yy_unique_for_pert, pert_idx_np, stim_idx_np))
                else:
                    yy_p = np.array(attractor.find_attractor_pert_v3(self.dummy_weightmat, self.dummy_basal, yy_unique_for_pert, pert_idx_np, stim_idx_np))
                el = [x.shape[0] for x in yy_p]
                yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))
                if self.use_v4:
                    yy_p_aver += 1
                    yy_p_aver = yy_p_aver / 2
            
                change = np.log2((yy_p_aver+(1e-3))/(yy_aver+(1e-3)))
                checkidx = np.setdiff1d(np.setdiff1d(self.dummy_y_index, pert_idx_np), stim_idx_np)
                fc = change[checkidx]
                if np.sum(fc!=0) < numnode/2:
                    continue
                self.dummy_y.iloc[:,i] = change[self.dummy_y_index]
                i += 1
        
        #base = self.dummy_y.abs().max().max()
        #dy = self.dummy_y.values / (base * 2) + 0.5
        dy = self.dummy_y.values# + np.random.normal(scale=0.5, size=self.dummy_y.shape)
        self.dummy_y = pd.DataFrame(data=dy, columns=self.dummy_y.columns, index = self.dummy_y.index)
        self.dummy_params_num.value = ww.size
        self.dummy_nodes_num.value = numnode
        self.dummy_params[:ww.size] = np.clip(ww, -self.scale, self.scale)
        stidx = np.where(self.dummy_weightmat)
        stidx = np.concatenate([st.reshape([-1,1]) for st in stidx], axis=1)
        self.dummy_link_idx[:ww.size*2] = stidx.reshape([-1])
        self.taskseed.value = newseed
        self.averact[:numnode] = yy_aver.reshape([-1])


    def return_reward_notmut(self):
        while True:
            self.START_CALC_EVENT.wait()
            self.START_CALC_EVENT.clear()
            if self.do_reset_initials.value == 1:
                self.reset_initials()
                self.do_reset_initials.value = 0
            if self.do_initialize_task.value == 1:
                self.initialize_task(mb_size=int(self.mb_size.value))

                self.params[:int(self.dummy_params_num.value)] = np.random.choice(np.arange(1,self.scale+1), size=int(self.dummy_params_num.value)) * np.sign(self.dummy_params[:int(self.dummy_params_num.value)])

            a_pred = np.array(self.params[:].copy())
            seed = int(self.seed.value)

            current_sm = self.sign_mask
            current_origin = self.weightmat[self.weightmat!=0]
            cur_y_index = self.y_index
            if self.tvt.value == 0:
                np.random.seed(seed)
                mini_batch_size = int(self.mb_size.value)                
                miniidx = np.random.permutation(self.num_sample)[:mini_batch_size]
                data_mini = self.train_y.iloc[:,miniidx].values
                pert_mini = self.perturbation.iloc[:,miniidx]!=0
            elif self.tvt.value == 1:
                data_mini = self.train_y_val.values
                pert_mini = self.perturbation_val!=0
                mini_batch_size = self.valsize
            elif self.tvt.value == 2:
                data_mini = self.train_y_test.values
                pert_mini = self.perturbation_test!=0
                mini_batch_size = self.testsize
            elif self.tvt.value == 3:   # for dummy task
                data_mini = self.dummy_y.values
                pert_mini = self.dummy_perturbation
                mini_batch_size = int(self.mb_size.value)
                current_sm = self.dummy_sign_mask
                current_origin = self.dummy_weightmat[self.dummy_weightmat!=0]
                cur_y_index = self.dummy_y_index
            else:
                print('ERROR!!!!')
                exit(1)

            newpa = np.array(current_sm.copy())
            if self.tvt.value != 3:
                newpa[newpa!=0] = np.round(a_pred[:np.count_nonzero(current_sm)]).astype(np.int)
                #newpa[newpa!=0] = np.abs(np.round(a_pred[:-len(self.Nodes[self.num_input_node:])]).astype(np.int))
            else:
                newpa[newpa!=0] = np.round(a_pred[:int(self.dummy_params_num.value)]).astype(np.int)

            #print('Ture param',current_origin[0:3])
            #current_origin[0:10] = a_pred[0:10]
            #newpa[newpa!=0] = np.abs(np.round(current_origin).astype(np.int))

            if not self.use_randsign:
                newpa = newpa * current_sm
            #newba = np.round(a_pred[-len(self.Nodes[self.num_input_node:]):]).astype(np.int).reshape([-1,1])
            #newba = np.concatenate((np.zeros([self.num_input_node,1],dtype=np.int),newba))
            newba = self.basal.copy()

            if self.use_v4:
                yy = np.array(attractor.find_attractor_v4(newpa, self.initials))
            else:
                yy = np.array(attractor.find_attractor_v3(newpa, newba, self.initials))
    #        yy_aver = np.mean(np.concatenate(yy),axis=0)
            yy_unique = np.concatenate(yy,axis=0)
            yy_unique, counts = np.unique(yy_unique, axis=0, return_counts=True)
            yy_aver = np.average(yy_unique,axis=0,weights=counts)
            if self.use_v4:
                yy_aver += 1
                yy_aver = yy_aver / 2

            for i in range(len(yy)):
                if len(yy[i]) > 1:
                    #yy[i] = yy[i][np.random.choice(len(yy[i]))].reshape([1,-1])
                    yy[i] = yy[i][0].reshape([1,-1])

            yy_unique_for_pert = np.concatenate(yy, axis=0)
            yy_unique_for_pert, counts = np.unique(yy_unique_for_pert, axis=0, return_counts=True)
    
            zero_count = 0
            reward = 0
            spear = 0
            mi = 0.0
            for i in range(mini_batch_size):
                stim_idx, pert_idx = [], []
                if self.tvt.value != 3:
                    pert_nodes = [s for s in list(pert_mini.index[pert_mini.iloc[:,i]])]
                    
                    for pn in pert_nodes:
                        if pn[-1] == 'i':
                            pert_idx.append(np.where(self.Nodes==pn[:-1])[0][0])
                        else:
                            stim_idx.append(np.where(self.Nodes==pn)[0][0])
                else:
                    pert_nodes = [s for s in list(pert_mini.index[pert_mini.iloc[:,i]!=0])]

                    for pn in pert_nodes:
                        if pert_mini.iloc[pn,i] == -1:
                            pert_idx.append(pn)
                        else:
                            stim_idx.append(pn)

                stim_idx_np = np.unique(np.array(stim_idx))
                pert_idx_np = np.unique(np.array(pert_idx))
                
    #            yy_p = np.array(attractor.find_attractor_pert_v2(newpa, newba, self.initials, pert_idx))
                if self.use_v4:
                    yy_p = np.array(attractor.find_attractor_pert_v4(newpa, yy_unique_for_pert, pert_idx_np, stim_idx_np))
                else:
                    yy_p = np.array(attractor.find_attractor_pert_v3(newpa, newba, yy_unique_for_pert, pert_idx_np, stim_idx_np))
                el = [x.shape[0] for x in yy_p]
                yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))
                if self.use_v4:
                    yy_p_aver += 1
                    yy_p_aver = yy_p_aver / 2

                change = np.log2((yy_p_aver+(1e-3))/(yy_aver+(1e-3)))
                zero_count += (np.abs(change) < 0.001).sum()
                if self.y_scaling:
                    if self.use_v4:
                        change = change / (np.log2((2 + (1e-3)) / (1e-3)) * 2) + 0.5
                    else:
                        change = change / (np.log2((1 + (1e-3)) / (1e-3)) * 2) + 0.5
                try:
                    nonanidx = ~np.isnan(data_mini[:,i])
                    if self.tvt.value != 3:
                        nonanidx = nonanidx & ~pert_mini.iloc[self.num_input_node:,i].values
                    else:
                        nonanidx = nonanidx & ~(pert_mini.iloc[:,i].values!=0)
                    cor = np.corrcoef(change[cur_y_index][nonanidx], data_mini[:,i][nonanidx])[0,1]
                    cor_e = 1/(np.mean((change[cur_y_index][nonanidx] - data_mini[:,i][nonanidx]) ** 2)+1)
                    #cor = -np.mean((change[self.y_index][nonanidx] - data_mini[:,i][nonanidx]) ** 2)
                    cors = spearmanr(change[cur_y_index][nonanidx], data_mini[:,i][nonanidx])[0]
                    mi_ = mutual_info_regression(change[cur_y_index][nonanidx].reshape([-1,1]),
                            data_mini[:,i][nonanidx].reshape([-1]))
                    if np.isnan(cor):
                        cor = 0.0
                    if np.isnan(cor_e):
                        cor_e = 0.0
                    if np.isnan(cors):
                        cors = 0.0
                    if np.isnan(mi_):
                        mi_ = 0.0

                    if cors < 0:
                        cor = 0.0  #eue
                        #cor = -100.0 #eucminus
                        #pass
    
                except:
                        cor = 0.0
                        cors = 0.0
                        cor_e = 0.0

                reward += (cor * cor_e)
                spear += cors
                mi += mi_
            reward = reward / mini_batch_size
            spear = spear / mini_batch_size
            mi = mi / mini_batch_size
            zero_count = zero_count / mini_batch_size

            if self.tvt.value == 3:
                predw = np.round(a_pred[:np.count_nonzero(current_sm)]).astype(np.int)
                ham = 1 - distance.hamming(current_origin, predw)
                reward = reward * ham
                spear = spear * ham
                #mi = mi * ham
                mi = ham

            #self.averact[:] = (yy_aver.reshape([-1]) / 2 - 0.5)
            self.zc.put(np.array(mi))
            self.rq.put(np.array(reward))
            self.sq.put(np.array(spear))
        

    def return_reward(self, a_pred, seed, mini_batch_size):
        
        newpa = np.array(self.weightmat.copy())
        newpa[newpa!=0] = np.round(a_pred[:-len(self.Nodes[self.num_input_node:])]).astype(np.int)
        newpa = newpa * self.sign_mask
        newba = np.round(a_pred[-len(self.Nodes[self.num_input_node:]):]).astype(np.int).reshape([-1,1])
        newba = np.concatenate((np.zeros([self.num_input_node,1],dtype=np.int),newba))

#        yy = np.array(attractor.find_attractor_v3(newpa, newba, self.initials))
##        yy_aver = np.mean(np.concatenate(yy),axis=0)
#        yy_unique = np.concatenate(yy,axis=0)
#        yy_unique = np.unique(yy_unique, axis=0)
#        yy_aver = np.mean(yy_unique,axis=0)
        
        np.random.seed(seed)
        miniidx = np.random.permutation(self.num_sample)[:mini_batch_size]
        data_mini = self.train_y.iloc[:,miniidx].values
        pert_mini = self.perturbation.iloc[:,miniidx]!=0
        status_mini = self.status.iloc[:,miniidx]
        
        reward = 0
        spear = 0
        for i in range(mini_batch_size):
            status_pert_nodes = list(status_mini.index[status_mini.iloc[:,i]==-1])
            status_stim_nodes = list(status_mini.index[status_mini.iloc[:,i]==1])
                        
            stim_idx = []
            pert_idx = []
            for pn in status_pert_nodes:
                pert_idx.append(np.where(self.Nodes==pn)[0][0])
            for sn in status_stim_nodes:
                stim_idx.append(np.where(self.Nodes==sn)[0][0])
            stim_idx_np = np.array(stim_idx)
            pert_idx_np = np.array(pert_idx)

            yy = np.array(attractor.find_attractor_pert_v3(newpa, newba, self.initials, pert_idx_np, stim_idx_np))
    #        yy_aver = np.mean(np.concatenate(yy),axis=0)
            yy_unique = np.concatenate(yy,axis=0)
            yy_unique, counts = np.unique(yy_unique, axis=0, return_counts=True)
            yy_aver = np.average(yy_unique,axis=0,weights=counts)
            
            pert_nodes = [s.split(':')[1] for s in list(pert_mini.index[pert_mini.iloc[:,i]])]
            for pn in pert_nodes:
                if pn[-1] == 'i':
                    pert_idx.append(np.where(self.Nodes==pn[:-1])[0][0])
                else:
                    stim_idx.append(np.where(self.Nodes==pn)[0][0])
           
            # Remove activated nodes if they are perturbed
            inters = np.intersect1d(stim_idx, pert_idx)
            if inters.shape[0] > 0:
                for inin in inters:
                    stim_idx.remove(inin)
                    
            stim_idx_np = np.unique(np.array(stim_idx))
            pert_idx_np = np.unique(np.array(pert_idx))
            
#            yy_p = np.array(attractor.find_attractor_pert_v2(newpa, newba, self.initials, pert_idx))
            yy_p = np.array(attractor.find_attractor_pert_v3(newpa, newba, yy_unique, pert_idx_np, stim_idx_np))
            el = [x.shape[0] for x in yy_p]
            yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))

            
            change = np.log2((yy_p_aver+(1e-3))/(yy_aver+(1e-3)))
            try:
                nonanidx = ~np.isnan(data_mini[:,i])
                cor = np.corrcoef(change[self.y_index][nonanidx], data_mini[:,i][nonanidx])[0,1]
                cors = spearmanr(change[self.y_index][nonanidx], data_mini[:,i][nonanidx])[0]
                if np.isnan(cor):
                    cor = 0
                if np.isnan(cors):
                    cors = 0
            except:
                    cor = 0
                    cors = 0
            
            reward += cor
            spear += cors
        reward = reward / mini_batch_size
        spear = spear / mini_batch_size

        return np.array(reward), np.array(spear)
    
