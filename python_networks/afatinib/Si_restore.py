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
    def __init__(self, num_initial, drug = 'Afatinib', forexp=False, start_process = True, mb_size = 8, y_scaling=False, use_randsign=False, max_task_num=10000, name=None, scale=5):
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
        self.calc_type = 'original' #original, original_r, stable_point
        
        self.adjmat = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/afatinib_network.csv', header=0,index_col=0)
        self.weightmat = self.adjmat.values
        input_link_num = np.abs(self.weightmat).sum(axis=1)
        input_link_num_order = np.argsort(input_link_num)
        self.weightmat = self.weightmat[input_link_num_order]
        self.weightmat = self.weightmat[:,input_link_num_order]
        self.basal = 0
        self.Nodes = self.adjmat.columns.ravel().astype(np.str)
        self.Nodes = self.Nodes[input_link_num_order]
        self.Nodes_string = np.array2string(self.Nodes, max_line_width=np.inf, separator=',')
        self.Nodes_string = re.sub('[\[\] \']','', self.Nodes_string)

        wmidx = np.where(self.weightmat)
        wmidx = np.concatenate([st.reshape([-1,1]) for st in wmidx], axis=1)
        self.weightmat_link_idx = Array('d', 2000) #maximum 500 indices
        self.weightmat_link_idx[:np.count_nonzero(self.weightmat)*2] = wmidx.reshape([-1])


        self.sign_mask = self.weightmat.copy()
        self.sign_mask = np.sign(self.sign_mask)

        self.use_v4 = True
        self.outmut_initial = False
        
        if forexp:
            raise NotImplementedError
            #pert_data = np.transpose(pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/binaryexpression.csv', header=0, index_col=0))
        else:
            pert_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/Core_Profile_MM.csv', header=0, index_col=0)
            remove = ~pd.isna(pert_data.loc['AUC'])
            pert_data = pert_data.loc[:,remove]
            pdcols = list(pert_data.columns)
            pdcols = [str(x) for x in pdcols]
            pert_data.columns = pdcols
            '''
            exp_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/trametinib/trame_geneexp.csv', header=0, index_col=0)
            exp_data = exp_data.loc[:,remove.values]
            exp_data.columns = pdcols
            '''

        status_base_temp = pert_data.iloc[1:,:]
        status_base = pd.DataFrame(data=np.zeros((self.Nodes.size,status_base_temp.shape[1])), index=self.Nodes, columns=status_base_temp.columns)
        status_base.loc[status_base_temp.index] = status_base_temp
        
        self.num_input_node = np.sum(input_link_num==0)
        if forexp:
            self.perturbation_whole = status_base
            self.train_y_whole = pert_data.iloc[63:,:]
            self.status_whole = status_base
            
        elif self.drug == 'ALL':
#            self.sign_mask = self.weightmat.copy()
#            celllist = ['CCK81','COLO320HSR','HCT116','HT29','HT115','SKCO1','SNUC5','SW620','SW837','SW1116','SW1463']
#            pert_house = []
#            status_house = []
#            y_house = []
#            for cn in celllist:
#                pert_data = np.transpose(pd.read_excel(os.path.dirname(os.path.abspath(__file__))+'/Perturbations and Networks.xlsx',
#                                            sheet_name=cn+'_Pert', header=0, index_col=None))        
#                perturbation = pert_data.iloc[:12,:]
#                train_y = pert_data.iloc[12:,:]
#
#                if self.y_scaling:
#                    base = train_y.abs().max(axis=1).values
#                    tra = train_y.values / (base.reshape([-1,1]) * 2) + 0.5
#                    train_y = pd.DataFrame(data=tra, columns=train_y.columns, index=train_y.index)
#                
#                probadjmat = pd.read_excel(os.path.dirname(os.path.abspath(__file__))+'/Perturbations and Networks.xlsx',
#                                        sheet_name=cn+'_Net', header=0, index_col=0)
#                status = probadjmat['status']
#                status = pd.DataFrame(np.tile(status,(perturbation.shape[1],1)).transpose(),index=status.index)
#                
#                pert_house.append(perturbation)
#                status_house.append(status)
#                y_house.append(train_y)
#            
#            self.perturbation_whole = pd.concat(pert_house, axis=1, ignore_index=True)
#            self.status_whole = pd.concat(status_house, axis=1, ignore_index=True)
#            self.train_y_whole = pd.concat(y_house, axis=1, ignore_index=True)
            pass

        else:
            drugtarget = np.array([['EGFR', 'ERBB2']])
            #drugtarget = np.array([['CDK4', 'CDK6']])
            #drugtarget = np.array([['MAP2K1', 'MAP2K2']])
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
    
        if self.drug != 'ALL':
            self.valsize = int(self.perturbation_whole.shape[1] * 0.2)
            self.testsize = int(self.perturbation_whole.shape[1] * 0.2)
            np.random.seed(0)
            tvsplit = np.random.permutation(self.perturbation_whole.shape[1])
        else:
            self.valsize = 42
            self.testsize = 21
            tvsplit = np.arange(self.perturbation_whole.shape[1])

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

        '''
        expscaler = MinMaxScaler((0,1))
        expscaler.fit(self.train_y_exp.T)
        self.train_y_exp.iloc[:] = expscaler.transform(self.train_y_exp.T).T
        self.train_y_exp_val.iloc[:] = expscaler.transform(self.train_y_exp_val.T).T
        self.train_y_exp_test.iloc[:] = expscaler.transform(self.train_y_exp_test.T).T
        '''

        '''
        exptemp = self.train_y_exp.transpose()
        meme = exptemp.mean(); stst = exptemp.std()

        self.train_y_exp = ((exptemp - meme) / stst).transpose()

        exptemp = self.train_y_exp_val.transpose()
        self.train_y_exp_val = ((exptemp - meme) / stst).transpose()
        exptemp = self.train_y_exp_test.transpose()
        self.train_y_exp_test = ((exptemp - meme) / stst).transpose()
        '''


        if forexp:
            y_index_name = [x.split('_')[0] for x in self.train_y.index]
        else:
#            y_index_name = ['TCF7L2']
            y_index_name = ['Proliferation'] #  positively effects on viability 
            #y_index_name_neg = ['Apoptosis'] #  negatively effects on viability 
            #y_exp_index_name = list(self.train_y_exp.index)
        self.y_index = [] # index of measured nodes at weightmat
        for i in y_index_name:
            self.y_index.append(np.where(i == self.Nodes)[0][0])
        '''
        self.y_exp_index = []
        for i in y_exp_index_name:
            self.y_exp_index.append(np.where(i == self.Nodes)[0][0])
        '''
        if not forexp:
            self.y_index_neg = [] # index of measured nodes at weightmat
            #for i in y_index_name_neg:
            #    self.y_index_neg.append(np.where(i == self.Nodes)[0][0])

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

    def initialize_task(self, mb_size=16, maxnode=31, scale=5, pert=None):
        reset_dummy_weightmat = True
        while reset_dummy_weightmat:
            np.random.seed()
            newseed = np.random.choice(int(self.max_task_num.value))
            if newseed in self.dontidx:
                continue
            np.random.seed(newseed)
            reset_dummy_weightmat = False

            while True:
                genseed = np.random.choice(int(1e10))
                #Generating random network
                nettype = np.random.choice(['sf'])
                if nettype == 'sf':
                    numnode = np.random.randint(20,maxnode)
                    G = nx.generators.directed.scale_free_graph(numnode, alpha=0.2,beta=0.6,gamma=0.2, delta_in=1, delta_out=1, seed=genseed)
                elif nettype == 'gnp':
                    numnode = np.random.randint(13,31)
                    connectp = np.random.rand()*0.2 + 0.15
                    G = nx.generators.random_graphs.fast_gnp_random_graph(numnode, p=connectp, seed=genseed, directed=True)
                elif nettype == 'rko':
                    numnode = np.random.randint(20,maxnode)
                    outdeg = np.random.choice(3) + 2    #2~4
                    G = nx.generators.directed.random_k_out_graph(numnode, outdeg, 1, seed=genseed)
                else:
                    raise KeyError
                if not nx.is_weakly_connected(G):
                    continue
                self.dummy_weightmat = np.array(nx.convert_matrix.to_numpy_matrix(G)).T
                self.dummy_reset_initials(numnode)
                indegree = (self.dummy_weightmat!=0).sum(axis=1)
                np.random.seed(newseed)
                #np.random.seed()
                self.dummy_y_index = [np.random.choice(np.where((indegree != 0))[0])]
                #self.dummy_y_exp_index = list(np.setdiff1d(np.arange(numnode), self.dummy_y_index))
                self.dummy_y_exp_index = np.arange(numnode)
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
                self.dummy_basal = 0

                #if np.all(np.abs(np.sum(self.dummy_weightmat, axis=1)) <= 2):
                #    break
                break


            self.dummy_status = pd.DataFrame(data=np.zeros([numnode, mb_size], dtype=np.int), index=np.arange(numnode))
            self.dummy_perturbation = pd.DataFrame(data=np.zeros([numnode, mb_size], dtype=np.int), index=np.arange(numnode))
            np.random.seed()
            '''
            drug_pertnum = np.random.randint(1,6)
            self.dummy_perturbation.iloc[:,:] = 0.0
            self.dummy_perturbation.iloc[np.random.choice(np.arange(numnode),size=drug_pertnum,replace=False), :] = 1.0
            #self.dummy_perturbation.iloc[self.dummy_y_index,:] = 0.0
            '''
            self.dummy_exp = pd.DataFrame(data=np.zeros([len(self.dummy_y_exp_index), mb_size]), index=self.dummy_y_exp_index)
            self.dummy_fc = pd.DataFrame(data=np.zeros([len(self.dummy_y_exp_index), mb_size]), index=self.dummy_y_exp_index)
            #self.dummy_y = pd.DataFrame(data=np.zeros([len(self.Nodes[self.num_input_node:]), mb_size]), index=self.train_y_whole.index)
            self.dummy_y = pd.DataFrame(data=np.zeros([len(self.dummy_y_index), mb_size]), index=self.dummy_y_index)
            i = 0
            trial = 0
            while i < mb_size:
                #np.random.seed()
                trial += 1
                if trial % 1000 == 0:
                    '''
                    i = 0
                    drug_pertnum = np.random.randint(1,6)
                    self.dummy_perturbation.iloc[:,:] = 0.0
                    self.dummy_perturbation.iloc[np.random.choice(np.arange(numnode),size=drug_pertnum,replace=False), :] = 1.0
                    #self.dummy_perturbation.iloc[self.dummy_y_index,:] = 0.0
                    '''
                    if trial > 100:
                        reset_dummy_weightmat = True
                        self.dontidx.append(newseed)
                        break
                if pert is not None:
                    self.dummy_status = pert
                else:
                    self.dummy_status.iloc[:,i] = 0.0
                    #initial_pertnum = np.random.randint(5,int(numnode*0.9)) #outside mutation
                    initial_pertnum = np.random.randint(np.max([int(numnode*0.25),3]),np.max([int(numnode*0.8),2])) #outside mutation
                    self.dummy_status.iloc[np.random.choice(np.arange(numnode),size=initial_pertnum,replace=False), i] = np.random.choice([-2,2], initial_pertnum)

                    #pertnum = np.random.randint(1,int(numnode/2))
                    pertnum = np.random.randint(1,np.max([int(numnode*0.5),2]))
        #            self.dummy_perturbation[np.random.choice(np.arange(len(self.Nodes)),size=pertnum,replace=False), i] = np.random.choice([-1,1],size=pertnum,replace=True)#it's general version
                    self.dummy_status.iloc[np.random.choice(np.arange(numnode),size=pertnum,replace=False), i] = np.random.choice([-1,1], pertnum)
                    #self.dummy_status.iloc[self.dummy_y_index,i] = 0.0

                    drug_pertnum = np.random.randint(1,10)
                    self.dummy_perturbation.iloc[:,i] = 0.0
                    self.dummy_perturbation.iloc[np.random.choice(np.arange(numnode),size=drug_pertnum,replace=False), i] = 1.0
                    #self.dummy_perturbation.iloc[self.dummy_y_index,i] = 0.0

                pert_idx = list(self.dummy_status.index[self.dummy_status.iloc[:,i]==-1])
                stim_idx = list(self.dummy_status.index[self.dummy_status.iloc[:,i]==1])

                stim_idx_np = np.unique(np.array(stim_idx))
                pert_idx_np = np.unique(np.array(pert_idx))

                initial_pert_idx = list(self.dummy_status.index[self.dummy_status.iloc[:,i]==-2])
                initial_stim_idx = list(self.dummy_status.index[self.dummy_status.iloc[:,i]==2])

                initial_stim_idx_np = np.array(initial_stim_idx)
                initial_pert_idx_np = np.array(initial_pert_idx)

                cur_initials = self.initials.copy()
                if self.outmut_initial:
                    if initial_stim_idx_np.size > 0:
                        cur_initials[:,initial_stim_idx_np] = 1
                    if initial_pert_idx_np.size > 0:
                        cur_initials[:,initial_pert_idx_np] = -1
                    cur_initials = np.unique(cur_initials, axis=0)

                if self.use_v4:
                    yy, _ = np.array(attractor.find_attractor_pert_v4(self.dummy_weightmat, cur_initials, pert_idx_np, stim_idx_np, self.outmut_initial, initial_pert_idx_np, initial_stim_idx_np), dtype=np.object)
                else:
                    yy = np.array(attractor.find_attractor_pert_v3(self.dummy_weightmat, self.dummy_basal, self.initials, pert_idx_np, stim_idx_np))
                yy_unique = np.concatenate(yy,axis=0)
                yy_unique, counts = np.unique(yy_unique, axis=0, return_counts=True)
                yy_aver = np.average(yy_unique,axis=0,weights=counts)
                if self.use_v4:
                    yy_aver += 1
                    yy_aver = yy_aver / 2

                for iy in range(len(yy)):
                    if len(yy[iy]) > 1:
                        #yy[i] = yy[i][np.random.choice(len(yy[i]))].reshape([1,-1])
                        yy[iy] = yy[iy][0].reshape([1,-1])

                yy_unique_for_pert = np.concatenate(yy, axis=0)
                yy_unique_for_pert, counts = np.unique(yy_unique_for_pert, axis=0, return_counts=True)

                pert_nodes = [s for s in list(self.dummy_perturbation.index[self.dummy_perturbation.iloc[:,i]==1])]
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

                if self.use_v4:
                    yy_p, _ = np.array(attractor.find_attractor_pert_v4(self.dummy_weightmat, yy_unique_for_pert, pert_idx_np, stim_idx_np, self.outmut_initial, initial_pert_idx_np, initial_stim_idx_np), dtype=np.object)
                else:
                    yy_p = np.array(attractor.find_attractor_pert_v3(self.dummy_weightmat, self.dummy_basal, yy_unique_for_pert, pert_idx_np, stim_idx_np))
                el = [x.shape[0] for x in yy_p]
                yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))
                if self.use_v4:
                    yy_p_aver += 1
                    yy_p_aver = yy_p_aver / 2

                viability = np.mean(yy_p_aver[self.dummy_y_index]) - np.mean(yy_aver[self.dummy_y_index])

                change = np.log2((yy_p_aver+self.eps)/(yy_aver+self.eps))
                checkidx = np.setdiff1d(self.dummy_y_exp_index, np.concatenate((pert_idx_np, stim_idx_np)))
                fc = change[checkidx]
                if  np.all(np.abs(fc)<(1e-5)):
                    continue
                self.dummy_y.iloc[:,i] = viability
                #self.dummy_y.iloc[:,i] = change[self.y_index]
                self.dummy_exp.iloc[:,i] = yy_aver[self.dummy_y_exp_index]
                #self.dummy_fc.iloc[:,i] = yy_p_aver[self.dummy_y_exp_index] - yy_aver[self.dummy_y_exp_index]
                self.dummy_fc.iloc[:,i] = np.log2((yy_p_aver[self.dummy_y_exp_index]+self.eps)/(yy_aver[self.dummy_y_exp_index]+self.eps))
                i += 1

            '''
            if (np.max(self.dummy_y.values) - np.min(self.dummy_y.values)) < 0.4:
                reset_dummy_weightmat = True
            '''

        #base = self.dummy_y.abs().max().max()
        #dy = self.dummy_y.values / (base * 2) + 0.5
        dy = self.dummy_y.values# + np.random.normal(scale=0.1, size=self.dummy_y.shape)
        #self.dummy_exp = self.dummy_exp + np.random.normal(scale=0.1, size=self.dummy_exp.shape)
        #self.dummy_exp = self.dummy_exp.T.apply(zscore).T
        self.dummy_y = pd.DataFrame(data=dy, columns=self.dummy_y.columns, index = self.dummy_y.index)
        self.dummy_pheno_idx[:len(self.dummy_y_index)] = self.dummy_y_index
        pi = list(self.dummy_perturbation.index[self.dummy_perturbation.iloc[:,0]!=0])
        self.dummy_pert_num.value = len(pi)
        self.dummy_pert_idx[:len(pi)] = pi
        self.dummy_params_num.value = ww.size
        self.dummy_nodes_num.value = numnode
        self.dummy_params[:ww.size] = np.clip(ww, -self.scale, self.scale)
        stidx = np.where(self.dummy_weightmat)
        stidx = np.concatenate([st.reshape([-1,1]) for st in stidx], axis=1)
        self.dummy_link_idx[:ww.size*2] = stidx.reshape([-1])
        self.taskseed.value = newseed
        self.averact[:numnode] = yy_aver.reshape([-1])

    def return_reward_2att(self):
        pool = Pool(8)
        while True:
            self.START_CALC_EVENT.wait()
            self.START_CALC_EVENT.clear()
            self.reset_initials()
            if self.do_reset_initials.value == 1:
                self.reset_initials()
                self.do_reset_initials.value = 0
            if self.do_initialize_task.value == 1:
                self.initialize_task(mb_size=int(self.mb_size.value), maxnode=int(self.maxnode.value))
                #self.params[:int(self.dummy_params_num.value)] = np.sign(self.dummy_params[:int(self.dummy_params_num.value)]) * 2
            a_pred = np.array(self.params[:].copy())
            seed = int(self.seed.value)
            
            current_sm = self.sign_mask
            current_origin = self.weightmat[self.weightmat!=0]
            cur_y_index = self.y_index
            #cur_y_exp_index = self.y_exp_index
            if self.tvt.value == 0:
                np.random.seed(seed)
                mini_batch_size = int(self.mb_size.value)
                #mini_batch_size = 300
                miniidx = np.random.permutation(self.num_sample)[:mini_batch_size]
                data_mini = self.train_y.iloc[:,miniidx].values
                #data_exp_mini = self.train_y_exp.iloc[:,miniidx].values
                pert_mini = self.perturbation.iloc[:,miniidx]!=0
                status_mini = self.status.iloc[:,miniidx]
            elif self.tvt.value == 1:
                data_mini = self.train_y_val.values
                #data_exp_mini = self.train_y_exp_val.values
                pert_mini = self.perturbation_val!=0
                status_mini = self.status_val
                mini_batch_size = self.valsize
            elif self.tvt.value == 2:
                data_mini = self.train_y_test.values
                #data_exp_mini = self.train_y_exp_test.values
                pert_mini = self.perturbation_test!=0
                status_mini = self.status_test
                mini_batch_size = self.testsize
            elif self.tvt.value == 3:   # for dummy task
                data_mini = self.dummy_y.values
                data_exp_mini = self.dummy_exp.values
                data_fc_mini = self.dummy_fc.values
                pert_mini = self.dummy_perturbation!=0
                status_mini = self.dummy_status
                mini_batch_size = int(self.mb_size.value)
                current_sm = self.dummy_sign_mask
                current_origin = self.dummy_weightmat[self.dummy_weightmat!=0]
                cur_y_index = self.dummy_y_index
                cur_y_exp_index = self.dummy_y_exp_index
            else:
                print('ERROR!!!!')
                exit(1)

            observation = []
            prediction = []
            first_att_prol_act = []
            if self.calc_type == 'original':
                '''
                if mini_batch_size > 100:
                    pool = Pool(2)
                else:
                    pool = Pool(2)
                '''
                newpa = np.array(current_sm.copy())
                if self.tvt.value != 3:
                    newpa[newpa!=0] = np.round(a_pred[:np.count_nonzero(current_sm)]).astype(np.int)
                    #newpa[newpa!=0] = np.abs(np.round(a_pred[:-len(self.Nodes[self.num_input_node:])]).astype(np.int))
                else:
                    newpa[newpa!=0] = np.round(a_pred[:int(self.dummy_params_num.value)]).astype(np.int)
                    #newpa = self.dummy_weightmat

                if not self.use_randsign:
                    newpa = newpa * current_sm
                newba = np.zeros(self.Nodes.size)

                '''
                ooinit = np.stack((np.ones(self.initials.shape[1]), np.ones(self.initials.shape[1])*-1))
                ooatt = np.array(attractor.find_attractor_v4(newpa, ooinit), dtype=np.object)
                self.averact_one[:self.initials.shape[1]] = np.mean(ooatt[0].reshape([-1, self.initials.shape[1]]), axis=0)
                self.averact_mone[:self.initials.shape[1]] = np.mean(ooatt[1].reshape([-1, self.initials.shape[1]]), axis=0)
                '''
        
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
                for sam in range(mini_batch_size):
                    mutidx = list(status_mini.index[(status_mini.iloc[:,sam] == 1) | (status_mini.iloc[:,sam] == -1)])
                    '''
                    if tvt == 3:
                        data_exp_idx = np.arange(data_exp_mini.shape[0])[~np.in1d(self.dummy_exp.index, mutidx)]
                    else:
                        data_exp_idx = np.arange(data_exp_mini.shape[0])[~np.in1d(self.train_y_exp.index, mutidx)]
                        mutidx = np.where(np.in1d(self.Nodes, mutidx))[0]
                    exp_compare_idx = cur_y_exp_index[~np.in1d(cur_y_exp_index,mutidx)]

                    ec = np.corrcoef(data_exp_mini[data_exp_idx,sam], yy_aver[sam,exp_compare_idx])[0,1]
                    expcorr.append(ec)
                    es = spearmanr(data_exp_mini[data_exp_idx,sam], yy_aver[sam,exp_compare_idx])[0]
                    expspear.append(es)
                    em = mutual_info_regression(data_exp_mini[data_exp_idx,sam].reshape([-1,1]), yy_aver[sam,exp_compare_idx].reshape([-1]))[0]
                    expmi.append(em)
                    ee = 1/(np.sum((data_exp_mini[data_exp_idx,sam] - yy_aver[sam,exp_compare_idx])**2)+1)
                    exp_e.append(ee)
                    '''

                    if tvt == 3:
                        mutpertidx = np.unique(np.concatenate([mutidx, pert_mini.index[pert_mini.iloc[:,sam]]]))
                        data_fc_idx = np.arange(data_fc_mini.shape[0])[~np.in1d(self.dummy_fc.index, mutpertidx)]
                        fc_compare_idx = cur_y_exp_index[~np.in1d(cur_y_exp_index,mutpertidx)]

                        #dumfc = yy_p_aver[sam,exp_compare_idx] - yy_aver[sam,exp_compare_idx]
                        dumfc = np.log2((yy_p_aver[sam,fc_compare_idx]+self.eps)/(yy_aver[sam,fc_compare_idx]+self.eps))
                        cc = np.corrcoef(data_fc_mini[data_fc_idx,sam], dumfc)[0,1]
                        ss = spearmanr(data_fc_mini[data_fc_idx,sam], dumfc)[0]
                        fee = 1/(np.mean((data_fc_mini[data_fc_idx,sam] - dumfc)**2)+1)
                        try:
                            mm = mutual_info_regression(data_fc_mini[data_fc_idx,sam].reshape([-1,1]), dumfc.reshape([-1]))
                        except:
                            mm = 0.0

                        if np.isnan(cc):
                            cc = 0.0
                        if np.isnan(ss):
                            ss = 0.0
                        if np.isnan(fee):
                            fee = 0.0
                        if np.isnan(mm):
                            mm = 0.0

                        if ss < 0:
                            cc = 0.0

                        fcc.append(cc * fee)
                        fcs.append(ss)
                        fcm.append(mm)

                '''
                expcorr = np.array(expcorr); expspear = np.array(expspear)
                expmi = np.array(expmi); exp_e = np.array(exp_e)
                expcorr[np.isnan(expcorr)] = 0.0
                expspear[np.isnan(expspear)] = 0.0
                expmi[np.isnan(expmi)] = 0.0
                exp_e[np.isnan(exp_e)] = 0.0
                '''
                if self.tvt.value != 3:
                    exp_e = 1
                    fcc = 1
                    fcs = 1
                    fcm = 1

                #pool.close()
                #pool.join()


            observation = np.array(observation).reshape([-1])
            prediction = np.array(prediction).reshape([-1])
            cor = np.corrcoef(observation, prediction)[0,1]
            cors = spearmanr(observation, prediction)[0]
            mi = mutual_info_regression(observation.reshape([-1,1]), prediction.reshape([-1]))
            if self.tvt.value == 3:
                ee = 1/(np.sum((observation - prediction) ** 2) + 1)
            else:
                ee = 1
            if np.isnan(cor):
                cor = 0.0
            if np.isnan(cors):
                cors = 0.0
            if np.isnan(mi):
                mi = 0.0
               
            #non_zero = np.mean(prediction != 0)
            '''
            expfit = np.mean(expcorr*exp_e)
            expspear_m = np.mean(expspear)
            expmi_m = np.mean(expmi)
            '''
            fcc_m = np.mean(fcc)
            fcs_m = np.mean(fcs)
            fcm_m = np.mean(fcm)
            #sparsity = -np.mean(np.abs(np.round(a_pred)))
            #AUC prediction accuracy + high proliferation at first attractor
            #reward = cor + 0.1 * np.mean(first_att_prol_act) + 0.1 * expfit# + 0.1 * non_zero + 0.1 * sparsity + 0.1 * total_change
            #reward = ((cor + cors) / 2) * ee + 0.1 * expfit# + 0.1 * non_zero + 0.1 * sparsity + 0.1 * total_change
            if cors < 0:
                cor = 0.0
            '''
            if cor > 0.9:
                #if expfit < 0.5:
                if fcc_m < 0.5:
                    cor = 0.0
                    cors = 0.0
                    mi = 0.0
            '''
            #reward = ((cor + cors) / 2) * ee * expfit# + 0.1 * non_zero + 0.1 * sparsity + 0.1 * total_change
            #reward = 0.5 * (cor * ee) + 0.5 * expfit
            if self.tvt.value == 3:
                reward = fcc_m
                spear = fcs_m
                mi = fcm_m
            else:
                reward = cor
                pea = cor
                spear = cors
            #reward = ((cor + cors) / 2) * ee

            self.rq.put(np.array(reward))
            #self.pq.put(np.array(pea))
            self.sq.put(np.array(spear))
            self.zc.put(np.array(mi))
            '''
            self.eq.put(np.array(expfit))
            self.esq.put(np.array(expspear_m))
            self.emq.put(np.array(expmi_m))
            '''

            #gc.collect()


    def return_reward_exp(self):
        while True:
            self.START_CALC_EVENT.wait()
            self.START_CALC_EVENT.clear()
            if self.do_reset_initials.value == 1:
                self.reset_initials()
                self.do_reset_initials.value = 0
            a_pred = np.array(self.params[:].copy())
            seed = int(self.seed.value)
            
            newpa = np.array(self.weightmat.copy())
            newpa[newpa!=0] = np.round(a_pred[:-len(self.Nodes[self.num_input_node:])]).astype(np.int)
            newpa = newpa * self.sign_mask
            newba = np.round(a_pred[-len(self.Nodes[self.num_input_node:]):]).astype(np.int).reshape([-1,1])
            newba = np.concatenate((np.zeros([self.num_input_node,1],dtype=np.int),newba))
                
            if self.tvt.value == 0:
                np.random.seed(seed)
                mini_batch_size = int(self.mb_size.value)                
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
    
            reward = 0
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
               
                # Remove activated nodes if they are perturbed
                inters = np.intersect1d(stim_idx, pert_idx)
                if inters.shape[0] > 0:
                    for inin in inters:
                        stim_idx.remove(inin)
                        
                stim_idx_np = np.unique(np.array(stim_idx))
                pert_idx_np = np.unique(np.array(pert_idx))
                
    #            yy_p = np.array(attractor.find_attractor_pert_v2(newpa, newba, self.initials, pert_idx))
                #yy_p = np.array(attractor.find_attractor_pert_v3(newpa, newba, self.initials, pert_idx_np, stim_idx_np))
                yy_p = np.array(attractor.find_attractor_pert_v4(newpa, self.initials, pert_idx_np, stim_idx_np))
                yy_p_aver = np.average(np.concatenate(yy_p),axis=0)
                
                try:
#                    cor = np.corrcoef(data_mini[:,i], yy_p_aver[self.y_index])[0,1]
                    cor = 1 - hamming(data_mini[:,i], yy_p_aver[self.y_index]>0.5)
                    if np.isnan(cor):
                        cor = 0.0
                except:
                    cor = 0.0
                    
                reward += cor

            reward = reward / mini_batch_size
    
            self.rq.put(np.array(reward))
            self.sq.put(np.array(0.0))
            
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
    
    #change = np.log2((yy_p_aver+self.eps)/(yy_aver+self.eps))
    #viability = ((np.mean(yy_p_aver[self.y_index]) - np.mean(yy_p_aver[self.y_index_neg])) + 2) / 4
    #viability = change[self.y_index] + np.mean(yy_p_aver[self.y_index])

    #scale 0~1
    yy_aver = yy_aver / 2
    yy_p_aver = yy_p_aver / 2
    return data_mini[:,i], yy_aver, yy_p_aver
