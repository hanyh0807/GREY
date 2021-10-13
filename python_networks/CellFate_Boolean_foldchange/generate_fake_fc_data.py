# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

import attractor

weightmat = [[0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0],
                     [1,1,0,0,-3,0,0,0,0,0,0],
                     [0,0,2,0,0,2,0,-2,0,0,0],
                     [1,1,0,-4,0,0,0,1,0,0,0],
                     [0,0,0,1,0,1,0,0,0,-3,0],
                     [0,0,0,0,0,0,0,0,0,0,-2],
                     [0,0,0,-3,0,0,2,0,0,2,0],
                     [0,0,1,-3,0,0,0,0,0,0,1],
                     [0,0,0,-2,2,0,0,0,0,0,4],
                     [0,0,0,-2,0,0,0,0,1,0,0]]
basal = [[0],[0],[0],[-3],[0],[0],[1],[-3],[0],[-1],[0]]
Nodes = np.array(['TNF', 'FAS', 'RIP1', 'NFkB', 'C8', 'cIAP', 'ATP', 'C3', 'ROS', 'MOMP', 'MPT'])

num_sample = 512
num_input_node = 2

initials = np.array(attractor.generate_initial_state_all(Nodes[num_input_node:],num_sample))
initials = np.concatenate((np.zeros([num_sample, num_input_node], dtype='int32'), initials),axis=1)


yy = np.array(attractor.find_attractor_v3(weightmat, basal, initials))
yy_unique = np.concatenate(yy,axis=0)
yy_unique, counts = np.unique(yy_unique, axis=0, return_counts=True)
yy_aver = np.average(yy_unique,axis=0,weights=counts)

for i in range(len(yy)):
    if len(yy[i]) > 1:
        yy[i] = yy[i][np.random.choice(len(yy[i]))].reshape([1,-1])

yy_unique_for_pert = np.concatenate(yy, axis=0)
yy_unique_for_pert, counts = np.unique(yy_unique_for_pert, axis=0, return_counts=True)

# Single perturbation
pertmat = np.zeros([len(Nodes[num_input_node:]), len(Nodes)], dtype=np.int)
fcmat = np.zeros([len(Nodes[num_input_node:]), len(Nodes[num_input_node:])])
for i in range(num_input_node,len(Nodes)):
    pert_nodes = [Nodes[i]]
    pert_idx = []
    for pn in pert_nodes:
        pert_idx.append(np.where(Nodes==pn)[0][0])
    pert_idx_np = np.unique(np.array(pert_idx))
    pertmat[i-num_input_node, pert_idx_np] = 1
    
    yy_p = np.array(attractor.find_attractor_pert_v3(weightmat, basal, yy_unique_for_pert, pert_idx_np, np.array([])))
    el = [x.shape[0] for x in yy_p]
    yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))
    
    change = np.log2((yy_p_aver+(1e-3))/(yy_aver+(1e-3)))
    fcmat[i-num_input_node,:] = change[num_input_node:]

# Input with single perturbation
pertmat_i = np.zeros([len(Nodes[num_input_node:])*num_input_node, len(Nodes)], dtype=np.int)
fcmat_i = np.zeros([len(Nodes[num_input_node:])*num_input_node, len(Nodes[num_input_node:])])
for inp in range(num_input_node):
    for i in range(num_input_node,len(Nodes)):
        pert_nodes = [Nodes[i]]
        pert_idx = []
        for pn in pert_nodes:
            pert_idx.append(np.where(Nodes==pn)[0][0])
        pert_idx_np = np.unique(np.array(pert_idx))
        pertmat_i[inp*(len(Nodes)-num_input_node) + (i-num_input_node), pert_idx_np] = 1
        pertmat_i[inp*(len(Nodes)-num_input_node) + (i-num_input_node), inp] = 1
        
        yy_p = np.array(attractor.find_attractor_pert_v3(weightmat, basal, yy_unique_for_pert, pert_idx_np, np.array([inp])))
        el = [x.shape[0] for x in yy_p]
        yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))
        
        change = np.log2((yy_p_aver+(1e-3))/(yy_aver+(1e-3)))
        fcmat_i[inp*(len(Nodes)-num_input_node) + (i-num_input_node),:] = change[num_input_node:]

    
# Double perturbation
pertmat_d = np.zeros([int((len(Nodes[num_input_node:])*(len(Nodes[num_input_node:])-1))/2), len(Nodes)], dtype=np.int)
fcmat_d = np.zeros([int((len(Nodes[num_input_node:])*(len(Nodes[num_input_node:])-1))/2), len(Nodes[num_input_node:])])
dcount = 0
for i in range(num_input_node,len(Nodes)):
    for j in range(i+1,len(Nodes)):
        if i==j:
            continue
        pert_nodes = [Nodes[i], Nodes[j]]
        pert_idx = []
        for pn in pert_nodes:
            pert_idx.append(np.where(Nodes==pn)[0][0])
        pert_idx_np = np.unique(np.array(pert_idx))
        pertmat_d[dcount, pert_idx_np] = 1
        
        yy_p = np.array(attractor.find_attractor_pert_v3(weightmat, basal, yy_unique_for_pert, pert_idx_np, np.array([])))
        el = [x.shape[0] for x in yy_p]
        yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))
        
        change = np.log2((yy_p_aver+(1e-3))/(yy_aver+(1e-3)))
        fcmat_d[dcount,:] = change[num_input_node:]
        dcount += 1

# Input with Double perturbation
pertmat_d_i = np.zeros([int((len(Nodes[num_input_node:])*(len(Nodes[num_input_node:])-1))/2)*num_input_node, len(Nodes)], dtype=np.int)
fcmat_d_i = np.zeros([int((len(Nodes[num_input_node:])*(len(Nodes[num_input_node:])-1))/2)*num_input_node, len(Nodes[num_input_node:])])
dcount = 0
for inp in range(num_input_node):
    for i in range(num_input_node,len(Nodes)):
        for j in range(i+1,len(Nodes)):
            if i==j:
                continue
            pert_nodes = [Nodes[i], Nodes[j]]
            pert_idx = []
            for pn in pert_nodes:
                pert_idx.append(np.where(Nodes==pn)[0][0])
            pert_idx_np = np.unique(np.array(pert_idx))
            pertmat_d_i[dcount, pert_idx_np] = 1
            pertmat_d_i[dcount, inp] = 1
            
            yy_p = np.array(attractor.find_attractor_pert_v3(weightmat, basal, yy_unique_for_pert, pert_idx_np, np.array([inp])))
            el = [x.shape[0] for x in yy_p]
            yy_p_aver = np.average(np.concatenate(yy_p),axis=0,weights=np.repeat(counts,el))
            
            change = np.log2((yy_p_aver+(1e-3))/(yy_aver+(1e-3)))
            fcmat_d_i[dcount,:] = change[num_input_node:]
            dcount += 1


pert_out = np.concatenate((pertmat, pertmat_i, pertmat_d, pertmat_d_i), axis=0)
fc_out = np.concatenate((fcmat, fcmat_i, fcmat_d, fcmat_d_i), axis=0)

output = np.concatenate((pert_out, fc_out), axis=1)
Nd = np.array(['fc:'+x for x in Nodes[num_input_node:]])
Np = np.array([x if i < num_input_node else x+'i' for i, x in enumerate(Nodes)])
output = pd.DataFrame(data=output, columns=np.concatenate((Np,Nd)))

output.to_csv('simulated_fc.csv', index=False)
