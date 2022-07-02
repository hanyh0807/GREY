# -*- coding: utf-8 -*-


import numpy as np
import random
import pandas as pd
import copy
from numba import jit
import numba as nb


# =============================================================================
# def datetime_decorator(func):
#     def decorated():
#         print (datetime.datetime.now())
#         func()
#         print (datetime.datetime.now())
#     return decorated()
# 
# =============================================================================




"""input filename output weightmat, genelist"""
# v1 = (x1, +1, x2)

def make_weightmat(filename): #filenmae = txt file
    netdata=[]
    gene=[]
    with open(filename, 'r') as f:
        for line in f.readlines():
            tmp = line.split()
            if tmp[0] not in gene:
                gene.append(tmp[0])
            if tmp[2] not in gene:
                gene.append(tmp[2])
            netdata.append(tuple(tmp))
    #print(netdata[0])
    num_nodes = len(gene)
    weightmat = np.zeros((num_nodes, num_nodes), dtype = np.int32)
    for idx, edge in enumerate(netdata):
        row = gene.index(edge[2]) #target index
        col = gene.index(edge[0]) #source index
        
        if edge[1] == 'inhibition':
            weightmat[row][col] = (-1)
        else:
            weightmat[row][col] = 1
    
    basal = list(np.zeros(len(gene), dtype = np.int32)) #random
    
    return weightmat, gene, basal #np.array, list, list

        
        
# v2 = weighted sum matrix



def make_weightmat_from_file(filename_net, filename_basal, filename_gene, num_mut):
    gene=[]
    with open(filename_gene, 'r') as f:
        for line in f.readlines():
            tmp = line.split()
            if tmp[0] not in gene:
                gene.append(tmp[0])
             
    dat = pd.read_csv(filename_net,sep=',', header = None)
    weightmat = np.array(dat, dtype = np.int32)

    b = pd.read_csv(filename_basal, header = None)
    basal = list(b.iloc[num_mut,:]) #normal

    return weightmat, gene, basal








"""initial_state"""
#random N or all combination 2^n#V2 

def generate_initial_state_all(node_list, initial_num):   
    
    n = len(node_list)
    
    rand_int =[]
    for i in range(initial_num):
        rand_int.append(random.getrandbits(n))
    rand_int = np.unique(rand_int)
    while (rand_int.shape[0] < initial_num):
        temp = []
        current = rand_int.shape[0]
        for i in range(initial_num-current):
            temp.append(random.getrandbits(n))
        rand_int = np.concatenate((rand_int, np.array(temp)))
        rand_int = np.unique(rand_int)    
    
#    rand_int = np.random.choice(2**n, initial_num, replace=False)
    s=[bin(int(x))[2:] for x in list(rand_int)]
    
    initset=['0'*(n-len(x))+x for x in list(s)]   
    
    out = []
    for s in initset:
        out.append([int(i) for i in s])
        
    return out



"""weighted matrix, basal, attractor"""
@jit
def find_attractor(weightmat, basal, initial_state): #np.array, list, np.array
    tot_att = []
    initial_state = np.array(initial_state)
    
    for inis in range(initial_state.shape[0]):
        m = weightmat #별명
        b = np.array(basal) #별명
        trajectory = []
        current_state = copy.copy(initial_state[inis,:]) # copy
        #print(initial_state)
        while 1:
            attractor=[]
            trajectory.append(current_state) #np
            current_state_temp = current_state.reshape([-1,1])
            weighted_sum = (np.dot(m,current_state_temp) + b).reshape([-1])
            
            next_state = copy.copy(current_state) #copy
    
            next_state[weighted_sum>0]=1
            next_state[weighted_sum<0]=0
            # if n == 0 그대로 유지
            
            
            checked_state = copy.copy(next_state) #np
            # is point attractor
            
            for i in range(len(trajectory)):
                if np.linalg.norm(trajectory[i]-checked_state) == 0 :
                    attractor = trajectory[i:] #list, array
                    break
                else:
                    continue
            
            if attractor == []:
                current_state = copy.copy(next_state)
            else:
                break
        tot_att.append(attractor)    
                
    return tot_att


@jit
def find_attractor_pert(weightmat, basal, initial_state, pert_idx): #np.array, list, np.array, np.array
    tot_att = []
    initial_state = np.array(initial_state)
    
    for inis in range(initial_state.shape[0]):
        m = weightmat #별명
        b = np.array(basal) #별명
        trajectory = []
        current_state = copy.copy(initial_state[inis,:]) # copy
        #print(initial_state)
        while 1:
            attractor=[]
            current_state[pert_idx] = 1
            trajectory.append(current_state) #np
            current_state_temp = current_state.reshape([-1,1])
            weighted_sum = (np.dot(m,current_state_temp) + b).reshape([-1])
            
            next_state = copy.copy(current_state) #copy
    
            next_state[weighted_sum>0]=1
            next_state[weighted_sum<0]=0
            next_state[pert_idx] = 1
            # if n == 0 그대로 유지
            
            
            checked_state = copy.copy(next_state) #np
            # is point attractor
            
            for i in range(len(trajectory)):
                if np.linalg.norm(trajectory[i]-checked_state) == 0 :
                    attractor = trajectory[i:] #list, array
                    break
                else:
                    continue
            
            if attractor == []:
                current_state = copy.copy(next_state)
            else:
                break
        tot_att.append(attractor)    
                
    return tot_att

def find_attractor_v2(weightmat, basal, initial_state): #np.array, list, np.array
    num_initial = len(initial_state)
    tot_att = [[]] * num_initial
    initial_state = np.array(initial_state)
    
    surviver = np.arange(num_initial)
    
    current_state = initial_state.copy()
    trajectory = np.expand_dims(initial_state.copy(),axis=0)
    
    weightmat = np.transpose(weightmat)
    basal = np.transpose(basal)
    
    while 1:
        weighted_sum = (np.dot(current_state, weightmat) + basal)
        
        next_state = current_state.copy()
        next_state[weighted_sum>0] = 1
        next_state[weighted_sum<0] = 0
        
        need_modify = False
        remainder = np.ones_like(surviver, dtype=np.bool)        
        for i in range(trajectory.shape[0]):
            check = trajectory[i] - next_state
            checksum = np.sum(np.abs(check),axis=1)
            matched = np.where(checksum == 0)[0]
            for idx in matched:
#                print(idx)
                tot_att[surviver[idx]] = trajectory[i:,idx,:]
                remainder[idx] = False
                need_modify = True
                
        if need_modify:
            next_state = next_state[remainder,:]
            trajectory = trajectory[:,remainder,:]
            surviver = surviver[remainder]
        
        if next_state.shape[0] == 0:
            break
        trajectory = np.concatenate((trajectory, np.expand_dims(next_state,axis=0)))
        
        current_state = next_state
        
    return tot_att
  
def find_attractor_pert_v2(weightmat, basal, initial_state, pert_idx, stim_idx): #np.array, list, np.array
    num_initial = len(initial_state)
    tot_att = [[]] * num_initial
    initial_state = np.array(initial_state)
    
    surviver = np.arange(num_initial)
    
    current_state = initial_state.copy()
    trajectory = np.expand_dims(initial_state.copy(),axis=0)
    
    weightmat = np.transpose(weightmat)
    basal = np.transpose(basal)
    
    while 1:
        if pert_idx.shape[0] > 0:
            current_state[:,pert_idx] = 0
        if stim_idx.shape[0] > 0:
            current_state[:,stim_idx] = 1
        weighted_sum = (np.dot(current_state, weightmat) + basal)

        next_state = current_state.copy()
        next_state[weighted_sum>0] = 1
        next_state[weighted_sum<0] = 0
        if pert_idx.shape[0] > 0:
            next_state[:,pert_idx] = 0
        if stim_idx.shape[0] > 0:
            next_state[:,stim_idx] = 1
        
        need_modify = False
        remainder = np.ones_like(surviver, dtype=np.bool)
        for i in range(trajectory.shape[0]):
            check = trajectory[i] - next_state
            checksum = np.sum(np.abs(check),axis=1)
            matched = np.where(checksum == 0)[0]            
            for idx in matched:
                tot_att[surviver[idx]] = trajectory[i:,idx,:]
                remainder[idx] = False
                need_modify = True
                
        if need_modify:
            next_state = next_state[remainder,:]
            trajectory = trajectory[:,remainder,:]
            surviver = surviver[remainder]
        
        if next_state.shape[0] == 0:
            break
        trajectory = np.concatenate((trajectory, np.expand_dims(next_state,axis=0)))
        
        current_state = next_state
        
    return tot_att


@nb.njit(fastmath=True,parallel=True)
def matmul_basal(C, W, B):
    out = np.empty_like(C, dtype=C.dtype)
       
    for i in nb.prange(out.shape[0]):
        for j in range(out.shape[1]):
            temp = 0
            for k in range(out.shape[1]):
                temp += C[i,k] * W[k,j]
            out[i,j] = temp + B[0,j]
    
    return out

@nb.njit(fastmath=True)
def matmul_basal_(C, W, B):
    out = np.empty_like(C, dtype=C.dtype)
    
#    C = np.ascontiguousarray(C)
#    W = np.ascontiguousarray(W)
#    B = np.ascontiguousarray(B)
    
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            temp = 0
            for k in range(out.shape[1]):
                temp += C[i,k] * W[k,j]
            out[i,j] = temp + B[0,j]
    
    return out

@nb.njit(fastmath=True,parallel=True)
def matmul(C, W):
    out = np.empty_like(C, dtype=C.dtype)
       
    for i in nb.prange(out.shape[0]):
        for j in range(out.shape[1]):
            temp = 0
            for k in range(out.shape[1]):
                temp += C[i,k] * W[k,j]
            out[i,j] = temp
    
    return out

@nb.njit(fastmath=True)
def matmul_(C, W):
    out = np.empty_like(C, dtype=C.dtype)
    
#    C = np.ascontiguousarray(C)
#    W = np.ascontiguousarray(W)
    
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            temp = 0
            for k in range(out.shape[1]):
                temp += C[i,k] * W[k,j]
            out[i,j] = temp
    
    return out


@nb.njit(fastmath=True)
def get_checksum(T, N):
#    check = np.empty_like(T, dtype=T.dtype)
    checksum = np.empty(T.shape[:2], dtype=T.dtype)
       
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            temp = 0
            for k in range(T.shape[2]):
#                check[i,j,k] = abs(T[i,j,k] - N[j,k])
                temp += abs(T[i,j,k] - N[j,k])
            checksum[i,j] = temp
    
    return checksum

def find_attractor_v3(weightmat, basal, initial_state): #np.array, list, np.array
    num_initial = len(initial_state)
    tot_att = [[]] * num_initial
    initial_state = np.array(initial_state)
    
    surviver = np.arange(num_initial)
    
    current_state = initial_state.copy()
    trajectory = np.expand_dims(initial_state.copy(),axis=0)
    weightmat = np.transpose(weightmat)
    basal = np.transpose(basal)
    
    while 1:
        if current_state.shape[0] < 500:
            weighted_sum = matmul_basal_(current_state, weightmat, basal)
        else:
            weighted_sum = matmul_basal(current_state, weightmat, basal)
        
        next_state = current_state.copy()
        next_state[weighted_sum>0] = 1
        next_state[weighted_sum<0] = 0
        
        need_modify = False
        remainder = np.ones_like(surviver, dtype=np.bool)
        checksum = get_checksum(trajectory, next_state)
        matched = np.array(np.where(checksum==0))
        for i in range(matched.shape[1]):            
            tot_att[surviver[matched[1,i]]] = trajectory[matched[0,i]:,matched[1,i],:]
            remainder[matched[1,i]] = False
            need_modify = True
                
        if need_modify:
            next_state = next_state[remainder,:]
            trajectory = trajectory[:,remainder,:]
            surviver = surviver[remainder]
        
        if next_state.shape[0] == 0:
            break
        trajectory = np.concatenate((trajectory, np.expand_dims(next_state,axis=0)))
        
        current_state = next_state
        
    return tot_att
    
def find_attractor_pert_v3(weightmat, basal, initial_state, pert_idx, stim_idx): #np.array, list, np.array
    num_initial = len(initial_state)
    tot_att = [[]] * num_initial
    initial_state = np.array(initial_state)
    
    surviver = np.arange(num_initial)
    
    current_state = initial_state.copy()
    trajectory = np.expand_dims(initial_state.copy(),axis=0)
    
    weightmat = np.transpose(weightmat)
    basal = np.transpose(basal)

    while 1:
        if pert_idx.shape[0] > 0:
            current_state[:,pert_idx] = 0
        if stim_idx.shape[0] > 0:
            current_state[:,stim_idx] = 1
        
        if current_state.shape[0] < 500:
            weighted_sum = matmul_basal_(current_state, weightmat, basal)
        else:
            weighted_sum = matmul_basal(current_state, weightmat, basal)

        next_state = current_state.copy()
        next_state[weighted_sum>0] = 1
        next_state[weighted_sum<0] = 0
        if pert_idx.shape[0] > 0:
            next_state[:,pert_idx] = 0
        if stim_idx.shape[0] > 0:
            next_state[:,stim_idx] = 1
        
        need_modify = False
        remainder = np.ones_like(surviver, dtype=np.bool)
        checksum = get_checksum(trajectory, next_state)
        matched = np.array(np.where(checksum==0))
        for i in range(matched.shape[1]):            
            tot_att[surviver[matched[1,i]]] = trajectory[matched[0,i]:,matched[1,i],:]
            remainder[matched[1,i]] = False
            need_modify = True
                
        if need_modify:
            next_state = next_state[remainder,:]
            trajectory = trajectory[:,remainder,:]
            surviver = surviver[remainder]
        
        if next_state.shape[0] == 0:
            break
        trajectory = np.concatenate((trajectory, np.expand_dims(next_state,axis=0)))
        
        current_state = next_state
        
    return tot_att


def find_attractor_v4(weightmat, initial_state): #np.array, np.array
    num_initial = len(initial_state)
    tot_att = [[]] * num_initial
    initial_state = np.array(initial_state)
    
    surviver = np.arange(num_initial)
    
    current_state = initial_state.copy()
    trajectory = np.expand_dims(initial_state.copy(),axis=0)
    weightmat = np.transpose(weightmat)
    
    while 1:
        if current_state.shape[0] < 500:
            weighted_sum = matmul_(current_state, weightmat)
        else:
            weighted_sum = matmul(current_state, weightmat)
        
        next_state = current_state.copy()
        next_state[weighted_sum>0] = 1
        next_state[weighted_sum<0] = -1
        
        need_modify = False
        remainder = np.ones_like(surviver, dtype=np.bool)
        checksum = get_checksum(trajectory, next_state)
        matched = np.array(np.where(checksum==0))
        for i in range(matched.shape[1]):            
            tot_att[surviver[matched[1,i]]] = trajectory[matched[0,i]:,matched[1,i],:].copy()
            remainder[matched[1,i]] = False
            need_modify = True
                
        if need_modify:
            next_state = next_state[remainder,:]
            trajectory = trajectory[:,remainder,:]
            surviver = surviver[remainder]
        
        if next_state.shape[0] == 0:
            break
        trajectory = np.concatenate((trajectory, np.expand_dims(next_state,axis=0)))
        
        current_state = next_state
        
    return tot_att
    
def find_attractor_pert_v4(weightmat, initial_state, pert_idx, stim_idx): #np.array, np.array
    num_initial = len(initial_state)
    tot_att = [[]] * num_initial
    initial_state = np.array(initial_state)
    
    surviver = np.arange(num_initial)
    
    current_state = initial_state.copy()
    trajectory = np.expand_dims(initial_state.copy(),axis=0)
    
    weightmat = np.transpose(weightmat)
    
    while 1:
        if pert_idx.shape[0] > 0:
            current_state[:,pert_idx] = -1
        if stim_idx.shape[0] > 0:
            current_state[:,stim_idx] = 1
        
        if current_state.shape[0] < 500:
            weighted_sum = matmul_(current_state, weightmat)
        else:
            weighted_sum = matmul(current_state, weightmat)

        next_state = current_state.copy()
        next_state[weighted_sum>0] = 1
        next_state[weighted_sum<0] = -1
        if pert_idx.shape[0] > 0:
            next_state[:,pert_idx] = -1
        if stim_idx.shape[0] > 0:
            next_state[:,stim_idx] = 1
        
        need_modify = False
        remainder = np.ones_like(surviver, dtype=np.bool)
        checksum = get_checksum(trajectory, next_state)
        matched = np.array(np.where(checksum==0))
        for i in range(matched.shape[1]):            
            tot_att[surviver[matched[1,i]]] = trajectory[matched[0,i]:,matched[1,i],:].copy()
            remainder[matched[1,i]] = False
            need_modify = True
                
        if need_modify:
            next_state = next_state[remainder,:]
            trajectory = trajectory[:,remainder,:]
            surviver = surviver[remainder]
        
        if next_state.shape[0] == 0:
            break
        trajectory = np.concatenate((trajectory, np.expand_dims(next_state,axis=0)))
        
        current_state = next_state
        
    return tot_att


