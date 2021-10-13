# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:05:35 2020

@author: Younghyun Han
"""
import os
import sys
import warnings
#os.environ["OPENBLAS_NUM_THREADS"] = "10"

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import threading
from multiprocessing import Lock, Array, Value, Barrier
import multiprocessing
import tensorflow as tf
import tensorflow_probability as tfp
from graph_nets import blocks, graphs, modules, utils_np, utils_tf
import sonnet as snt
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from time import sleep, time
from collections import deque
import copy
import pickle
import networkx as nx

import python_networks.CellFate_Boolean_foldchange.Simul_tvt_split_mp_netgen as test_prob
import python_networks.afatinib.Simul_tvt_split_mp_netgen as test_prob_val


def discount_rewards(r,gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def OI(first_r, r):
    z = [first_r]
    z.extend(r)
    z = np.array(z)
    oi_r = np.zeros(len(z))

#    oi_r[0] = r[0]
    oi_r[0] = 0
    for i in range(1, z.size):
        oi_r[i] = np.max([z[i] - np.max(z[:i]), 0])
    return oi_r[1:]

def reset_vars(scope, old_var):
    scopevars = tf.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
    var_shapes = [i.shape for i in scopevars]

    op_holder = []
    for i,j in zip(scopevars,old_var):
        op_holder.append(i.assign(j))
    return op_holder

class experience_buffer():
    def __init__(self, buffer_size = 512, buffer_ready_size=4, isselect=False, iscontext=False, isgraphnet=False, isgsest=False, ismessage=False):
        self.key_list = ['prevAveract', 'prevAction', 'prevReward', 'prevReward2',
                'action_holder', 'action_probs', 'value_estimates', 'discounted_rewards', 'advantages',
                'dummyListIdx']
        if isselect:
            self.key_list += ['prevSelect', 'prevNewval']
        if iscontext:
            self.key_list += ['dummyParam']
        if isgraphnet:
            self.key_list += ['prevAction_graph']
        if isgsest:
            self.key_list += ['gs_state']
        if ismessage:
            self.key_list += ['message']
        self.reset()
        self.buffer_size = buffer_size
        self.buffer_ready_size = buffer_ready_size
        self.epi_idx = []

    def add(self,experience):
        if len(self.buffer[self.key_list[0]]) >= self.buffer_size:
            for key in self.key_list:
                self.buffer[key][0:self.buffer_ready_size] = []
        for key in self.key_list:
            self.buffer[key].append(experience[key])

    def shuffle(self):
        self.epi_idx = np.arange(len(self.buffer[self.key_list[0]])-self.buffer_ready_size, len(self.buffer[self.key_list[0]]))
        np.random.shuffle(self.epi_idx)

    def shuffle_for_context(self):
        self.epi_idx = np.arange(len(self.buffer[self.key_list[0]]))
        np.random.shuffle(self.epi_idx)

    def make_mini_batch(self,start,end):
        sampled_episodes = {}
        for key in self.key_list:
            #sampled_episodes[key] = np.asarray(self.buffer[key])[self.epi_idx[start:end]]
            sampled_episodes[key] = [self.buffer[key][_] for _ in self.epi_idx[start:end]]

        return sampled_episodes

    def reset(self):
        self.buffer = {}
        for key in self.key_list:
            self.buffer[key] = []

funclist = [nx.betweenness_centrality, nx.degree_centrality, nx.eigenvector_centrality_numpy,
nx.katz_centrality_numpy, nx.closeness_centrality, nx.load_centrality, nx.harmonic_centrality,
nx.pagerank_numpy]
funclist_edge = [nx.edge_betweenness_centrality]
#funclist_edge = []
def make_graph_dict(s, scale, adj, fitness):
    G = nx.convert_matrix.from_numpy_matrix(adj.T, create_using=nx.DiGraph)
    indices = np.where(adj)
    edges = np.eye(int(scale))[s.astype(np.int)]

    nodefeatures = []
    for f in funclist:
        try:
            fea = f(G)
            fea = list(fea.values())
            fea = [0 if (np.isinf(elem) or np.isnan(elem)) else elem for elem in fea]
        except:
            fea = [0] * adj.shape[0]
        nodefeatures.append(fea)

    '''
    pheno = np.zeros(adj.shape[0])
    pheno[phenoidx] = 1
    nodefeatures.append(list(pheno))

    pert = np.zeros(adj.shape[0])
    pert[pertidx] = 1
    nodefeatures.append(list(pert))

    nodefeatures.append(list(averact_one))
    nodefeatures.append(list(averact_mone))
    '''

    edgefeatures = []
    for f in funclist_edge:
        fea = f(G)
        edgefeatures.append([fea.get((indices[1][_],indices[0][_])) for _ in range(indices[0].size)])
    edgefeatures.append(list(adj[adj!=0]))
    if len(funclist_edge) > 0:
        edgefeatures = np.stack(edgefeatures).T
        edges = np.concatenate((edges, edgefeatures), axis=1)
    
    nodes = np.stack(nodefeatures).T
    n_nodes = adj.shape[0]
    return {
            "n_node":n_nodes,
            "nodes":nodes,
            "edges":edges,
            #"globals":fitness,
            "senders":indices[1].astype(np.int),
            "receivers":indices[0].astype(np.int),
            }

def make_msg_graph_dict(s, scale, adj, nf, lf, gf, fitness):
    indices = np.where(adj)
    edges_oh = np.eye(int(scale))[s.astype(np.int)]

    nodes = nf
    edges = np.concatenate((lf, edges_oh), axis=1)
    #glo = np.concatenate((gf.ravel(), fitness))
    glo = gf.ravel()

    n_nodes = adj.shape[0]
    return {
            "n_node":n_nodes,
            "nodes":nodes,
            "edges":edges,
            "globals":glo,
            "senders":indices[1].astype(np.int),
            "receivers":indices[0].astype(np.int),
            }

class AC_Network():
    def __init__(self, parameter_dict,action_width,scope,trainer,num_workers,global_step,worker_lists=None):
        self.worker_lists = worker_lists
        self.action_width = action_width
        self.a_size = parameter_dict['a_size']
        self.h_size = parameter_dict['h_size']
        self.node_size = parameter_dict['node_size']
        self.link_size = parameter_dict['link_size']
        self.num_input_node = parameter_dict['num_input_node']
        self.curiosity_encode_size = parameter_dict['curiosity_encode_size']
        self.curiosity_strength = parameter_dict['curiosity_strength']
        self.use_noisynet = parameter_dict['use_noisynet']
        self.use_update_noise = parameter_dict['use_update_noise']
        self.use_attention = parameter_dict['use_attention']
        self.use_context = parameter_dict['use_context']
        self.probabilistic_context = parameter_dict['probabilistic_context']
        self.use_context_v2 = parameter_dict['use_context_v2']
        self.use_gs_estimator = parameter_dict['use_gs_estimator']
        self.use_message = parameter_dict['use_message']
        self.message_dim = parameter_dict['message_dim']
        self.averact_context_target = parameter_dict['averact_context_target']
        self.gs_attention = parameter_dict['gs_attention']
        self.use_ib = parameter_dict['use_ib']
        self.use_varout = parameter_dict['use_varout']
        self.input_type = parameter_dict['input_type']
        self.output_type = parameter_dict['output_type']
        self.sg_unit_string = parameter_dict['sg_unit'].split('-')
        self.sg_unit = int(self.sg_unit_string[0])
        self.sg_layer = int(self.sg_unit_string[1])
        self.action_type = parameter_dict['action_type']
        self.action_branching = parameter_dict['action_branching']
        self.incremental_action = parameter_dict['incremental_action']
        self.stay_prev = parameter_dict['stay_prev']
        self.inc_bound = parameter_dict['inc_bound']
        self.action_width_real = action_width
        if self.action_type == 'continuous':
            self.action_width_real += 1
        if self.incremental_action:
            self.action_width = np.zeros_like(action_width) + self.inc_bound
        else:
            self.action_width = action_width
        self.autoregressive = parameter_dict['autoregressive']
        self.adj_mat = parameter_dict['adj_mat']

        self.trainer_pre = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
        self.trainer_context = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

        weightmat = self.adj_mat[self.num_input_node:,:]
        param_split = np.append(0,np.cumsum((weightmat!=0).sum(1)))
        idx_house = []
        for nn in range(self.node_size):
            #idx_house.append(np.append(np.arange(param_split[nn],param_split[nn+1]),self.link_size+nn))
            idx_house.append(np.arange(param_split[nn],param_split[nn+1]))
        self.idx_house = np.array(idx_house)

        with tf.compat.v1.variable_scope(scope):
            if self.action_type == 'select':
                self.entph = tf.compat.v1.placeholder(shape=[2],dtype=tf.float32)
            else:
                self.entph = tf.compat.v1.placeholder(shape=[None],dtype=tf.float32)
            self.entropy_histogram = tf.compat.v1.summary.histogram('entropy_histogram',self.entph)
            
            #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
            if self.input_type == 'fc':
                self.bs = tf.placeholder(shape=(), dtype=tf.int32)
                self.bst = tf.placeholder(shape=(), dtype=tf.int32)
                self.prevAction =  tf.placeholder(shape=[None,None,self.a_size],dtype=tf.float32) # [batch_size, trainLength, action_size(# of PKN params)]
                #l = self.prevAction
                l = []
                for asize in range(self.a_size):
                    l.append(tf.one_hot(tf.cast(self.prevAction[:,:,asize], tf.int32), self.action_width_real[asize], dtype=tf.float32))
                self.l = l = tf.concat(l,-1)
                self.prevReward = tf.placeholder(shape=[None,None,1],dtype=tf.float32)
                #l = tf.concat([l,self.prevReward],axis=2)
                #l = tf.layers.dense(l, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'))
            elif self.input_type == 'conv':
                self.prevAction =  tf.placeholder(shape=[None,self.node_size+self.num_input_node,self.node_size+self.num_input_node+1,1],dtype=tf.float32) # [batch_size, trainLength, action_size(# of PKN params)]
                self.prevReward = tf.placeholder(shape=[None,None,1],dtype=tf.float32)

                l = self.prevAction
                l = tf.layers.conv2d(l, 16, (4,4), activation=tf.nn.relu, padding='SAME')
                #l = tf.keras.layers.LocallyConnected2D(16, (4,4), activation=tf.nn.relu, implementation=1)(l)
                l = tf.layers.average_pooling2d(l, (2,2), (2,2), padding='SAME')
                #l = tf.layers.max_pooling2d(l, (2,2), (2,2), padding='SAME')
                l = tf.layers.conv2d(l, 32, (2,2), activation=tf.nn.relu, padding='SAME')
                #l = tf.keras.layers.LocallyConnected2D(32, (2,2), activation=tf.nn.relu, implementation=1)(l)
                l = tf.layers.average_pooling2d(l, (2,2), (2,2), padding='SAME')
                #l = tf.layers.max_pooling2d(l, (2,2), (2,2), padding='SAME')
                l = tf.reshape(l, [-1, int(np.prod(l.get_shape().as_list()[1:]))])
                l = tf.expand_dims(l, 0) #batch size if always 1
                #l = tf.concat([l,self.prevReward],axis=2)
                l = tf.layers.dense(l, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'))
            elif self.input_type == 'semi_graph':
                self.prevReward = tf.placeholder(shape=[None,None,1],dtype=tf.float32)
                self.bst = tf.placeholder(shape=(), dtype=tf.int32)
                self.prevAction = []
                for ns in range(self.node_size):
                    #elemnum = np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])+1
                    elemnum = np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])
                    tph = tf.placeholder(tf.float32, [None, elemnum])
                    self.prevAction.append(tph)
                l = [tf.reshape(tf.one_hot(tf.cast(pa, tf.int32),self.action_width_real[0]), [-1, int(pa.get_shape()[1]*self.action_width_real[0])]) for pa in self.prevAction]
                for sgl in range(self.sg_layer):
                    l = self._make_semi_graph(l, self.sg_unit, sgl)

                if self.use_attention:
                    la = [tf.expand_dims(lt, 1) for lt in l]
                    la = tf.concat(la, axis=1)
                    l = self._attention(la, 512, 8)
                    l = tf.reshape(l, shape=[-1, np.prod(l.get_shape()[1:])])
                else:
                    l = tf.concat(l, axis=1)
                    #l = tf.expand_dims(l, 0)

                fd = l.get_shape()[-1]
                l = tf.reshape(l,shape=[-1,self.bst,fd])
                l = tf.layers.dense(l, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'), use_bias=False)
            elif self.input_type == 'gru_semi_graph':
                self.bs = tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
                self.bst = tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
                self.prevReward = tf.compat.v1.placeholder(shape=[None,None,1],dtype=tf.float32)
                self.prevReward2 = tf.compat.v1.placeholder(shape=[None,None,2],dtype=tf.float32)
                self.prevAction = []
                '''
                for ns in range(self.node_size):
                    self.prevAction.append(tf.placeholder(tf.float32, [None, np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])+1]))
                l = self.prevAction
                '''
                for ns in range(self.node_size):
                    #elemnum = np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])+1
                    elemnum = np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])
                    tph = tf.compat.v1.placeholder(tf.float32, [None, elemnum])
                    self.prevAction.append(tph)
                l_first = l = [tf.reshape(tf.one_hot(tf.cast(pa, tf.int32),self.action_width_real[0]), [-1, int(pa.get_shape()[1]*self.action_width_real[0])]) for pa in self.prevAction]

                def make_input_layer(x):
                    l = self._make_semi_graph(x, self.sg_unit, 0)
                    l = self._make_gru_semi_graph(l, self.sg_unit, self.sg_layer)

                    if self.use_attention:
                        la = [tf.expand_dims(lt, 1) for lt in l]
                        la = tf.concat(la, axis=1)
                        l = self._attention(la, 512, 8)
                        l = tf.reshape(l, shape=[-1, np.prod(l.get_shape()[1:])])
                    elif self.use_varout:
                        l = tf.concat([tf.expand_dims(lt, 1) for lt in l], axis=1)
                    else:
                        l = tf.concat(l, axis=1)
                        #l = tf.expand_dims(l, 0)

                    fd = l.get_shape()
                    if self.use_varout:
                        with tf.compat.v1.variable_scope('gs'):
                            l = tf.reshape(l, shape=[-1,self.bst,fd[-2],fd[-1]])
                            if self.gs_attention:
                                q = tf.layers.dense(l, self.sg_unit)
                                k = tf.layers.dense(l, self.sg_unit)
                                v = tf.layers.dense(l, self.sg_unit)

                                matmul_qk = tf.matmul(q, k, transpose_b=True)
                                dk = tf.cast(tf.shape(k)[-1], tf.float32)
                                scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
                                attention_weights = tf.nn.softmax(scaled_attention_logits)

                                lcb = tf.matmul(attention_weights, v)

                                l = tf.transpose(l, [0,2,1,3])
                                lcb = tf.transpose(lcb, [0,2,1,3])
                            else:
                                #self.lcb = lcb = tf.reshape(l, [-1, self.bst, int(np.prod(fd[-2:]))])
                                l = tf.transpose(l, [0,2,1,3])
                                lcb = tf.reduce_mean(l, axis=1)
                                lcb = tf.layers.dense(lcb, 128, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                    else:
                        l = tf.reshape(l,shape=[-1,self.bst,fd[-1]])
                        lcb = l
                    return l, lcb

                l, lcb = make_input_layer(l)
                self.lcb = lcb


                if self.use_context_v2:
                    with tf.compat.v1.variable_scope('context'):
                        lc, lcbc = make_input_layer(l_first)
                        self.lcbc = lcbc

                if self.use_context:
                    with tf.compat.v1.variable_scope('sg_mix'):
                        lc = tf.layers.dense(lcb, 128, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                    self.dummy_prevAction = []
                    '''
                    for ns in range(self.node_size):
                        self.prevAction.append(tf.placeholder(tf.float32, [None, np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])+1]))
                    l = self.prevAction
                    '''
                    for ns in range(self.node_size):
                        #elemnum = np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])+1
                        elemnum = np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])
                        tph = tf.placeholder(tf.float32, [None, elemnum])
                        self.dummy_prevAction.append(tph)
                    dl = [tf.reshape(tf.one_hot(tf.cast(pa, tf.int32),self.action_width_real[0]), [-1, int(pa.get_shape()[1]*self.action_width_real[0])]) for pa in self.dummy_prevAction]
                    dl = self._make_semi_graph(dl, self.sg_unit, 0, reuse=True)
                    dl = self._make_gru_semi_graph(dl, self.sg_unit, self.sg_layer, reuse=True)

                    if self.use_varout:
                        dl = tf.concat([tf.expand_dims(lt, 1) for lt in dl], axis=1)
                    else:
                        dl = tf.concat(dl, axis=1)

                    fd = dl.get_shape()
                    if self.use_varout:
                        dl = tf.reshape(dl, shape=[-1,self.bst,fd[-2],fd[-1]])
                        dl = tf.transpose(dl, [0,2,1,3])
                    else:
                        dl = tf.reshape(dl,shape=[-1,self.bst,fd[-1]])
                    with tf.compat.v1.variable_scope('sg_mix'):
                        if self.use_varout:
                            dl = tf.reduce_sum(dl, axis=1)
                        context_vector_target = tf.layers.dense(dl, 128, activation=None, reuse=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                    if self.probabilistic_context:
                        c_mean_target = context_vector_target[:,:,:self.h_size//2]
                        c_var_target = tf.nn.softplus(context_vector_target[:,:,self.h_size//2:]) + 1e-7
                        self.context_vector_target_dist = tfp.distributions.Normal(c_mean_target, c_var_target)
                        #self.context_vector_target = tf.stop_gradient(self.context_vector_target_dist)
                    else:
                        self.context_vector_target = tf.stop_gradient(context_vector_target)

            elif self.input_type == 'graph_net':
                self.bs = tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
                self.bst = tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
                self.prevReward = tf.compat.v1.placeholder(shape=[None,None,1],dtype=tf.float32)
                self.prevReward2 = tf.compat.v1.placeholder(shape=[None,None,3],dtype=tf.float32)

                if self.use_message:
                    def get_msg_graph_data_dict(num_nodes, num_edges):
                        return {
                            "globals": np.random.rand(self.message_dim).astype(np.float32),
                            "nodes": np.random.rand(num_nodes, self.message_dim).astype(np.float32),
                            "edges": np.random.rand(num_edges, int(self.action_width[0]+self.message_dim)).astype(np.float32),
                            "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
                            "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
                        }
                    msg_graph_input_template = [get_msg_graph_data_dict(2,2)]

                    self.prevMsg = utils_tf.placeholders_from_data_dicts(msg_graph_input_template[0:1])

                def get_graph_data_dict(num_nodes, num_edges):
                    return {
                        #"globals": np.random.rand(4).astype(np.float32),
                        "nodes": np.random.rand(num_nodes, len(funclist)).astype(np.float32),
                        "edges": np.random.rand(num_edges, int(self.action_width[0]+len(funclist_edge)+1)).astype(np.float32),
                        "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
                        "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
                    }
                graph_input_template = [get_graph_data_dict(2,2)]

                self.prevAction = utils_tf.placeholders_from_data_dicts(graph_input_template[0:1])

                def make_input_layer(x, sgunit, ismsg=False):
                    if not ismsg:
                        #x = utils_tf.set_zero_node_features(x, 1)
                        x = utils_tf.set_zero_global_features(x, 1)
                    def mish(inp):
                        return inp * tf.nn.tanh(tf.nn.softplus(inp))
                    def make_mlp_model():
                        return snt.Sequential([
                          snt.nets.MLP([sgunit] * 1, activation=tf.nn.elu, activate_final=True),
                          snt.LayerNorm()
                        ])
                    def make_mlp_model_sig():
                        return snt.Sequential([
                          snt.nets.MLP([sgunit] * 1, activation=tf.nn.sigmoid, activate_final=True),
                        ])
                    def make_mlp_model_tanh():
                        return snt.Sequential([
                          snt.nets.MLP([sgunit] * 1, activation=tf.nn.tanh, activate_final=True),
                        ])

                    encoder = modules.GraphIndependent(
                        edge_model_fn=make_mlp_model,
                        node_model_fn=make_mlp_model,
                        global_model_fn=make_mlp_model)
                    processor = modules.GraphNetwork(
                        edge_model_fn=make_mlp_model,
                        node_model_fn=make_mlp_model,
                        global_model_fn=make_mlp_model)
                    if ismsg:
                        decoder = modules.GraphIndependent(
                            edge_model_fn=make_mlp_model_sig,
                            node_model_fn=make_mlp_model_sig,
                            global_model_fn=make_mlp_model_sig)
                    else:
                        decoder = modules.GraphIndependent(
                            edge_model_fn=make_mlp_model,
                            node_model_fn=make_mlp_model,
                            global_model_fn=make_mlp_model)
                    output_transform = modules.GraphIndependent(
                        edge_model_fn=lambda: snt.Linear(sgunit*2, name="edge_output"),
                        node_model_fn=lambda: snt.Linear(sgunit*2, name="node_output"),
                        global_model_fn=lambda: snt.Linear(sgunit*2, name="global_output"))

                    graph_out = encoder(x)
                    graph_init = graph_out
                    for _ in range(self.sg_layer):
                        core_input = utils_tf.concat([graph_init, graph_out], axis=1)
                        graph_out = processor(core_input)
                    graph_out = decoder(graph_out)
                    #graph_out = output_transform(graph_out)
                    l = graph_out.nodes
                    le = graph_out.edges
                    lcb = graph_out.globals

                    return l, le, lcb, graph_out

                l, le, lcb, graph_out = make_input_layer(self.prevAction, self.sg_unit)
                self.lcb = lcb
                self.graph_out = graph_out
                self.linkperbatch = tf.reshape(self.prevAction.n_edge, [-1,self.bst])[:,0]

                if self.use_context_v2:
                    with tf.compat.v1.variable_scope('context'):
                        lc, lec, lcbc, _ = make_input_layer(self.prevAction, self.sg_unit)
                        self.lcbc = lcbc
                if self.use_message:
                    with tf.compat.v1.variable_scope('message'):
                        lm, lem, lcbm, _ = make_input_layer(self.prevMsg, self.message_dim, ismsg=True)
                        self.msg_gf = self.lcbm = lcbm
                        self.msg_nf = lm
                        self.msg_lf = lem
                else:
                    self.msg_nf = tf.constant(0)
                    self.msg_lf = tf.constant(0)
                    self.msg_gf = tf.constant(0)


            elif self.input_type == 'attention':
                self.bs = tf.placeholder(shape=(), dtype=tf.int32)
                self.bst = tf.placeholder(shape=(), dtype=tf.int32)
                self.prevReward = tf.placeholder(shape=[None,None,1],dtype=tf.float32)
                self.prevAction = []
                '''
                for ns in range(self.node_size):
                    self.prevAction.append(tf.placeholder(tf.float32, [None, np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])+1]))
                l = [tf.expand_dims(tf.layers.dense(pa, self.sg_unit), 1) for pa in self.prevAction]
                '''
                for ns in range(self.node_size):
                    #elemnum = np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])+1
                    elemnum = np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])
                    tph = tf.placeholder(tf.float32, [None, elemnum])
                    self.prevAction.append(tph)
                l = [tf.reshape(tf.one_hot(tf.cast(pa, tf.int32),self.action_width_real[0]), [-1, int(pa.get_shape()[1]*self.action_width_real[0])]) for pa in self.prevAction]
                l = [tf.expand_dims(tf.layers.dense(pa, self.sg_unit), 1) for pa in l]
                l = tf.concat(l, axis=1)
                for _ in range(self.sg_layer):
                    l = self._attention(l, 512, 8)
                l = tf.reshape(l, shape=[-1, np.prod(l.get_shape()[1:])])

                fd = l.get_shape()[-1]
                l = tf.reshape(l,shape=[-1,self.bst,fd])

                l = tf.layers.dense(l, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'))

            #self.prevAveract = tf.compat.v1.placeholder(shape=[None,None,self.node_size+self.num_input_node],dtype=tf.float32) # [batch_size, trainLength, node_size]
            self.prevAveract = tf.compat.v1.placeholder(shape=[None,1],dtype=tf.float32) # [batch_size*node_size, 1]
            #pa = tf.layers.dense(self.prevAveract, self.sg_unit, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-3))
            if self.action_type == 'select':
                self.prevSelect = tf.placeholder(shape=[None,None,1],dtype=tf.float32)
                self.ps = ps = tf.squeeze(tf.one_hot(tf.cast(self.prevSelect, tf.int32),self.a_size), -2)
                #self.ps = ps = tf.squeeze(tf.one_hot(tf.cast(self.prevSelect, tf.int32),self.node_size), -2)
                self.prevNewval = tf.placeholder(shape=[None,None,1],dtype=tf.float32)
                self.nv = nv = tf.squeeze(tf.one_hot(tf.cast(self.prevNewval, tf.int32),self.action_width[0]), -2)
                #self.nv = nv = tf.squeeze(tf.one_hot(tf.cast(self.prevNewval, tf.int32),5), -2)
                psnv_emb = tf.concat([ps,nv], axis=-1)
                psnv_emb = tf.layers.dense(psnv_emb, self.sg_unit, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
            
            self.context_state_init, self.context_rnn_state = 0, tf.constant(0)
            if self.use_context:
                context_input = tf.concat([lc, self.prevReward], axis=2)
                #self.context_vector = tf.layers.dense(context_input, 128, activation=None)
                context_lstm = tf.contrib.rnn.LSTMBlockCell(self.h_size)
                cc_init = np.zeros((1, context_lstm.state_size.c), np.float32)
                ch_init = np.zeros((1, context_lstm.state_size.h), np.float32)
                self.context_state_init = [cc_init, ch_init]
                cc_in = tf.compat.v1.placeholder(tf.float32, [None, context_lstm.state_size.c])
                ch_in = tf.compat.v1.placeholder(tf.float32, [None, context_lstm.state_size.h])
                self.context_state_in = (cc_in, ch_in)
                context_state_in = tf.nn.rnn_cell.LSTMStateTuple(cc_in, ch_in)
                self.context_vector,self.context_rnn_state = tf.nn.dynamic_rnn(\
                    inputs=context_input,cell=context_lstm,dtype=tf.float32,initial_state=context_state_in,scope='context_rnn')
                self.context_vector_for_input = tf.stop_gradient(self.context_vector)

                if self.probabilistic_context:
                    c_mean = self.context_vector[:,:,:self.h_size//2]
                    c_var = tf.nn.softplus(self.context_vector[:,:,self.h_size//2:]) + 1e-7
                    self.context_vector_dist = tfp.distributions.Normal(c_mean, c_var)
                    self.context_vector_for_input = tf.stop_gradient(self.context_vector_dist.sample())

            if self.use_varout:
                self.iidx = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 2, 2])
                self.zpsize = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
                if self.input_type != 'graph_net':
                    zp = tf.zeros([self.zpsize,self.num_input_node,self.bst,self.sg_unit])
                    self.lp = lp = tf.concat([zp,l], axis=1)
                    link_feature = tf.gather_nd(lp, self.iidx)
                    link_feature = tf.transpose(link_feature, [0,2,1,3])
                    link_feature = tf.reshape(link_feature, [-1, self.bst, np.prod(link_feature.get_shape()[-2:])])
                else:
                    self.graph_iidx = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
                    self.lp = lp = l
                    #link_feature = tf.gather_nd(lp, self.prevAction.senders)
                    link_feature_s = tf.gather(lp, self.prevAction.senders)
                    link_feature_r = tf.gather(lp, self.prevAction.receivers)
                    link_feature = tf.concat([link_feature_s, link_feature_r, le], -1)
                    link_feature = tf.gather(link_feature, self.graph_iidx) #[batch*link*time,depth]
                    link_feature = tf.reshape(link_feature, [-1, self.bst, link_feature.get_shape()[-1]]) #[batch*link,time,depth]
                    self.link_feature = link_feature
                    self.senrec = link_feature[:,:,:(link_feature_s.get_shape()[-1]*2)]
                    self.linklink = link_feature[:,:,(link_feature_s.get_shape()[-1]*2):]
                if self.use_context_v2:
                    with tf.compat.v1.variable_scope('context'):
                        if self.input_type != 'graph_net':
                            self.lpc = lpc = tf.concat([zp,lc], axis=1)
                            link_feature_c = tf.gather_nd(lpc, self.iidx)
                            link_feature_c = tf.transpose(link_feature_c, [0,2,1,3])
                            link_feature_c = tf.reshape(link_feature_c, [-1, self.bst, np.prod(link_feature_c.get_shape()[-2:])])
                        else:
                            self.lpc = lpc = lc
                            link_feature_cs = tf.gather(lpc, self.prevAction.senders)
                            link_feature_cr = tf.gather(lpc, self.prevAction.receivers)
                            link_feature_c = tf.concat([link_feature_cs, link_feature_cr, lec], -1)
                            link_feature_c = tf.gather(link_feature_c, self.graph_iidx)
                            link_feature_c = tf.reshape(link_feature_c, [-1, self.bst, link_feature_c.get_shape()[-1]])
                if self.use_message:
                    with tf.compat.v1.variable_scope('message'):
                        self.lpm = lpm = lm
                        link_feature_ms = tf.gather(lpm, self.prevMsg.senders)
                        link_feature_mr = tf.gather(lpm, self.prevMsg.receivers)
                        link_feature_m = tf.concat([link_feature_ms, link_feature_mr, lem], -1)
                        link_feature_m = tf.gather(link_feature_m, self.graph_iidx)
                        link_feature_m = tf.reshape(link_feature_m, [-1, self.bst, link_feature_m.get_shape()[-1]])

            if self.use_varout:
                self.netvec = link_feature
                if self.use_context:
                    self.context_vector_for_input = tf.expand_dims(self.context_vector_for_input, axis=1)
                    self.context_vector_for_input = tf.tile(self.context_vector_for_input, [1,self.link_size,1,1])
                    self.context_vector_for_input = tf.reshape(self.context_vector_for_input, [-1, self.bst, self.context_vector_for_input.get_shape()[-1]])
                if self.gs_attention:
                    self.lcbp = lcbp = tf.concat([zp,self.lcb], axis=1)
                    gs = tf.gather_nd(lcbp, self.iidx)
                    gs = tf.transpose(gs, [0,2,1,3])
                    self.global_state = gs = tf.reshape(gs, [-1, self.bst, np.prod(gs.get_shape()[-2:])])
                else:
                    if self.input_type != 'graph_net':
                        gs = tf.expand_dims(self.lcb, axis=1)
                        gs = tf.tile(gs, [1,self.link_size,1,1])
                        self.global_state = gs = tf.reshape(gs, [-1, self.bst, gs.get_shape()[-1]])
                    else:
                        gs = self.lcb
                        gs = tf.reshape(gs, [-1, self.bst, gs.get_shape()[-1]])
                        gs = tf.repeat(gs, self.linkperbatch, axis=0)
                        self.global_state = gs
                if self.use_context_v2:
                    with tf.compat.v1.variable_scope('context'):
                        if self.gs_attention:
                            self.lcbcp = lcbcp = tf.concat([zp,self.lcbc], axis=1)
                            gsc = tf.gather_nd(lcbcp, self.iidx)
                            gsc = tf.transpose(gsc, [0,2,1,3])
                            gsc = tf.reshape(gsc, [-1, self.bst, np.prod(gsc.get_shape()[-2:])])
                        else:
                            if self.input_type != 'graph_net':
                                gsc = tf.expand_dims(self.lcbc, axis=1)
                                gsc = tf.tile(gsc, [1,self.link_size,1,1])
                                gsc = tf.reshape(gsc, [-1, self.bst, gsc.get_shape()[-1]])
                            else:
                                gsc = self.lcbc
                                gsc = tf.reshape(gsc, [-1, self.bst, gsc.get_shape()[-1]])
                                gsc = tf.repeat(gsc, self.linkperbatch, axis=0)
                if self.use_message:
                    with tf.compat.v1.variable_scope('message'):
                        gsm = self.lcbm
                        gsm = tf.reshape(gsm, [-1, self.bst, gsm.get_shape()[-1]])
                        gsm = tf.repeat(gsm, self.linkperbatch, axis=0)
                        self.global_state_msg = gsm
                if self.input_type != 'graph_net':
                    pr = tf.tile(self.prevReward, [1,1,self.link_size])
                    pr = tf.transpose(pr, [0,2,1])
                    self.pr = pr = tf.reshape(pr, [-1, self.bst, 1])
                    self.num_additional_reward = self.prevReward2.get_shape()[-1]
                    self.pr2 = pr2 = tf.reshape(tf.tile(self.prevReward2,[1,self.link_size,1]), [-1, self.bst, self.num_additional_reward])
                else:
                    self.pr = pr = tf.repeat(self.prevReward, self.linkperbatch, axis=0)
                    self.num_additional_reward = self.prevReward2.get_shape()[-1]
                    self.pr2 = pr2 = tf.repeat(self.prevReward2, self.linkperbatch, axis=0)
                self.prevAction_raw = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,None,1])
                self.pa_oh = pa_oh = tf.squeeze(tf.one_hot(tf.cast(self.prevAction_raw, tf.int32), self.action_width[0]), -2)
            else:
                self.netvec = l

            if self.use_context_v2:
                with tf.compat.v1.variable_scope('context'):
                    lfm_c = tf.stop_gradient(link_feature_m)
                    gsm_c = tf.stop_gradient(self.global_state_msg)
                    context_input = tf.concat([link_feature_c, pr, pr2, pa_oh, gsc, lfm_c, gsm_c], axis=2)
                    context_input = tf.layers.dense(context_input, self.h_size, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                    #context_input = tf.layers.dense(context_input, self.h_size, activation=None)
                    #context_lstm = tf.contrib.rnn.LSTMBlockCell(self.h_size)
                    context_lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                    cc_init = np.zeros((1, context_lstm.state_size.c), np.float32)
                    ch_init = np.zeros((1, context_lstm.state_size.h), np.float32)
                    self.context_state_init = [cc_init, ch_init]
                    cc_in = tf.compat.v1.placeholder(tf.float32, [None, context_lstm.state_size.c])
                    ch_in = tf.compat.v1.placeholder(tf.float32, [None, context_lstm.state_size.h])
                    self.context_state_in = (cc_in, ch_in)
                    context_state_in = tf.nn.rnn_cell.LSTMStateTuple(cc_in, ch_in)
                    self.context_vector,self.context_rnn_state = tf.nn.dynamic_rnn(\
                        inputs=context_input,cell=context_lstm,dtype=tf.float32,initial_state=context_state_in,scope='context_rnn')
                    if self.use_ib:
                        c_mean = tf.layers.dense(self.context_vector, 128, use_bias=False, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                        c_var = 0.001 + (1 - 0.001) * tf.nn.sigmoid(tf.math.log(tf.layers.dense(self.context_vector, 128, use_bias=False, activation=tf.nn.softplus, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))))

                        self.context_vector_dist = tfp.distributions.Normal(c_mean, c_var)
                        chid = self.context_vector_dist.sample()
                    else:
                        chid = self.context_vector

                    chid = tf.layers.dense(chid, 128, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                    chid = tf.layers.dense(chid, 128, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                    self.context_vector_for_input = tf.stop_gradient(chid)
                    if self.averact_context_target:
                        self.context_logit_aa = tf.layers.dense(chid, 2, use_bias=False)
                    if self.stay_prev:
                        self.context_logit = tf.layers.dense(chid, self.action_width_real[0]-1, use_bias=False)
                    else:
                        self.context_logit = tf.layers.dense(chid, self.action_width_real[0], use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))

            if self.use_gs_estimator:
                with tf.compat.v1.variable_scope('gs_estimator'):
                    #context_input = tf.concat([link_feature_c, pr, pr2, pa_oh, gsc], axis=2)
                    #context_input = tf.layers.dense(context_input, self.h_size, activation=tf.nn.elu)
                    gs_lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(int(self.h_size/2))
                    gc_init = np.zeros((1, gs_lstm.state_size.c), np.float32)
                    gh_init = np.zeros((1, gs_lstm.state_size.h), np.float32)
                    self.gs_state_init = [gc_init, gh_init]
                    gc_in = tf.compat.v1.placeholder(tf.float32, [None, gs_lstm.state_size.c])
                    gh_in = tf.compat.v1.placeholder(tf.float32, [None, gs_lstm.state_size.h])
                    self.gs_state_in = (gc_in, gh_in)
                    gs_state_in = tf.nn.rnn_cell.LSTMStateTuple(gc_in, gh_in)

                    #self.gs_est = tf.compat.v1.placeholder(tf.float32, [None, gs_lstm.state_size.c+gs_lstm.state_size.h])
                    self.gs_est = tf.compat.v1.placeholder(tf.float32, [None, 128*3])
                    self.gs_est_for_input = tf.reshape(self.gs_est, [-1,self.bst,self.gs_est.get_shape()[-1]])

            if self.action_type == 'select':
                if self.use_context:
                    lstm_input = tf.concat([l,self.prevReward,self.context_vector_for_input,psnv_emb],axis=2)
                else:
                    lstm_input = tf.concat([l,self.prevReward,psnv_emb],axis=2)
            else:
                input_list = [self.netvec,pr,pr2,pa_oh,gs]
                if self.use_context or self.use_context_v2:
                    input_list.append(self.context_vector_for_input)
                if self.use_gs_estimator:
                    input_list.append(self.gs_est_for_input)
                if self.use_message:
                    input_list.append(link_feature_m)
                    input_list.append(gsm)
                self.li = lstm_input = tf.concat(input_list,axis=2)
            lstm_input = tf.layers.dense(lstm_input, self.h_size, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
            #lstm_input = tf.layers.dense(lstm_input, self.h_size, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
            if not self.use_update_noise:
                #lstm_cell1 = tf.contrib.rnn.LSTMBlockCell(self.h_size)
                #lstm_cell2 = tf.contrib.rnn.LSTMBlockCell(self.h_size)
                lstm_cell1 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                lstm_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
            else:
                lstm_cell1 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                lstm_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
            rnn_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell1, lstm_cell2])
            #rnn_cells = lstm_cell1
            c_init = np.zeros((1, lstm_cell1.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell1.state_size.h), np.float32)
            self.state_init = [[c_init, h_init]]*2
            #self.state_init = [c_init, h_init]
            c_in1 = tf.compat.v1.placeholder(tf.float32, [None, lstm_cell1.state_size.c])
            h_in1 = tf.compat.v1.placeholder(tf.float32, [None, lstm_cell1.state_size.h])
            c_in2 = tf.compat.v1.placeholder(tf.float32, [None, lstm_cell2.state_size.c])
            h_in2 = tf.compat.v1.placeholder(tf.float32, [None, lstm_cell2.state_size.h])
            self.state_in = ((c_in1, h_in1),(c_in2, h_in2))
            #self.state_in = (c_in1, h_in1)
            state_in1 = tf.nn.rnn_cell.LSTMStateTuple(c_in1, h_in1)
            state_in2 = tf.nn.rnn_cell.LSTMStateTuple(c_in2, h_in2)
            self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
                    inputs=lstm_input,cell=rnn_cells,dtype=tf.float32,initial_state=(state_in1, state_in2),scope='meta_rnn')
                    #inputs=lstm_input,cell=rnn_cells,dtype=tf.float32,initial_state=state_in1,scope='meta_rnn')
            
            self.rnn = tf.reshape(self.rnn,shape=[-1,self.h_size])
            '''
            if self.input_type == 'semi_graph' or self.input_type == 'gru_semi_graph' or self.input_type == 'attention':
                actcon = tf.concat(self.prevAction, axis=1)
                self.rnn = tf.concat([actcon, self.rnn], axis=1)
            '''
            #self.rnn = self.noisy_dense(self.rnn, self.h_size*2, 'hidden_fc', bias=True, activation_fn=tf.nn.relu)
            #self.rnn = tf.layers.dropout(self.rnn, training=True)

            if self.action_type == 'discrete':
                if self.action_branching:
                    with tf.compat.v1.variable_scope('action_branch'):
                        self.ab = []
                        for ns in range(self.node_size):
                            self.ab.append(tf.layers.dense(self.rnn, 32, activation=tf.nn.relu))
                        if self.output_type == 'fc':
                            abl = self.ab
                        elif self.output_type == 'semi_graph':
                            abl = self.ab
                            for sgl in range(self.sg_layer):
                                abl = self._make_semi_graph(abl, self.sg_unit, sgl)
                        elif self.output_type == 'gru_semi_graph':
                            abl = self._make_gru_semi_graph(self.ab, self.sg_unit, self.sg_layer)

                        link_weight_branches = []
                        basal_branches = []
                        for ns in range(self.node_size):
                            for nsa in range(self.idx_house[ns].size):
                                if self.incremental_action:
                                    actout = tf.layers.dense(abl[ns], self.inc_bound, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
                                else:
                                    actout = tf.layers.dense(abl[ns], self.action_width[self.idx_house[ns][nsa]], activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

                                link_weight_branches.append(actout)
                        '''        
                                if nsa == (self.idx_house[ns].size-1):
                                    basal_branches.append(actout)
                                else:
                                    link_weight_branches.append(actout)

                        link_weight_branches.extend(basal_branches)
                        '''        
                        policy_branches = link_weight_branches

                elif self.autoregressive:
                    policy_branches = self._make_autoregressive_output(self.rnn)
                elif self.use_varout:

                    pb_temp = tf.layers.dense(self.rnn, self.h_size, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                    self.pb =policy_branches = [tf.layers.dense(pb_temp, self.action_width[0], activation=None, use_bias=False,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))]
                else:
                    policy_branches = []
                    for size in self.action_width:
                        policy_branches.append(tf.layers.dense(self.rnn, size, activation=None, use_bias=False,
                            #kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.01)))
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)))
                                                               

                self.all_log_probs = tf.concat([branch for branch in policy_branches], axis=1, name="action_probs")

                if self.use_varout:
                    output, normalized_logits = self.create_discrete_action_masking_layer(self.all_log_probs, [self.action_width[0]])
                    self.policy = tf.identity(output, name="action")
                    action_idx = [0] + list(np.cumsum(self.action_width[0],dtype=np.int32))
                    branches_logits = [normalized_logits[:, action_idx[i]:action_idx[i + 1]] for i in range(len([self.action_width[0]]))]
                    mode_output = tf.concat([tf.expand_dims(tf.argmax(bl, axis=1),axis=1) for bl in branches_logits], axis=1)
                    self.mode_policy = tf.identity(mode_output, name='mode_action')
                else:
                    output, normalized_logits = self.create_discrete_action_masking_layer(self.all_log_probs, self.action_width)
                    self.policy = tf.identity(output, name="action")
                    action_idx = [0] + list(np.cumsum(self.action_width,dtype=np.int32))
                    branches_logits = [normalized_logits[:, action_idx[i]:action_idx[i + 1]] for i in range(len(self.action_width))]
                    mode_output = tf.concat([tf.expand_dims(tf.argmax(bl, axis=1),axis=1) for bl in branches_logits], axis=1)
                    self.mode_policy = tf.identity(mode_output, name='mode_action')

            elif self.action_type == 'continuous':
                if self.action_branching:
                    with tf.compat.v1.variable_scope('action_branch'):
                        self.ab = []
                        for ns in range(self.node_size):
                            self.ab.append(tf.layers.dense(self.rnn, 32, activation=tf.nn.relu))
                        if self.output_type == 'semi_graph':
                            abl = self.ab
                            for sgl in range(self.sg_layer):
                                abl = self._make_semi_graph(abl, self.sg_unit, sgl)
                        elif self.output_type == 'gru_semi_graph':
                            abl = self._make_gru_semi_graph(self.ab, self.sg_unit, self.sg_layer)

                        link_weight_alpha = []
                        link_weight_beta = []
                        basal_alpha = []
                        basal_beta = []
                        for ns in range(self.node_size):
                            actout_alpha = tf.layers.dense(abl[ns], self.idx_house[ns].size-1, activation=tf.nn.softplus)+ 1 + 1e-7
                            actout_beta = tf.layers.dense(abl[ns], self.idx_house[ns].size-1, activation=tf.nn.softplus) + 1 + 1e-7

                            link_weight_alpha.append(actout_alpha)
                            link_weight_beta.append(actout_beta)

                        '''
                            actout_b_alpha = tf.layers.dense(abl[ns], 1, activation=tf.nn.softplus) + 1 + 1e-7
                            actout_b_beta = tf.layers.dense(abl[ns], 1, activation=tf.nn.softplus) + 1 + 1e-7
                            basal_alpha.append(actout_b_alpha)
                            basal_beta.append(actout_b_beta)

                        link_weight_alpha.extend(basal_alpha)
                        link_weight_beta.extend(basal_beta)
                        '''
                        self.alpha = tf.concat(link_weight_alpha, 1)
                        self.beta = tf.concat(link_weight_beta, 1)


                else:
                    if not self.use_noisynet:
                        alpha = tf.layers.dense(self.rnn, self.a_size, activation=tf.nn.softplus, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'), kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                        self.alpha = alpha + 1 + 1e-7
                        beta = tf.layers.dense(self.rnn, self.a_size, activation=tf.nn.softplus, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'), kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                        self.beta = beta + 1 + 1e-7
                    else:
                        alpha = self.noisy_dense(self.rnn, self.a_size, 'alpha', bias=False, activation_fn=tf.nn.softplus)
                        self.alpha = alpha + 1 + 1e-7
                        beta = self.noisy_dense(self.rnn, self.a_size, 'beta', bias=False, activation_fn=tf.nn.softplus)
                        self.beta = beta + 1 + 1e-7

                self.sample_dist = tf.distributions.Beta(self.alpha, self.beta) # small alpha -> left skewed distribution
                output = self.sample_dist.sample([1])
                output = tf.clip_by_value(output, 1e-7, 1-(1e-7))
                self.policy = tf.identity(output, name="action")
                mode_output = self.sample_dist.mode()
                mode_output = tf.clip_by_value(mode_output, 1e-7, 1-(1e-7))
                self.mode_policy = tf.identity(mode_output, name='mode_action')
            
                all_log_probs = self.sample_dist.log_prob(self.policy)
                self.all_log_probs = tf.identity(all_log_probs, name="action_probs")
            
            elif self.action_type == 'select':
                if not self.autoregressive:
                    select_branch = tf.layers.dense(self.rnn, 128, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                    select_branch = tf.layers.dense(select_branch, self.a_size, activation=None, use_bias=False)
                    #select_branch = tf.layers.dense(select_branch, self.node_size, activation=None, use_bias=False)
                    param_branch = tf.layers.dense(self.rnn, 128, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                    if self.incremental_action:
                        param_branch = tf.layers.dense(param_branch, self.inc_bound, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                        self.select_action_array = [self.a_size, self.inc_bound]
                    else:
                        param_branch = tf.layers.dense(param_branch, self.action_width[0], activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                        self.select_action_array = [self.a_size, self.action_width[0]]
                        #param_branch = tf.layers.dense(param_branch, 5, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-3))
                        #self.select_action_array = [self.node_size, 5]

                    policy_branches = [select_branch, param_branch]

                    self.all_log_probs = tf.concat(policy_branches, axis=1, name="action_probs")

                    output, normalized_logits = self.create_discrete_action_masking_layer(self.all_log_probs, self.select_action_array)
                    self.policy = tf.identity(output, name="action")
                    action_idx = [0] + list(np.cumsum(self.select_action_array,dtype=np.int32))
                    branches_logits = [normalized_logits[:, action_idx[i]:action_idx[i + 1]] for i in range(len(self.select_action_array))]
                    mode_output = tf.concat([tf.expand_dims(tf.argmax(bl, axis=1),axis=1) for bl in branches_logits], axis=1)
                    self.mode_policy = tf.identity(mode_output, name='mode_action')
                else:
                    select_branch = tf.layers.dense(self.rnn, 128, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                    select_branch = tf.layers.dense(select_branch, self.a_size, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                    #select_branch = tf.layers.dense(select_branch, self.node_size, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-3))

                    output_s, normalized_logits_s = self.create_discrete_action_masking_layer(select_branch, [self.a_size])

                    pbinput = tf.squeeze(tf.one_hot(tf.cast(output_s, tf.int32), self.a_size, dtype=tf.float32), -2)
                    #pbinput = tf.squeeze(tf.one_hot(tf.cast(output_s, tf.int32), self.node_size, dtype=tf.float32), -2)
                    pbinput = tf.concat([self.rnn, pbinput], axis=-1)

                    param_branch = tf.layers.dense(pbinput, 128, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                    if self.incremental_action:
                        param_branch = tf.layers.dense(param_branch, self.inc_bound, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                        self.select_action_array = [self.a_size, self.inc_bound]
                    else:
                        param_branch = tf.layers.dense(param_branch, self.action_width[0], activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
                        #param_branch = tf.layers.dense(param_branch, 5, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-3))
                        self.select_action_array = [self.a_size, self.action_width[0]]
                        #self.select_action_array = [self.node_size, 5]

                    policy_branches = [select_branch, param_branch]

                    self.all_log_probs = tf.concat(policy_branches, axis=1, name="action_probs")

                    output_p, normalized_logits_p = self.create_discrete_action_masking_layer(param_branch, [self.select_action_array[1]])

                    normalized_logits = tf.concat([normalized_logits_s, normalized_logits_p], axis=-1)

                    self.policy = tf.identity(tf.concat([output_s, output_p], -1), name="action")
                    action_idx = [0] + list(np.cumsum(self.select_action_array,dtype=np.int32))
                    branches_logits = [normalized_logits[:, action_idx[i]:action_idx[i + 1]] for i in range(len(self.select_action_array))]
                    mode_output = tf.concat([tf.expand_dims(tf.argmax(bl, axis=1),axis=1) for bl in branches_logits], axis=1)
                    self.mode_policy = tf.identity(mode_output, name='mode_action')

            with tf.compat.v1.variable_scope('value'):
                value = tf.layers.dense(self.rnn, self.h_size, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                value = tf.layers.dense(value, 1, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                self.value = tf.identity(value, name="value_estimate")

            if self.use_gs_estimator:
                if self.action_type == 'discrete' and self.use_varout:
                    with tf.compat.v1.variable_scope('gs_estimator'):
                        #gnnout = tf.concat([self.netvec, self.global_state, self.context_vector_for_input, pa_oh], axis=-1)
                        gnnout = tf.concat([self.netvec, self.global_state], axis=-1)
                        gnnout = tf.stop_gradient(gnnout)
                        gnnout = tf.concat([gnnout, link_feature_m, self.global_state_msg], axis=-1)
                        self.is_rollout = tf.compat.v1.placeholder(tf.bool, ())
                        gseinput_roll = tf.cast(self.policy, tf.float32)
                        self.gseinput_ph = tf.compat.v1.placeholder(tf.float32, [None, len(policy_branches)])

                        self.gigi = gseinput = tf.compat.v1.cond(self.is_rollout, lambda: gseinput_roll, lambda: self.gseinput_ph)
                        gseinput = tf.stop_gradient(tf.reshape(gseinput, [-1, self.bst, gseinput.get_shape()[-1]]))
                        self.gioh = gseinput = tf.squeeze(tf.one_hot(tf.cast(gseinput, tf.int32), self.action_width[0], dtype=tf.float32), -2)
                        gseinput = tf.concat([gnnout, gseinput], axis=-1)
                        gseinput = tf.layers.dense(gseinput, self.h_size, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))

                        gs_estimate,self.gs_rnn_state = tf.nn.dynamic_rnn(\
                            inputs=gseinput,cell=gs_lstm,dtype=tf.float32,initial_state=gs_state_in,scope='gs_rnn')

                        self.gs_estimate_global = tf.layers.dense(gs_estimate, 128, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                        #self.gs_estimate = tf.layers.dense(self.gs_estimate, self.sg_unit, activation=None)
                        self.gs_estimate_mu = tf.layers.dense(self.gs_estimate_global, self.global_state.get_shape()[-1], use_bias=False, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                        self.gs_estimate_sigma = 0.001 + (1 - 0.001) * tf.nn.sigmoid(tf.math.log(tf.layers.dense(self.gs_estimate_global, self.global_state.get_shape()[-1], use_bias=False, activation=tf.nn.softplus, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))))
                
                        self.gs_estimate_node = tf.layers.dense(gs_estimate, 128, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                        self.gs_estimate_mu_node = tf.layers.dense(self.gs_estimate_node, self.senrec.get_shape()[-1], use_bias=False, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                        self.gs_estimate_sigma_node = 0.001 + (1 - 0.001) * tf.nn.sigmoid(tf.math.log(tf.layers.dense(self.gs_estimate_node, self.senrec.get_shape()[-1], use_bias=False, activation=tf.nn.softplus, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))))

                        self.gs_estimate_link = tf.layers.dense(gs_estimate, 128, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                        self.gs_estimate_mu_link = tf.layers.dense(self.gs_estimate_link, self.linklink.get_shape()[-1], use_bias=False, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                        self.gs_estimate_sigma_link = 0.001 + (1 - 0.001) * tf.nn.sigmoid(tf.math.log(tf.layers.dense(self.gs_estimate_link, self.linklink.get_shape()[-1], use_bias=False, activation=tf.nn.softplus, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))))


                        self.gs_estimate_mu_reshape = tf.reshape(self.gs_estimate_mu, [-1,self.bst,self.gs_estimate_mu.get_shape()[-1]])
                        self.gs_estimate_sigma_reshape = tf.reshape(self.gs_estimate_sigma, [-1,self.bst,self.gs_estimate_sigma.get_shape()[-1]])
                        gsest_dist = tfp.distributions.Normal(self.gs_estimate_mu_reshape, self.gs_estimate_sigma_reshape)

                        self.gs_estimate_mu_node_reshape = tf.reshape(self.gs_estimate_mu_node, [-1,self.bst,self.gs_estimate_mu_node.get_shape()[-1]])
                        self.gs_estimate_sigma_node_reshape = tf.reshape(self.gs_estimate_sigma_node, [-1,self.bst,self.gs_estimate_sigma_node.get_shape()[-1]])
                        gsest_node_dist = tfp.distributions.Normal(self.gs_estimate_mu_node_reshape, self.gs_estimate_sigma_node_reshape)

                        self.gs_estimate_mu_link_reshape = tf.reshape(self.gs_estimate_mu_link, [-1,self.bst,self.gs_estimate_mu_link.get_shape()[-1]])
                        self.gs_estimate_sigma_link_reshape = tf.reshape(self.gs_estimate_sigma_link, [-1,self.bst,self.gs_estimate_sigma_link.get_shape()[-1]])
                        gsest_link_dist = tfp.distributions.Normal(self.gs_estimate_mu_link_reshape, self.gs_estimate_sigma_link_reshape)


                        next_gf = gsest_dist.sample()
                        next_nf = gsest_node_dist.sample()
                        next_lf = gsest_link_dist.sample()
                        next_f = tf.concat([next_nf, next_lf, next_gf], axis=-1)
                        next_f = tf.stop_gradient(next_f)
                        next_f = tf.concat([next_f, link_feature_m, self.global_state_msg], axis=-1)
                        self.gs_estimate_next_a = tf.layers.dense(next_f, 128, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                        self.next_a = tf.layers.dense(self.gs_estimate_next_a, self.action_width[0], activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))

                        self.gs_estimate = tf.concat([self.gs_estimate_node, self.gs_estimate_link, self.gs_estimate_global], axis=-1)

            if not self.use_varout:
                self.actpred = tf.layers.dense(self.netvec, self.prevAveract.get_shape()[-1], activation=None, use_bias=False)

            if self.action_type == 'select':
                self.action_holder = tf.placeholder(shape=[None, len(self.select_action_array)], dtype=tf.float32, name="action_holder")
            elif self.use_varout:
                self.action_holder = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32, name="action_holder")
            else:
                self.action_holder = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name="action_holder")
            if parameter_dict['use_curiosity']:
                with tf.compat.v1.variable_scope('curiosity'):
                    self.create_curiosity_module()    
 
            if scope != 'global':           
    
                if self.action_type == 'discrete':
                    self.action_holder = tf.compat.v1.placeholder(shape=[None, len(policy_branches)], dtype=tf.int32, name="action_holder")
                    if self.use_varout:
                        self.action_oh = tf.concat([tf.one_hot(self.action_holder[:, i], self.action_width[i]) for i in range(1)], axis=1)
                        self.selected_actions = tf.stop_gradient(self.action_oh)

                        self.all_old_log_probs = tf.compat.v1.placeholder(shape=[None, self.action_width[0]], dtype=tf.float32, name='old_probabilities')
                        _, old_normalized_logits = self.create_discrete_action_masking_layer(self.all_old_log_probs, [self.action_width[0]])

                        action_idx = [0] + list(np.cumsum(self.action_width[0],dtype=np.int32))

                        self.entropy_each = tf.stack([
                            tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=tf.nn.softmax(self.all_log_probs[:, action_idx[i]:action_idx[i + 1]]),
                                logits=self.all_log_probs[:, action_idx[i]:action_idx[i + 1]])
                            for i in range(1)], axis=1)
                        self.entropy = tf.reduce_sum(self.entropy_each, axis=1)
                        self.entropy = tf.reduce_mean(self.entropy)

                        self.log_probs = tf.reduce_sum((tf.stack([
                            -tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=self.action_oh[:, action_idx[i]:action_idx[i + 1]],
                                logits=normalized_logits[:, action_idx[i]:action_idx[i + 1]]
                            )
                            for i in range(1)], axis=1)), axis=1, keepdims=True)

                        self.old_log_probs = tf.reduce_sum((tf.stack([
                            -tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=self.action_oh[:, action_idx[i]:action_idx[i + 1]],
                                logits=old_normalized_logits[:, action_idx[i]:action_idx[i + 1]]
                            )
                            for i in range(1)], axis=1)), axis=1, keepdims=True)

                        self.target_v = tf.compat.v1.placeholder(shape=[None],dtype=tf.float32)
                        self.advantages = tf.compat.v1.placeholder(shape=[None,1],dtype=tf.float32)

                        self.old_value = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32, name='old_value_estimates')

        #                decay_epsilon = tf.train.polynomial_decay(0.2, global_step, 10000, 0.1, power=1.0)
        #                decay_beta = tf.train.polynomial_decay(5e-3, global_step, 10000, 1e-5, power=1.0)
                        decay_epsilon = 0.2
                        decay_beta = 2e-3 #tf.train.cosine_decay_restarts(0.01, global_step, 200, t_mul=1, alpha=0.1)

                        maskA = tf.zeros([1,tf.cast(self.bst/2,tf.int32)])
                        maskB = tf.ones([1, tf.cast(self.bst/2,tf.int32)])
                        mask = tf.concat([maskA, maskB], axis=1)
                        clipped_value_estimate = self.old_value + tf.clip_by_value(tf.reduce_sum(self.value, axis=1) - self.old_value,
                                                                           - decay_epsilon, decay_epsilon)

                        v_opt_a = tf.math.squared_difference(self.target_v, tf.reduce_sum(self.value, axis=1))
                        v_opt_b = tf.math.squared_difference(self.target_v, clipped_value_estimate)
                        v_opt = tf.reshape(tf.maximum(v_opt_a, v_opt_b), [-1, self.bst])
                        v_opt = tf.reshape(v_opt, [-1])
                        #v_opt = tf.reshape(v_opt * mask, [-1])
                        self.value_loss = tf.reduce_mean(v_opt)

                        r_theta = tf.exp(self.log_probs - self.old_log_probs)
                        p_opt_a = r_theta * self.advantages
                        p_opt_b = tf.clip_by_value(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantages
                        p_opt = tf.reshape(tf.minimum(p_opt_a, p_opt_b), [-1, self.bst, 1])
                        p_opt = tf.reshape(p_opt, [-1, 1])
                        #p_opt = tf.reshape(p_opt * mask[:,:,tf.newaxis], [-1, 1])
                        self.policy_loss = -tf.reduce_mean(p_opt)
                        #self.actpred_loss = tf.reduce_mean(tf.abs(self.actpred - self.prevAveract))
                        self.actpred_loss = 0

                        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * decay_beta + tf.compat.v1.losses.get_regularization_loss()

                        if self.use_context:
                            if self.probabilistic_context:
                                prior = tfp.distributions.Normal(tf.zeros(self.h_size//2), tf.ones(self.h_size//2))
                                target_kl = tfp.distributions.kl_divergence(self.context_vector_target_dist, prior)
                                #context_kl = tfp.distributions.kl_divergence(self.context_vector_dist, prior)
                                context_kl = tfp.distributions.kl_divergence(self.context_vector_dist, self.context_vector_target_dist)
                                target_kl = target_kl[:,-1,:]
                                context_kl = context_kl[:,-1,:]
                                self.context_loss = (0.01 * tf.reduce_mean(target_kl) + 0.01 * tf.reduce_mean(context_kl)) / (parameter_dict['num_epoch'] * (parameter_dict['buffer_ready_size'] / parameter_dict['update_batch_size']))
                                #self.context_loss = 0.01 * tf.reduce_mean(context_kl)
                                self.loss += self.context_loss
                            else:
                                self.context_loss = tf.reduce_mean(tf.squared_difference(self.context_vector[:,-1,:], self.context_vector_target[:,-1,:]))
                                #self.context_loss = tf.reduce_mean(tf.squared_difference(self.context_vector, self.context_vector_target))
                                self.loss += ((0.01 * self.context_loss) / (parameter_dict['num_epoch'] * (parameter_dict['buffer_ready_size'] / parameter_dict['update_batch_size'])))

                        if self.use_context_v2:
                            with tf.compat.v1.variable_scope('context'):
                                self.dummy_param = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])
                                if self.averact_context_target:
                                    prevAveract_s = tf.gather(self.prevAveract, self.prevAction.senders)
                                    prevAveract_r = tf.gather(self.prevAveract, self.prevAction.receivers)
                                    prevAveract = tf.concat([prevAveract_s, prevAveract_r], -1)
                                    prevAveract = tf.gather(prevAveract, self.graph_iidx) #[batch*link*time,depth]
                                    averact_context_target = tf.reshape(prevAveract, [-1, self.bst, prevAveract.get_shape()[-1]]) #[batch*link,time,depth]

                                if self.stay_prev:
                                    dummy_param = tf.one_hot(tf.cast(self.dummy_param, tf.int32),self.action_width_real[0]-1)
                                else:
                                    dummy_param = tf.one_hot(tf.cast(self.dummy_param, tf.int32),self.action_width_real[0])
                                dummy_param = tf.tile(dummy_param, [1,self.bst,1])

                                celoss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=dummy_param, logits=self.context_logit)
                                '''
                                maskA = tf.zeros([1,tf.cast(self.bst/2,tf.int32)])
                                maskB = tf.ones([1, tf.cast(self.bst/2,tf.int32)])
                                mask = tf.concat([maskA, maskB], axis=1)
                                celoss = celoss * mask
                                '''
                                #celoss = tf.keras.losses.KLD(dummy_param, tf.nn.softmax(self.context_logit))
                                self.context_loss = (tf.reduce_mean(celoss)) + tf.compat.v1.losses.get_regularization_loss(scope+'/context')

                                if self.averact_context_target:
                                    closs = tf.math.squared_difference(averact_context_target, self.context_logit_aa)
                                    '''
                                    maskA = tf.zeros([1,tf.cast(self.bst/2,tf.int32)])
                                    maskB = tf.ones([1, tf.cast(self.bst/2,tf.int32)])
                                    mask = tf.concat([maskA, maskB], axis=1)
                                    closs = closs * mask[:,:,tf.newaxis]
                                    '''
                                    self.context_loss += tf.reduce_mean(closs)

                                '''
                                iidx_param = tf.cast(tf.expand_dims(self.iidx[:,:,1], axis=1), tf.float32)
                                self.ip = iidx_param = tf.tile(iidx_param, [1,self.bst,1])
                                iidxloss = tf.reduce_mean(tf.square(tf.subtract(iidx_param,self.context_iidx_logit)))

                                self.context_loss += (0.1 * iidxloss)
                                '''

                                if self.use_ib:
                                    prior = tfp.distributions.Normal(0.0, 1.0)
                                    kl = tfp.distributions.kl_divergence(self.context_vector_dist, prior)
                                    self.context_loss += (0.01 * tf.reduce_mean(kl))

                                #self.loss += self.context_loss

                        sup_target = tf.reshape(dummy_param, [-1,dummy_param.get_shape()[-1]])
                        supceloss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=sup_target, logits=self.pb[0])
                        self.sup_loss = tf.reduce_mean(supceloss) + self.value_loss
                        if self.use_gs_estimator:
                            if self.action_type == 'discrete' and self.use_varout:
                                self.gs_target = tf.stop_gradient(self.global_state)
                                self.gs_node_target = tf.stop_gradient(self.senrec)
                                self.gs_link_target = tf.stop_gradient(self.linklink)

                                #$self.gs_estimate_reshape = tf.reshape(self.gs_estimate, [-1,self.bst,self.sg_unit])
                                #self.gs_loss = tf.reduce_mean(tf.math.squared_difference(self.gs_target[:,1:,:], self.gs_estimate_reshape[:,:-1,:]))
                                gsest_dist_s = tfp.distributions.Normal(self.gs_estimate_mu_reshape[:,:-1,:], self.gs_estimate_sigma_reshape[:,:-1,:])
                                gsest_node_dist_s = tfp.distributions.Normal(self.gs_estimate_mu_node_reshape[:,:-1,:], self.gs_estimate_sigma_node_reshape[:,:-1,:])
                                gsest_link_dist_s = tfp.distributions.Normal(self.gs_estimate_mu_link_reshape[:,:-1,:], self.gs_estimate_sigma_link_reshape[:,:-1,:])

                                self.gs_loss = -(tf.reduce_mean(tf.reduce_sum(gsest_dist_s.log_prob(self.gs_target[:,1:,:]), -1)) +
                                        tf.reduce_mean(tf.reduce_sum(gsest_node_dist_s.log_prob(self.gs_node_target[:,1:,:]), -1)) +
                                        tf.reduce_mean(tf.reduce_sum(gsest_link_dist_s.log_prob(self.gs_link_target[:,1:,:]), -1)))

                                self.action_oh_time = tf.reshape(self.action_oh, [-1, self.bst, self.action_oh.get_shape()[-1]])
                                gsce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.action_oh_time[:,1:,:], logits=self.next_a[:,:-1,:])
                                self.gs_loss += tf.reduce_mean(gsce)

                                #self.loss += self.gs_loss
                                #self.sup_loss += self.gs_loss
                                self.context_loss += (self.gs_loss + tf.compat.v1.losses.get_regularization_loss(scope+'/gs_estimator'))

                        self.sup_target = sup_target
                        self.dummy_param_reshape = dummy_param

                        local_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
                        local_context_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope+'/context')
                        local_gse_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope+'/gs_estimator')
                        local_msg_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope+'/message')
                        local_context_vars.extend(local_gse_vars)
                        local_value_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope+'/value')
                        local_vars = [lv for lv in local_vars if lv not in local_context_vars]
                        local_context_vars.extend(local_msg_vars)
                        #local_sup_vars = [lv for lv in local_vars if lv not in local_value_vars]
                        self.gradients = tf.gradients(self.loss,local_vars)
                        self.gradients_sup = tf.gradients(self.sup_loss,local_vars)
                        self.gradients_context = tf.gradients(self.context_loss,local_context_vars)

                    else:
                        self.action_oh = tf.concat([tf.one_hot(self.action_holder[:, i], self.action_width[i]) for i in range(len(self.action_width))], axis=1)
                        self.selected_actions = tf.stop_gradient(self.action_oh)

                        self.all_old_log_probs = tf.placeholder(shape=[None, sum(self.action_width)], dtype=tf.float32, name='old_probabilities')
                        _, old_normalized_logits = self.create_discrete_action_masking_layer(self.all_old_log_probs, self.action_width)

                        action_idx = [0] + list(np.cumsum(self.action_width,dtype=np.int32))

                        self.entropy_each = tf.stack([
                            tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=tf.nn.softmax(self.all_log_probs[:, action_idx[i]:action_idx[i + 1]]),
                                logits=self.all_log_probs[:, action_idx[i]:action_idx[i + 1]])
                            for i in range(len(self.action_width))], axis=1)
                        self.entropy = tf.reduce_sum(self.entropy_each, axis=1)
                        self.entropy = tf.reduce_mean(self.entropy)

                        self.log_probs = tf.reduce_sum((tf.stack([
                            -tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=self.action_oh[:, action_idx[i]:action_idx[i + 1]],
                                logits=normalized_logits[:, action_idx[i]:action_idx[i + 1]]
                            )
                            for i in range(len(self.action_width))], axis=1)), axis=1, keepdims=True)

                        self.old_log_probs = tf.reduce_sum((tf.stack([
                            -tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=self.action_oh[:, action_idx[i]:action_idx[i + 1]],
                                logits=old_normalized_logits[:, action_idx[i]:action_idx[i + 1]]
                            )
                            for i in range(len(self.action_width))], axis=1)), axis=1, keepdims=True)

                        self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                        self.advantages = tf.placeholder(shape=[None,1],dtype=tf.float32)

                        self.old_value = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value_estimates')

        #                decay_epsilon = tf.train.polynomial_decay(0.2, global_step, 10000, 0.1, power=1.0)
        #                decay_beta = tf.train.polynomial_decay(5e-3, global_step, 10000, 1e-5, power=1.0)
                        decay_epsilon = 0.2
                        decay_beta = 1e-5 #tf.train.cosine_decay_restarts(0.01, global_step, 200, t_mul=1, alpha=0.1)

                        clipped_value_estimate = self.old_value + tf.clip_by_value(tf.reduce_sum(self.value, axis=1) - self.old_value,
                                                                           - decay_epsilon, decay_epsilon)

                        v_opt_a = tf.squared_difference(self.target_v, tf.reduce_sum(self.value, axis=1))
                        v_opt_b = tf.squared_difference(self.target_v, clipped_value_estimate)
                        self.value_loss = tf.reduce_mean(tf.maximum(v_opt_a, v_opt_b))

                        r_theta = tf.exp(self.log_probs - self.old_log_probs)
                        p_opt_a = r_theta * self.advantages
                        p_opt_b = tf.clip_by_value(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantages
                        self.policy_loss = -tf.reduce_mean(tf.minimum(p_opt_a, p_opt_b))

                        self.actpred_loss = tf.reduce_mean(tf.abs(self.actpred - self.prevAveract))

                        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * decay_beta + 0.05 * self.actpred_loss
     
                        local_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
                        self.gradients = tf.gradients(self.loss,local_vars)
    #                    self.gradients, self.grad_norms = tf.clip_by_global_norm(tf.gradients(self.loss,local_vars), 40.0)

    #                    global_vars = tf.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'global')
    #                    self.apply_grads = trainer.apply_gradients(zip(self.gradients,global_vars),global_step=global_step)

                elif self.action_type == 'continuous':
                    self.all_old_log_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name='old_probabilities')

                    self.entropy_each = self.sample_dist.entropy()
                    self.entropy = tf.reduce_mean(self.entropy_each)

                    self.log_probs = tf.reduce_sum((self.sample_dist.log_prob(self.action_holder)), axis=1, keepdims=True)
                    self.old_log_probs = tf.reduce_sum((tf.identity(self.all_old_log_probs)), axis=1, keepdims=True)

                    self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                    self.advantages = tf.placeholder(shape=[None,1],dtype=tf.float32)

                    self.old_value = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value_estimates')

#                    decay_epsilon = tf.train.polynomial_decay(0.2, global_step, 10000, 0.1, power=1.0)
#                    decay_beta = tf.train.polynomial_decay(5e-3, global_step, 10000, 1e-5, power=1.0)
                    decay_epsilon = 0.2
                    decay_beta = 0.01 #tf.train.cosine_decay_restarts(0.1, global_step, 200, t_mul=1, alpha=0.1)

                    clipped_value_estimate = self.old_value + tf.clip_by_value(tf.reduce_sum(self.value, axis=1) - self.old_value,
                                                                   - decay_epsilon, decay_epsilon)

                    v_opt_a = tf.squared_difference(self.target_v, tf.reduce_sum(self.value, axis=1))
                    v_opt_b = tf.squared_difference(self.target_v, clipped_value_estimate)
                    self.value_loss = tf.reduce_mean(tf.maximum(v_opt_a, v_opt_b))

                    r_theta = tf.exp(self.log_probs - self.old_log_probs)
                    p_opt_a = r_theta * self.advantages
                    p_opt_b = tf.clip_by_value(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantages
                    self.policy_loss = -tf.reduce_mean(tf.minimum(p_opt_a, p_opt_b))

                    self.loss = 0.5 * self.value_loss + self.policy_loss# - self.entropy * decay_beta
                    if parameter_dict['use_curiosity']:
                        self.loss += 10*(0.2 * self.forward_loss + 0.8 * self.inverse_loss)

                    local_vars = tf.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
                    self.gradients = tf.gradients(self.loss,local_vars)
#                    self.gradients, self.grad_norms = tf.clip_by_global_norm(tf.gradients(self.loss,local_vars), 40.0)

                elif self.action_type == 'select':
                    self.action_holder = tf.placeholder(shape=[None, len(policy_branches)], dtype=tf.int32, name="action_holder")
                    self.action_oh = tf.concat([tf.one_hot(self.action_holder[:, i], self.select_action_array[i]) for i in range(len(self.select_action_array))], axis=1)
                    self.selected_actions = tf.stop_gradient(self.action_oh)

                    self.all_old_log_probs = tf.placeholder(shape=[None, sum(self.select_action_array)], dtype=tf.float32, name='old_probabilities')
                    _, old_normalized_logits = self.create_discrete_action_masking_layer(self.all_old_log_probs, self.select_action_array)

                    action_idx = [0] + list(np.cumsum(self.select_action_array,dtype=np.int32))

                    self.entropy_each = tf.stack([
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=tf.nn.softmax(self.all_log_probs[:, action_idx[i]:action_idx[i + 1]]),
                            logits=self.all_log_probs[:, action_idx[i]:action_idx[i + 1]]) / np.sqrt(self.select_action_array[i])
                        for i in range(len(self.select_action_array))], axis=1)
                    self.entropy = tf.reduce_sum(self.entropy_each, axis=1)
                    self.entropy = tf.reduce_mean(self.entropy)

                    self.log_probs = tf.reduce_sum((tf.stack([
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.action_oh[:, action_idx[i]:action_idx[i + 1]],
                            logits=normalized_logits[:, action_idx[i]:action_idx[i + 1]]
                        )
                        for i in range(len(self.select_action_array))], axis=1)), axis=1, keepdims=True)

                    self.old_log_probs = tf.reduce_sum((tf.stack([
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.action_oh[:, action_idx[i]:action_idx[i + 1]],
                            logits=old_normalized_logits[:, action_idx[i]:action_idx[i + 1]]
                        )
                        for i in range(len(self.select_action_array))], axis=1)), axis=1, keepdims=True)

                    self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                    self.advantages = tf.placeholder(shape=[None,1],dtype=tf.float32)

                    self.old_value = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value_estimates')

    #                decay_epsilon = tf.train.polynomial_decay(0.2, global_step, 10000, 0.1, power=1.0)
    #                decay_beta = tf.train.polynomial_decay(5e-3, global_step, 10000, 1e-5, power=1.0)
                    decay_epsilon = 0.2
                    decay_beta = 1e-2 #tf.train.cosine_decay_restarts(0.01, global_step, 200, t_mul=1, alpha=0.1)

                    clipped_value_estimate = self.old_value + tf.clip_by_value(tf.reduce_sum(self.value, axis=1) - self.old_value,
                                                                       - decay_epsilon, decay_epsilon)

                    v_opt_a = tf.squared_difference(self.target_v, tf.reduce_sum(self.value, axis=1))
                    v_opt_b = tf.squared_difference(self.target_v, clipped_value_estimate)
                    self.value_loss = tf.reduce_mean(tf.maximum(v_opt_a, v_opt_b))

                    r_theta = tf.exp(self.log_probs - self.old_log_probs)
                    p_opt_a = r_theta * self.advantages
                    p_opt_b = tf.clip_by_value(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantages
                    self.policy_loss = -tf.reduce_mean(tf.minimum(p_opt_a, p_opt_b))

                    self.actpred_loss = tf.reduce_mean(tf.abs(self.actpred - self.prevAveract))

                    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * decay_beta + 0.05 * self.actpred_loss + tf.losses.get_regularization_loss()

                    if self.use_context:
                        if self.probabilistic_context:
                            prior = tfp.distributions.Normal(tf.zeros(self.h_size//2), tf.ones(self.h_size//2))
                            target_kl = tfp.distributions.kl_divergence(self.context_vector_target_dist, prior)
                            #context_kl = tfp.distributions.kl_divergence(self.context_vector_dist, prior)
                            context_kl = tfp.distributions.kl_divergence(self.context_vector_dist, self.context_vector_target_dist)
                            target_kl = target_kl[:,-1,:]
                            context_kl = context_kl[:,-1,:]
                            self.context_loss = (0.01 * tf.reduce_mean(target_kl) + 0.01 * tf.reduce_mean(context_kl)) / (parameter_dict['num_epoch'] * (parameter_dict['buffer_ready_size'] / parameter_dict['update_batch_size']))
                            #self.context_loss = 0.01 * tf.reduce_mean(context_kl)
                            self.loss += self.context_loss
                        else:
                            self.context_loss = tf.reduce_mean(tf.squared_difference(self.context_vector[:,-1,:], self.context_vector_target[:,-1,:]))
                            self.loss += ((0.1 * self.context_loss) / (parameter_dict['num_epoch'] * (parameter_dict['buffer_ready_size'] / parameter_dict['update_batch_size'])))


                    local_vars = tf.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
                    self.gradients = tf.gradients(self.loss,local_vars)

                self.gradients, self.grad_norms = tf.clip_by_global_norm(self.gradients, 10.0)
                self.gradients_sup, _ = tf.clip_by_global_norm(self.gradients_sup, 10.0)
                self.gradients_context, _ = tf.clip_by_global_norm(self.gradients_context, 10.0)

            else:
                pl = []
                vl = []
                al = []
                cl = []
                sl = []
                ent = []
                grads = []
                grads_sup = []
                grads_context = []
                for i in range(num_workers):
                    pl.append(self.worker_lists[i].local_AC.policy_loss)
                    vl.append(self.worker_lists[i].local_AC.value_loss)
                    al.append(self.worker_lists[i].local_AC.gs_loss)
                    if self.use_context or self.use_context_v2:
                        cl.append(self.worker_lists[i].local_AC.context_loss)
                    sl.append(self.worker_lists[i].local_AC.sup_loss)
                    ent.append(self.worker_lists[i].local_AC.entropy)
                    grads.append(self.worker_lists[i].local_AC.gradients)
                    grads_sup.append(self.worker_lists[i].local_AC.gradients_sup)
                    grads_context.append(self.worker_lists[i].local_AC.gradients_context)
                self.policy_loss = tf.reduce_mean(pl)
                self.value_loss = tf.reduce_mean(vl)
                self.actpred_loss = tf.reduce_mean(al)
                self.context_loss = tf.constant(0)
                if self.use_context or self.use_context_v2:
                    self.context_loss = tf.reduce_mean(cl)
                self.sup_loss = tf.reduce_mean(sl)
                self.entropy = tf.reduce_mean(ent)
                averaged_gradients = self.average_gradients(grads)
                averaged_gradients_sup = self.average_gradients(grads_sup)
                averaged_gradients_context = self.average_gradients(grads_context)
                self.ag = averaged_gradients
                global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'global')
                global_vars_context = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'global/context')
                global_vars_gse = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'global/gs_estimator')
                global_vars_msg = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'global/message')
                global_vars_context.extend(global_vars_gse)
                global_value_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'global/value')
                global_vars = [gv for gv in global_vars if gv not in global_vars_context]
                global_vars_context.extend(global_vars_msg)
                #global_sup_vars = [gv for gv in global_vars if gv not in global_value_vars]
                self.apply_grads = trainer.apply_gradients(zip(averaged_gradients,global_vars),global_step=global_step)
                self.apply_grads_sup = self.trainer_pre.apply_gradients(zip(averaged_gradients_sup,global_vars),global_step=global_step)
                self.apply_grads_context = self.trainer_context.apply_gradients(zip(averaged_gradients_context,global_vars_context),global_step=global_step)

    def _make_semi_graph(self, x, hiddendim, sg_layer, reuse=False):
        with tf.compat.v1.variable_scope('sg{}'.format(sg_layer)):
            out = []
            for ns in range(self.node_size):
                temp = [x[ns]]
                ancestor = np.nonzero(self.adj_mat[ns+self.num_input_node,:])[0]
                for ac in ancestor:
                    if (ac-self.num_input_node) != ns and ac >= self.num_input_node:
                        temp.append(x[ac-self.num_input_node])
                temp = tf.concat(temp, axis=1)
                with tf.compat.v1.variable_scope('node{}'.format(ns)):
                    temp = tf.layers.dense(temp, hiddendim, activation=tf.nn.tanh, use_bias=False, reuse=reuse, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6))
                out.append(temp)

        return out

    def _make_gru_semi_graph(self, x, hiddendim, sg_layer, reuse=False):
        with tf.compat.v1.variable_scope('gru'):
            grus = []
            grustates = []
            for ns in range(self.node_size):
                cell = tf.contrib.rnn.GRUBlockCellV2(num_units=hiddendim)
                grus.append(cell)
                grustates.append(cell.zero_state(batch_size=self.bs, dtype=tf.float32))

            ins = x
            for ll in range(sg_layer):
                out = []
                outstate = []
                with tf.compat.v1.variable_scope('sg{}'.format(ll), reuse=reuse):
                    for ns in range(self.node_size):
                        temp = [ins[ns]]
                        ancestor = np.nonzero(self.adj_mat[ns+self.num_input_node,:])[0]
                        for ac in ancestor:
                            if (ac-self.num_input_node) != ns and ac >= self.num_input_node:
                                temp.append(ins[ac-self.num_input_node])
                        temp = tf.concat(temp, axis=1)

                        with tf.compat.v1.variable_scope('node{}'.format(ns)):
                            o, s = grus[ns](temp, grustates[ns])
                        out.append(o)
                        outstate.append(s)

                ins = out
                grustates = outstate

        return out

    def _make_autoregressive_output(self, rnn):
        actlstm_hsize = 32
        lstm_cell = tf.contrib.rnn.LSTMBlockCell(actlstm_hsize)
        lstm_state = lstm_cell.zero_state(batch_size=self.bs, dtype=tf.float32)

        policy_branches = []

        #initial_act = tf.zeros((self.bs, actlstm_hsize), dtype=tf.float32)
        initial_act = tf.zeros((self.bs, self.action_width[0]), dtype=tf.float32)
        out = tf.concat([rnn, initial_act], axis=-1)
        states = lstm_state
        for asize in self.action_width:
            out_, states = lstm_cell(out, states)
            policy_branches.append(tf.layers.dense(out_, asize, activation=None, use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)))
            out = tf.concat([rnn, policy_branches[-1]], axis=-1)

        return policy_branches

    def _calc_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def _attention(self, x, attn_unit, num_heads):
        x_shape = x.get_shape()
        def split_heads(xx):
            xx = tf.reshape(xx, (-1, x_shape[1], num_heads, attn_unit // num_heads))
            return tf.transpose(xx, perm=[0, 2, 1, 3])

        q = tf.layers.dense(x, attn_unit)   #[batch, #node, attn_unit]
        k = tf.layers.dense(x, attn_unit)
        v = tf.layers.dense(x, attn_unit)

        q = split_heads(q)  #[batch, num_heads, #node, depth]
        k = split_heads(k)
        v = split_heads(v)

        scaled_attention, self.attention_weights = self._calc_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])    #[batch, #node, num_heads, depth]
        concat_attention = tf.reshape(scaled_attention, (-1, x_shape[1], attn_unit))    #[batch, #node, attn_unit]

        output = tf.layers.dense(concat_attention, x_shape[-1])
        output = x + output

        ffn_output = tf.layers.dense(output, attn_unit, activation=tf.nn.relu)
        ffn_output = tf.layers.dense(ffn_output, x_shape[-1])
        output = ffn_output + output
        return output

    def average_gradients(self, grads):
        average_grads = []
        for grad_and_vars in zip(*grads):
            gra = []
            for g in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                gra.append(expanded_g)
            grad = tf.concat(gra, axis=0)
            grad = tf.reduce_mean(grad, 0)
            average_grads.append(grad)
        return average_grads

    @staticmethod
    def create_discrete_action_masking_layer(all_logits, action_size):
        """
        Creates a masking layer for the discrete actions
        :param all_logits: The concatenated unnormalized action probabilities for all branches
        :param action_masks: The mask for the logits. Must be of dimension [None x total_number_of_action]
        :param action_size: A list containing the number of possible actions for each branch
        :return: The action output dimension [batch_size, num_branches] and the concatenated normalized logits
        """
        action_idx = [0] + list(np.cumsum(action_size,dtype=np.int32))
        branches_logits = [all_logits[:, action_idx[i]:action_idx[i + 1]] for i in range(len(action_size))]
        raw_probs = [tf.nn.softmax(branches_logits[k]) + 1.0e-10
                     for k in range(len(action_size))]
        normalized_probs = [
            tf.divide(raw_probs[k], tf.reduce_sum(raw_probs[k] + 1.0e-10, axis=1, keepdims=True))
                            for k in range(len(action_size))]
        output = tf.concat([tf.random.categorical(tf.math.log(normalized_probs[k]), 1) for k in range(len(action_size))], axis=1)
        return output, tf.concat([tf.math.log(normalized_probs[k]) for k in range(len(action_size))], axis=1)

    def create_curiosity_module(self):
        if self.input_type == 'conv':
            self.nextAction = tf.placeholder(shape=[None,self.node_size+self.num_input_node,self.node_size+self.num_input_node+1,1],dtype=tf.float32)
            self.nextReward = tf.placeholder(shape=[None,None,1],dtype=tf.float32)
        elif self.input_type == 'semi_graph' or self.input_type == 'gru_semi_graph' or self.input_type == 'attention':
            self.nextAction = []
            for ns in range(self.node_size):
                self.nextAction.append(tf.placeholder(tf.float32, [None, np.count_nonzero(self.adj_mat[ns+self.num_input_node,:])+1]))
            self.nextReward = tf.placeholder(shape=[None,None,1],dtype=tf.float32)
        
        #prevStateEncodeInput = tf.reshape(tf.concat([self.prevAction,self.prevReward],axis=2), shape=[-1,self.a_size+1])
        #nextStateEncodeInput = tf.reshape(tf.concat([self.nextAction,self.nextReward],axis=2), shape=[-1,self.a_size+1])
        encode_current_state = self.create_encoder(self.prevAction, self.prevReward, self.curiosity_encode_size, 2, "curiosity_encoder", False)
        encode_next_state = self.create_encoder(self.nextAction, self.nextReward, self.curiosity_encode_size, 2, "curiosity_encoder", True)
        
        self.create_inverse_model(encode_current_state, encode_next_state)
        self.create_forward_model(encode_current_state, encode_next_state)

    def create_encoder(self, inputs, reward, h_size, num_layers, scope, reuse):
        with tf.compat.v1.variable_scope(scope):
            hidden = inputs
            if self.input_type == 'conv':
                #for i in range(num_layers):
                    #hidden = tf.layers.conv2d(hidden, int(16*(i+1)), (4,4), activation=tf.nn.relu, padding='SAME',
                    #                          name="conv_{}".format(i), reuse=reuse)
                    #hidden = tf.layers.max_pooling2d(hidden, (2,2), (2,2), padding='SAME')
                #hidden = tf.layers.conv2d(hidden, 16, kernel_size=(4,4), activation=tf.nn.relu,
                #                                 name="conv_1", reuse=reuse)
                hidden = tf.keras.layers.LocallyConnected2D(16, (4,4), activation=tf.nn.relu, implementation=1)(hidden)
                hidden = tf.layers.average_pooling2d(hidden, (2,2), (2,2), padding='SAME')
                #hidden = tf.layers.max_pooling2d(hidden, (2,2), (2,2), padding='SAME')
                #hidden = tf.layers.conv2d(hidden, 32, kernel_size=(2,2), activation=tf.nn.relu,
                #                                 name="conv_2", reuse=reuse)
                hidden = tf.keras.layers.LocallyConnected2D(32, (2,2), activation=tf.nn.relu, implementation=1)(hidden)
                hidden = tf.layers.average_pooling2d(hidden, (2,2), (2,2), padding='SAME')
                #hidden = tf.layers.max_pooling2d(hidden, (2,2), (2,2), padding='SAME')

                hidden = tf.reshape(hidden, [-1, int(np.prod(hidden.get_shape().as_list()[1:]))])
            elif self.input_type == 'semi_graph':
                for sgl in range(3):
                    hidden = self._make_semi_graph(hidden, 8, sgl, reuse)
                hidden = tf.concat(hidden, axis=1)
                #hidden = tf.concat([hidden, tf.reshape(reward, shape=[-1,1])], axis=1)

            for i in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, activation=tf.nn.relu, reuse=reuse,
                                     name="hidden_{}".format(i),
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'))
        return hidden
    
    def create_inverse_model(self, encode_current_state, encode_next_state):
        self.cuclass = tf.placeholder(shape=[None,2],dtype=tf.float32)

        combined_input = tf.concat([encode_current_state, encode_next_state], axis=1)
        hidden = tf.layers.dense(combined_input, 256, activation=tf.nn.relu)
        
        #pred_action = tf.concat(
        #    [tf.layers.dense(hidden, i, activation=tf.nn.softmax)
        #     for i in self.action_width], axis=1)
        #cross_entropy = tf.reduce_sum(-tf.log(pred_action + 1e-10) * self.selected_actions, axis=1)

        pred_action = tf.layers.dense(hidden, self.a_size, activation=None)
        distance = tf.squared_difference(pred_action, self.action_holder)
        self.inverse_loss = tf.reduce_mean(distance)

        #pred_cuclass = tf.layers.dense(hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer())
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_cuclass, labels=self.cuclass)

        #self.inverse_loss = tf.reduce_mean(cross_entropy)

    
    def create_forward_model(self, encode_current_state, encode_next_state):
        combined_input = tf.concat([encode_current_state, self.action_holder], axis=1)
        hidden = tf.layers.dense(combined_input, 256, activation=tf.nn.relu)
        # We compare against the concatenation of all observation streams, hence `self.vis_obs_size + int(self.vec_obs_size > 0)`.
        pred_next_state = tf.layers.dense(hidden, self.curiosity_encode_size, activation=None)

        squared_difference = 0.5 * tf.reduce_sum(tf.squared_difference(pred_next_state, encode_next_state), axis=1)
        self.intrinsic_reward = tf.clip_by_value(self.curiosity_strength * squared_difference, 0, 1)
        self.forward_loss = tf.reduce_mean(squared_difference)        

    def noisy_dense(self, x, size, name, bias=True, activation_fn=tf.identity):
        # From https://github.com/wenh123/NoisyNet-DQN/blob/master/tf_util.py
        # the function used in eq.10,11
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        # Initializer of \mu and \sigma 
        # Sample noise from gaussian. Factorised gaussian 
#        mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),     
#                                                    maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
#        sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))
#        p = tf.random.normal(shape=[x.get_shape().as_list()[1], 1])
#        q = tf.random.normal(shape=[1, size])
#        f_p = f(p); f_q = f(q)
#        w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)
        # Independent gaussian
        mu_init = tf.random_uniform_initializer(minval=-1*np.power(3/x.get_shape().as_list()[1], 0.5),     
                                                    maxval=1*np.power(3/x.get_shape().as_list()[1], 0.5))
        sigma_init = tf.constant_initializer(0.017)
        #sigma_init = tf.constant_initializer(0.0002)
        w_epsilon = tf.random.normal(shape=[x.get_shape().as_list()[1], size])
        b_epsilon = tf.random.normal(shape=[size])
    
        # w = w_mu + w_sigma*w_epsilon
        w_mu = tf.get_variable(name + "/w_mu", [x.get_shape()[1], size], initializer=mu_init, regularizer=tf.contrib.layers.l2_regularizer(5e-3))
        #w_mu = tf.get_variable(name + "/w_mu", [x.get_shape()[1], size], initializer=mu_init)
        w_sigma = tf.get_variable(name + "/w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
        w = w_mu + tf.multiply(w_sigma, w_epsilon)
        ret = tf.matmul(x, w)
        if bias:
            # b = b_mu + b_sigma*b_epsilon
            b_mu = tf.get_variable(name + "/b_mu", [size], initializer=mu_init)
            b_sigma = tf.get_variable(name + "/b_sigma", [size], initializer=sigma_init)
            b = b_mu + tf.multiply(b_sigma, b_epsilon)
            return activation_fn(ret + b)
        else:
            return activation_fn(ret)


class Master():
    def __init__(self,env,parameter_dict,master_network,trainer,global_step,memory_dict,workers,global_epi):
        self.env = env
        self.master_network = master_network
        self.a_size = parameter_dict['a_size']
        self.h_size = parameter_dict['h_size']
        self.max_ep = parameter_dict['max_ep']
        self.reset_step = parameter_dict['reset_step']
        self.adj_mat = parameter_dict['adj_mat']
        self.trainer = trainer
        self.global_step = global_step
        self.memory_dict = memory_dict
        self.workers = workers
        self.global_epi = global_epi
        self.buffer_size = parameter_dict['buffer_size']
        self.buffer_ready_size = parameter_dict['buffer_ready_size']
        self.update_batch_size = parameter_dict['update_batch_size']
        self.num_epoch = parameter_dict['num_epoch']
        self.input_type = parameter_dict['input_type']
        self.action_type = parameter_dict['action_type']
        self.node_size = parameter_dict['node_size']
        self.num_input_node = parameter_dict['num_input_node']
        self.incremental_action = parameter_dict['incremental_action']
        self.stay_prev = parameter_dict['stay_prev']
        self.inc_bound = parameter_dict['inc_bound']
        self.use_context = parameter_dict['use_context']
        self.use_context_v2 = parameter_dict['use_context_v2']
        self.use_gs_estimator = parameter_dict['use_gs_estimator']
        self.use_message = parameter_dict['use_message']
        self.use_varout = parameter_dict['use_varout']

        self.context_update_period = self.buffer_size // self.buffer_ready_size
        self.update_counter = 0

        self.summary_writer = tf.compat.v1.summary.FileWriter(parameter_dict['model_path']+"/train_master")

        link_size, node_size = self.env.get_num_params()
        weightmat = self.adj_mat[self.env.num_input_node:,:]
        param_split = np.append(0,np.cumsum((weightmat!=0).sum(1)))
        idx_house = []
        for nn in range(self.node_size):
            #idx_house.append(np.append(np.arange(param_split[nn],param_split[nn+1]),link_size+nn))
            idx_house.append(np.arange(param_split[nn],param_split[nn+1]))
        self.idx_house = np.array(idx_house)

        scale = 5
        link_size, node_size = self.env.get_num_params()
        aw, ac = self.env.get_link_scale(level=3, scale=scale)
        #self.action_width = np.concatenate((aw, np.zeros(node_size) + scale*2))
        #self.action_center = np.concatenate((ac, np.zeros(node_size) + scale))
        self.action_width = aw
        self.action_center = ac
        if parameter_dict['action_type'] == 'discrete' or parameter_dict['action_type'] == 'select':
            self.action_width += 1
            if parameter_dict['stay_prev']:
                self.action_width += 1


        if self.incremental_action:
            self.action_width = np.zeros_like(self.action_width) + self.inc_bound

    def check(self, sess, coord, TRAIN_EVENT, COLLECT_EVENT, pretrain):
        while not coord.should_stop():
            starttime = time()
            TRAIN_EVENT.wait()
            self.train(sess, pretrain)
            COLLECT_EVENT.set()
            TRAIN_EVENT.clear()
#            print('One train time ', time()-starttime)

    def train(self, sess, pretrain):
        value_total, policy_total, actpred_total, context_total, ent_total, sup_total = [], [], [], [], [], []
        wn = self.memory_dict.keys()
        tbs = []
        for n in wn:
            tbs.append(self.memory_dict[n].pop())

        #state_train = [np.zeros([self.update_batch_size,self.h_size]),np.zeros([self.update_batch_size,self.h_size])]
        if self.use_context:
            context_state_train = [np.zeros([self.update_batch_size,self.h_size]),np.zeros([self.update_batch_size,self.h_size])]

        self.update_counter +=1 
        if pretrain.value == 1:
            cur_num_epoch = 1
        else:
            cur_num_epoch = self.num_epoch
        for k in range(cur_num_epoch):
            for tb in tbs:
                if pretrain.value == 1:
                    tb.shuffle_for_context()
                else:
                    tb.shuffle()
#                print(self.global_epi.value, len(tb.buffer['prevAction']))
            if pretrain.value == 1:
                sample_size = len(tbs[0].buffer['prevAction'])
            else:
                sample_size = self.buffer_ready_size
            for l in range(sample_size // self.update_batch_size):
                feed_dict = {}
                for i, tb in enumerate(tbs):
                    start = l * self.update_batch_size
                    end = (l + 1) * self.update_batch_size
                    mini_batch = tb.make_mini_batch(start,end)

                    self.max_ep = mini_batch['prevAction'][0].shape[0]
                    if self.use_varout:
                        #state_train = [[np.zeros([self.update_batch_size*self.a_size,self.h_size]),np.zeros([self.update_batch_size*self.a_size,self.h_size])]] * 2
                        total_asize = np.sum([pas.shape[1] for pas in mini_batch['prevAction']])
                        state_train = [[np.zeros([total_asize,self.h_size]),np.zeros([total_asize,self.h_size])]] * 2
                        if self.use_context_v2:
                            #context_state_train = [np.zeros([self.update_batch_size*self.a_size,self.h_size]),np.zeros([self.update_batch_size*self.a_size,self.h_size])] 
                            context_state_train = [np.zeros([total_asize,self.h_size]),np.zeros([total_asize,self.h_size])] 
                        if self.use_gs_estimator:
                            gs_state_train = [np.zeros([total_asize,int(self.h_size/2)]),np.zeros([total_asize,int(self.h_size/2)])] 
                    else:
                        state_train = [[np.zeros([self.update_batch_size,self.h_size]),np.zeros([self.update_batch_size,self.h_size])]] * 2

                    #feed_dict[self.workers[i].local_AC.prevAveract] = mini_batch['prevAveract'].reshape([-1,self.max_ep,self.node_size+self.num_input_node])
                    if self.action_type == 'select':
                        feed_dict[self.workers[i].local_AC.prevSelect] = mini_batch['prevSelect'].reshape([-1,self.max_ep,1])
                        feed_dict[self.workers[i].local_AC.prevNewval] = mini_batch['prevNewval'].reshape([-1,self.max_ep,1])
                    feed_dict[self.workers[i].local_AC.target_v] = np.concatenate(mini_batch['discounted_rewards'])
                    feed_dict[self.workers[i].local_AC.advantages] = np.concatenate(mini_batch['advantages']).reshape([-1, 1])
                    if self.input_type == 'fc':
                        feed_dict[self.workers[i].local_AC.prevAction] = mini_batch['prevAction'].reshape([-1,self.max_ep,self.a_size])
                        if self.use_context:
                            feed_dict[self.workers[i].local_AC.dummy_prevAction] = mini_batch['dummyParam'].reshape([-1,self.max_ep,self.a_size])
                        feed_dict[self.workers[i].local_AC.bs] = mini_batch['prevAction'].shape[0] * mini_batch['prevAction'].shape[1]
                        if self.use_varout:
                            feed_dict[self.workers[i].local_AC.zpsize] = self.update_batch_size
                    elif self.input_type == 'conv':
                        netwidth = self.node_size+self.env.num_input_node
                        netheight = self.node_size+self.env.num_input_node + 1
                        feed_dict[self.workers[i].local_AC.prevAction] = mini_batch['prevAction'].reshape([-1,netwidth,netheight,1])
                    else:
                        if self.use_context:
                            dobs = mini_batch['dummyParam'].reshape([-1,self.max_ep,self.a_size])
                        if self.use_context_v2:
                            feed_dict[self.workers[i].local_AC.dummy_param] = np.concatenate([_[0] for _ in mini_batch['dummyParam']]).reshape([-1,1])
                        if self.input_type != 'graph_net':
                            obs = mini_batch['prevAction'].reshape([-1,self.max_ep,self.a_size])
                            for ns in range(self.node_size):
                                feed_dict[self.workers[i].local_AC.prevAction[ns]] = obs[:,:,self.idx_house[ns]].reshape([-1,len(self.idx_house[ns])])
                                if self.use_context:
                                    feed_dict[self.workers[i].local_AC.dummy_prevAction[ns]] = dobs[:,:,self.idx_house[ns]].reshape([-1,len(self.idx_house[ns])])
                                if self.input_type == 'gru_semi_graph' or self.input_type == 'attention':
                                    feed_dict[self.workers[i].local_AC.bs] = mini_batch['prevAction'].shape[0] * mini_batch['prevAction'].shape[1]
                                    if self.use_varout:
                                        feed_dict[self.workers[i].local_AC.zpsize] = self.update_batch_size
                        else:
                            obs = np.concatenate(mini_batch['prevAction_graph']).reshape([-1])
                            obp = utils_tf.get_feed_dict(self.workers[i].local_AC.prevAction, utils_np.data_dicts_to_graphs_tuple(obs))
                            feed_dict.update(obp)
                            if self.use_message:
                                obsm = np.concatenate(mini_batch['message']).reshape([-1])
                                obpm = utils_tf.get_feed_dict(self.workers[i].local_AC.prevMsg, utils_np.data_dicts_to_graphs_tuple(obsm))
                                feed_dict.update(obpm)
                            graph_iidx = []
                            for gi in range(self.update_batch_size):
                                paramsize = mini_batch['prevAction'][gi].shape[-1]
                                gidx = np.arange(0,self.max_ep*paramsize, paramsize).reshape([1,-1])
                                gidx = np.tile(gidx, [paramsize, 1]) + np.arange(0,paramsize).reshape([-1,1])
                                if gi > 0:
                                    gidx += np.sum([_.size for _ in graph_iidx])
                                graph_iidx.append(gidx)
                            graph_iidx = np.concatenate(graph_iidx, axis=0).reshape([-1])
                            feed_dict[self.workers[i].local_AC.graph_iidx] = graph_iidx
                        feed_dict[self.workers[i].local_AC.bst] = self.max_ep

                    if self.use_varout:
                        listidx = mini_batch['dummyListIdx'].copy()
                        for bb in range(1, self.update_batch_size):
                            listidx[bb][:,0] += bb
                        feed_dict[self.workers[i].local_AC.iidx] = np.concatenate(listidx, axis=0).reshape([-1,2,2])

                    feed_dict[self.workers[i].local_AC.prevReward] = np.concatenate(mini_batch['prevReward']).reshape([-1,self.max_ep,1])
                    feed_dict[self.workers[i].local_AC.prevReward2] = np.concatenate(mini_batch['prevReward2'], axis=0).reshape([-1,self.max_ep,3])
                    if self.action_type == 'select':
                        feed_dict[self.workers[i].local_AC.action_holder] =mini_batch['action_holder'].reshape([-1,2])
                    elif self.use_varout:
                        feed_dict[self.workers[i].local_AC.action_holder] =np.concatenate(mini_batch['action_holder']).reshape([-1,1])
                        if self.stay_prev:
                            par = mini_batch['action_holder'].reshape([-1,self.max_ep,1])[:,:-1,:]
                            papre = np.zeros([par.shape[0],1,par.shape[2]]) - 1
                            par = np.concatenate([papre,par], axis=1)
                        else:
                            par = np.concatenate([_.T for _ in mini_batch['prevAction']], axis=0).reshape([-1, self.max_ep, 1])
                            for se in range(self.max_ep // self.reset_step):
                                par[:,int(se*self.reset_step),:] = -1

                        feed_dict[self.workers[i].local_AC.prevAction_raw] = par
                    else:
                        feed_dict[self.workers[i].local_AC.action_holder] =mini_batch['action_holder'].reshape([-1,self.a_size])
                    if self.action_type == 'discrete':
                        if self.use_varout:
                            feed_dict[self.workers[i].local_AC.all_old_log_probs] = np.concatenate(mini_batch['action_probs'], axis=0).reshape([-1,self.action_width[0].astype(np.int)])
                        else:
                            feed_dict[self.workers[i].local_AC.all_old_log_probs] = mini_batch['action_probs'].reshape([-1,self.action_width.sum(dtype=np.int)])
                    if self.action_type == 'continuous':
                        feed_dict[self.workers[i].local_AC.all_old_log_probs] = mini_batch['action_probs'].reshape([-1,self.a_size])
                    if self.action_type == 'select':
                        feed_dict[self.workers[i].local_AC.all_old_log_probs] = mini_batch['action_probs'].reshape([-1,int(self.a_size+self.action_width[0])])
                        #feed_dict[self.workers[i].local_AC.all_old_log_probs] = mini_batch['action_probs'].reshape([-1,int(self.node_size+5)])
                    feed_dict[self.workers[i].local_AC.old_value] = np.concatenate(mini_batch['value_estimates']).flatten()
                    if self.use_context or self.use_context_v2:
                        feed_dict[self.workers[i].local_AC.context_state_in[0]] = context_state_train[0]
                        feed_dict[self.workers[i].local_AC.context_state_in[1]] = context_state_train[1]
                    if self.use_gs_estimator:
                        feed_dict[self.workers[i].local_AC.gs_state_in[0]] = gs_state_train[0]
                        feed_dict[self.workers[i].local_AC.gs_state_in[1]] = gs_state_train[1]
                        feed_dict[self.workers[i].local_AC.gs_est] = np.concatenate(mini_batch['gs_state'], axis=0)
                        feed_dict[self.workers[i].local_AC.gseinput_ph] = np.concatenate(mini_batch['action_holder'], axis=0).reshape([-1,1])
                    #feed_dict[self.workers[i].local_AC.state_in[0]] = state_train[0]
                    #feed_dict[self.workers[i].local_AC.state_in[1]] = state_train[1]
                    feed_dict[self.workers[i].local_AC.state_in[0][0]] = state_train[0][0]
                    feed_dict[self.workers[i].local_AC.state_in[0][1]] = state_train[0][1]
                    feed_dict[self.workers[i].local_AC.state_in[1][0]] = state_train[1][0]
                    feed_dict[self.workers[i].local_AC.state_in[1][1]] = state_train[1][1]
                    feed_dict[self.workers[i].local_AC.is_rollout] = False


                if pretrain.value == 1:
                    sl, ent, _ = sess.run([self.master_network.sup_loss, self.master_network.entropy,
                              self.master_network.apply_grads_sup], feed_dict=feed_dict)
                    sup_total.append(sl)
                    value_total.append(0)
                    policy_total.append(0)
                    ent_total.append(ent)
                else:
                    pl, vl, ent, _ = sess.run([self.master_network.policy_loss, self.master_network.value_loss, self.master_network.entropy,
                              self.master_network.apply_grads], feed_dict=feed_dict)
                    value_total.append(vl)
                    policy_total.append(np.abs(pl))
                    ent_total.append(ent)
                for w in self.workers:
                    sess.run(w.update_local_ops)

        if self.update_counter % self.context_update_period == 0:
            self.update_counter = 0
            if pretrain.value == 1:
                cur_num_epoch_ = self.num_epoch
            else:
                cur_num_epoch_ = 1
            for k in range(cur_num_epoch_):
                for tb in tbs:
                    tb.shuffle_for_context()
#                print(self.global_epi.value, len(tb.buffer['prevAction']))
                for l in range(len(tbs[0].buffer['prevAction']) // self.update_batch_size):
                    feed_dict = {}
                    for i, tb in enumerate(tbs):
                        start = l * self.update_batch_size
                        end = (l + 1) * self.update_batch_size
                        mini_batch = tb.make_mini_batch(start,end)

                        self.max_ep = mini_batch['prevAction'][0].shape[0]
                        if self.use_varout:
                            total_asize = np.sum([pas.shape[1] for pas in mini_batch['prevAction']])
                            state_train = [[np.zeros([total_asize,self.h_size]),np.zeros([total_asize,self.h_size])]] * 2
                            if self.use_context_v2:
                                context_state_train = [np.zeros([total_asize,self.h_size]),np.zeros([total_asize,self.h_size])] 
                            if self.use_gs_estimator:
                                gs_state_train = [np.zeros([total_asize,int(self.h_size/2)]),np.zeros([total_asize,int(self.h_size/2)])]
                        else:
                            state_train = [[np.zeros([self.update_batch_size,self.h_size]),np.zeros([self.update_batch_size,self.h_size])]] * 2

                        #feed_dict[self.workers[i].local_AC.prevAveract] = mini_batch['prevAveract'].reshape([-1,self.max_ep,self.node_size+self.num_input_node])
                        if self.input_type == 'fc':
                            feed_dict[self.workers[i].local_AC.prevAction] = mini_batch['prevAction'].reshape([-1,self.max_ep,self.a_size])
                            if self.use_context:
                                feed_dict[self.workers[i].local_AC.dummy_prevAction] = mini_batch['dummyParam'].reshape([-1,self.max_ep,self.a_size])
                            feed_dict[self.workers[i].local_AC.bs] = mini_batch['prevAction'].shape[0] * mini_batch['prevAction'].shape[1]
                            if self.use_varout:
                                feed_dict[self.workers[i].local_AC.zpsize] = self.update_batch_size
                        else:
                            if self.use_context:
                                dobs = mini_batch['dummyParam'].reshape([-1,self.max_ep,self.a_size])
                            if self.use_context_v2:
                                feed_dict[self.workers[i].local_AC.dummy_param] = np.concatenate([_[0] for _ in mini_batch['dummyParam']]).reshape([-1,1])
                                feed_dict[self.workers[i].local_AC.prevAveract] = np.concatenate([_.reshape([-1,1]) for _ in mini_batch['prevAveract']], axis=0)
                            if self.input_type != 'graph_net':
                                obs = mini_batch['prevAction'].reshape([-1,self.max_ep,self.a_size])
                                for ns in range(self.node_size):
                                    feed_dict[self.workers[i].local_AC.prevAction[ns]] = obs[:,:,self.idx_house[ns]].reshape([-1,len(self.idx_house[ns])])
                                    if self.use_context:
                                        feed_dict[self.workers[i].local_AC.dummy_prevAction[ns]] = dobs[:,:,self.idx_house[ns]].reshape([-1,len(self.idx_house[ns])])
                                    if self.input_type == 'gru_semi_graph' or self.input_type == 'attention':
                                        feed_dict[self.workers[i].local_AC.bs] = mini_batch['prevAction'].shape[0] * mini_batch['prevAction'].shape[1]
                                        if self.use_varout:
                                            feed_dict[self.workers[i].local_AC.zpsize] = self.update_batch_size
                            else:
                                obs = np.concatenate(mini_batch['prevAction_graph']).reshape([-1])
                                obp = utils_tf.get_feed_dict(self.workers[i].local_AC.prevAction, utils_np.data_dicts_to_graphs_tuple(obs))
                                feed_dict.update(obp)
                                if self.use_message:
                                    obsm = np.concatenate(mini_batch['message']).reshape([-1])
                                    obpm = utils_tf.get_feed_dict(self.workers[i].local_AC.prevMsg, utils_np.data_dicts_to_graphs_tuple(obsm))
                                    feed_dict.update(obpm)
                                graph_iidx = []
                                for gi in range(self.update_batch_size):
                                    paramsize = mini_batch['prevAction'][gi].shape[-1]
                                    gidx = np.arange(0,self.max_ep*paramsize, paramsize).reshape([1,-1])
                                    gidx = np.tile(gidx, [paramsize, 1]) + np.arange(0,paramsize).reshape([-1,1])
                                    if gi > 0:
                                        gidx += np.sum([_.size for _ in graph_iidx])
                                    graph_iidx.append(gidx)
                                graph_iidx = np.concatenate(graph_iidx, axis=0).reshape([-1])
                                feed_dict[self.workers[i].local_AC.graph_iidx] = graph_iidx

                            feed_dict[self.workers[i].local_AC.bst] = self.max_ep

                        if self.use_varout:
                            listidx = mini_batch['dummyListIdx'].copy()
                            for bb in range(1, self.update_batch_size):
                                listidx[bb][:,0] += bb
                            feed_dict[self.workers[i].local_AC.iidx] = np.concatenate(listidx, axis=0).reshape([-1,2,2])

                        feed_dict[self.workers[i].local_AC.prevReward] = np.concatenate(mini_batch['prevReward']).reshape([-1,self.max_ep,1])
                        feed_dict[self.workers[i].local_AC.prevReward2] = np.concatenate(mini_batch['prevReward2'], axis=0).reshape([-1,self.max_ep,3])
                        if self.action_type == 'select':
                            feed_dict[self.workers[i].local_AC.action_holder] =mini_batch['action_holder'].reshape([-1,2])
                        elif self.use_varout:
                            feed_dict[self.workers[i].local_AC.action_holder] =np.concatenate(mini_batch['action_holder']).reshape([-1,1])
                            if self.stay_prev:
                                par = mini_batch['action_holder'].reshape([-1,self.max_ep,1])[:,:-1,:]
                                papre = np.zeros([par.shape[0],1,par.shape[2]]) - 1
                                par = np.concatenate([papre,par], axis=1)
                            else:
                                par = np.concatenate([_.T for _ in mini_batch['prevAction']], axis=0).reshape([-1, self.max_ep, 1])

                                for se in range(self.max_ep // self.reset_step):
                                    par[:,int(se*self.reset_step),:] = -1

                            feed_dict[self.workers[i].local_AC.prevAction_raw] = par
                        else:
                            feed_dict[self.workers[i].local_AC.action_holder] =mini_batch['action_holder'].reshape([-1,self.a_size])
                        if self.use_context or self.use_context_v2:
                            feed_dict[self.workers[i].local_AC.context_state_in[0]] = context_state_train[0]
                            feed_dict[self.workers[i].local_AC.context_state_in[1]] = context_state_train[1]
                        if self.use_gs_estimator:
                            feed_dict[self.workers[i].local_AC.gs_state_in[0]] = gs_state_train[0]
                            feed_dict[self.workers[i].local_AC.gs_state_in[1]] = gs_state_train[1]
                            feed_dict[self.workers[i].local_AC.gs_est] = np.concatenate(mini_batch['gs_state'], axis=0)
                            feed_dict[self.workers[i].local_AC.gseinput_ph] = np.concatenate(mini_batch['action_holder'], axis=0).reshape([-1,1])
                        feed_dict[self.workers[i].local_AC.state_in[0][0]] = state_train[0][0]
                        feed_dict[self.workers[i].local_AC.state_in[0][1]] = state_train[0][1]
                        feed_dict[self.workers[i].local_AC.state_in[1][0]] = state_train[1][0]
                        feed_dict[self.workers[i].local_AC.state_in[1][1]] = state_train[1][1]
                        feed_dict[self.workers[i].local_AC.is_rollout] = False

                    cl, al, _ = sess.run([self.master_network.context_loss, self.master_network.actpred_loss,
                              self.master_network.apply_grads_context], feed_dict=feed_dict)
                    context_total.append(cl)
                    actpred_total.append(al)

                    for w in self.workers:
                        sess.run(w.update_local_ops)

        summary = tf.compat.v1.Summary()
        if pretrain.value == 1:
            summary.value.add(tag='Losses/Sup Loss', simple_value=float(np.mean(sup_total)))
        summary.value.add(tag='Losses/Value Loss', simple_value=float(np.mean(value_total)))
        summary.value.add(tag='Losses/Policy Loss', simple_value=float(np.mean(policy_total)))
        summary.value.add(tag='Losses/Entropy', simple_value=float(np.mean(ent_total)))
        if len(context_total) > 0:
            summary.value.add(tag='Losses/Context_Loss', simple_value=float(np.mean(context_total)))
            summary.value.add(tag='Losses/gs Loss', simple_value=float(np.mean(actpred_total)))
        self.summary_writer.add_summary(summary, int(self.global_epi.value))

        self.summary_writer.flush()

      
class Worker():
    def __init__(self,name,parameter_dict,epsilon,trainer,num_workers,global_episodes,global_step,memory_dict,barrier,global_epi):
        self.name = "worker_" + str(name)
        self.number = name
        self.use_randsign = parameter_dict['use_randsign']
        scale = 5
        self.env = test_prob.simul(128, mb_size=parameter_dict['mb_size'], max_task_num=parameter_dict['max_task_num'], use_randsign=self.use_randsign,scale=scale)
        self.val_env = test_prob_val.simul(128, mb_size=parameter_dict['mb_size'], max_task_num=parameter_dict['max_task_num'], use_randsign=self.use_randsign,scale=scale)
        self.model_path = parameter_dict['model_path']
        self.a_size = parameter_dict['a_size']
        self.h_size = parameter_dict['h_size']
        self.node_size = parameter_dict['node_size']
        self.input_type = parameter_dict['input_type']
        self.action_type = parameter_dict['action_type']
        self.memory_dict = memory_dict
        self.adj_mat = parameter_dict['adj_mat']
        self.epsilon = epsilon
        self.pretrain_epi = parameter_dict['pretrain_epi']
        self.max_ep = parameter_dict['max_ep']
        self.reset_step = parameter_dict['reset_step']
        self.use_curiosity = parameter_dict['use_curiosity']
        self.use_update_noise = parameter_dict['use_update_noise']
        self.use_attention = parameter_dict['use_attention']
        self.use_context = parameter_dict['use_context']
        self.use_context_v2 = parameter_dict['use_context_v2']
        self.use_gs_estimator= parameter_dict['use_gs_estimator']
        self.use_message= parameter_dict['use_message']
        self.message_dim= parameter_dict['message_dim']
        self.use_varout = parameter_dict['use_varout']
        self.incremental_action = parameter_dict['incremental_action']
        self.stay_prev = parameter_dict['stay_prev']
        self.inc_bound = parameter_dict['inc_bound']
        starter_learning_rate = parameter_dict['start_lr']
#        self.learning_rate = learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                               64*1000, 0.99, staircase=False)
#        learning_rate = tf.train.cosine_decay_restarts(starter_learning_rate, global_step,
#                                               200, t_mul=1, alpha=0.1)
        self.trainer = trainer
        self.num_workers = num_workers
        #self.mb_size = parameter_dict['mb_size']
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.total_reward = []
        self.total_target_reward = []
        self.total_spear = []
        self.total_intrinsic = []
        self.initial_reward = -np.inf
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.model_path+"/train_"+str(self.number))
        
        self.barrier = barrier
        self.global_epi = global_epi

        self.training_buffer = experience_buffer(buffer_size=parameter_dict['buffer_size'], buffer_ready_size=parameter_dict['buffer_ready_size'], isselect=self.action_type=='select', iscontext=(self.use_context or self.use_context_v2), isgraphnet=self.input_type=='graph_net', isgsest=self.use_gs_estimator, ismessage=self.use_message)
        self.buffer_ready_size = parameter_dict['buffer_ready_size']
        self.update_batch_size = parameter_dict['update_batch_size']
        self.num_epoch = parameter_dict['num_epoch']

        link_size, node_size = self.env.get_num_params()
        aw, ac = self.env.get_link_scale(level=3, scale=scale)
        #self.action_width = np.concatenate((aw, np.zeros(node_size) + scale*2))
        #self.action_center = np.concatenate((ac, np.zeros(node_size) + scale))
        self.action_width = aw
        self.action_center = ac
        #aw, ac, bw, bc = self.env.get_link_scale(level=3, scale=scale, basal_scale=3)
        ##self.action_width = np.concatenate((aw, bw))
        ##self.action_center = np.concatenate((ac, bc))
        #self.action_width = np.concatenate((aw, np.zeros(0) + 0*2))
        #self.action_center = np.concatenate((ac, np.zeros(0) + 0))

        if parameter_dict['action_type'] == 'discrete' or parameter_dict['action_type'] == 'select':
            self.action_width += 1
            if parameter_dict['stay_prev']:
                self.action_width += 1
        self.action_width_real = self.action_width

        if self.env.num_sample > 256:
            self.mblimit = 256
        else:
            self.mblimit = self.env.num_sample

        weightmat = self.adj_mat[self.env.num_input_node:,:]
        param_split = np.append(0,np.cumsum((weightmat!=0).sum(1)))
        idx_house = []
        for nn in range(self.node_size):
            #idx_house.append(np.append(np.arange(param_split[nn],param_split[nn+1]),link_size+nn))
            idx_house.append(np.arange(param_split[nn],param_split[nn+1]))
        self.idx_house = np.array(idx_house)

        self.max_task_num = self.env.max_task_num

        self.mi = self.env.zc
        self.rq = self.env.rq
        self.sq = self.env.sq
        '''
        self.eq = self.env.eq
        self.esq = self.env.esq
        self.emq = self.env.emq
        '''
        self.averact = self.env.averact
        self.params = self.env.params
        self.weightmat_link_idx = self.env.weightmat_link_idx
        self.dummy_params = self.env.dummy_params
        self.dummy_link_idx = self.env.dummy_link_idx
        self.dummy_params_num = self.env.dummy_params_num
        '''
        self.dummy_pheno_idx = self.env.dummy_pheno_idx
        self.dummy_pert_num = self.env.dummy_pert_num
        self.dummy_pert_idx = self.env.dummy_pert_idx
        '''
        self.tvt = self.env.tvt
        self.seed = self.env.seed
        self.mb_size = self.env.mb_size
        self.do_reset_initials = self.env.do_reset_initials
        self.do_initialize_task = self.env.do_initialize_task
        self.START_CALC_EVENT = self.env.START_CALC_EVENT
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(parameter_dict,self.action_width,self.name,self.trainer,self.num_workers,global_step)

        if self.incremental_action:
            self.action_width = np.zeros_like(self.action_width) + self.inc_bound

        #self.update_local_ops = update_target_graph('global',self.name)
        self.update_local_ops = None

        if self.use_update_noise:
            self.name_perturb = self.name+'_perturb'
            self.local_AC_perturb = AC_Network(parameter_dict,self.action_width,self.name_perturb,self.trainer,self.num_workers,global_step)

            self.name_adaptive = self.name+'_adaptive'
            self.local_AC_adaptive = AC_Network(parameter_dict,self.action_width,self.name_adaptive,self.trainer,self.num_workers,global_step)

            self.noise_start_scale = parameter_dict['noise_start_scale']
            self.distance_threshold = parameter_dict['distance_threshold']
            self.param_noise_scale = tf.get_variable("param_noise_scale"+"_"+self.name, (), initializer=tf.constant_initializer(self.noise_start_scale), trainable=False)
            self.add_noise_ops = noisy_vars(self.name, self.name_perturb, noise_std=self.param_noise_scale)
            self.add_noise_adaptive_ops = noisy_vars(self.name, self.name_adaptive, noise_std=self.param_noise_scale)
            
            self.policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.local_AC.alpha - self.local_AC_adaptive.alpha))) + \
                                    tf.sqrt(tf.reduce_mean(tf.square(self.local_AC.beta - self.local_AC_adaptive.beta)))

        
    def process_and_add_experience(self, ep_history,gamma,bootstrap_value):
        ep_history = np.array(ep_history)

        '''
        rewards = ep_history[:,3].copy()
        rewards_plus = rewards.tolist()
        rewards_plus.append(bootstrap_value)
        discounted_rewards = discount(np.asarray(rewards_plus),gamma)[:-1]
        discounted_rewards = discounted_rewards.reshape([-1])
        #discounted_rewards = rewards.reshape([-1])
        #if self.use_curiosity:
        #    discounted_rewards +=  ep_history[:,-1].reshape([-1])
        #advantages = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        advantages = discounted_rewards - bootstrap_value.reshape([-1])   #according to Wang.2018.nature.neuroscience
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        '''
#
##        value_plus = np.asarray(ep_history[:,4].tolist() + [bootstrap_value[0]])
##        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
##        advantages = discount(advantages,gamma)
##        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
##        advantages = OI(advantages)
##        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        '''
        # oi
        rewards = ep_history[:,3].copy()
        rewards_plus = rewards.tolist()
        rewards_plus.append(bootstrap_value)
        rewards_plus.insert(0,ep_history[0,1])
        rewoi = OI(np.asarray(rewards_plus))[1:-1].reshape([-1])
        if self.use_curiosity:
            rewoi = rewoi + ep_history[:,-1].reshape([-1])
        rewards = np.asarray(rewoi)
        discounted_rewards = rewards.reshape([-1])
        advantages = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        '''

        '''
        # oi-discount
        rewards = ep_history[:,3].copy()
        rewoi = OI(rewards).reshape([-1])
        rewoi = rewoi.tolist()
        rewoi.append(bootstrap_value)
        rewoi = discount(np.asarray(rewoi), gamma)[:-1]
        rewards = np.asarray(rewoi)
        discounted_rewards = rewards.reshape([-1])
        advantages = discounted_rewards - bootstrap_value.reshape([-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.reshape([-1])
        '''

        # (rew-oi)-discount
        fitness = ep_history[:,3].copy()
        rewhouse = []
        for epi in range(self.max_ep // self.reset_step):
            curfitness = fitness[epi*self.reset_step:(epi+1)*self.reset_step]
            rewoi = OI(ep_history[epi*self.reset_step,1],curfitness).reshape([-1])
            rewards_ = 0.0 * curfitness + 1.0 * rewoi
            #rewards = np.zeros_like(rewards_)
            #rewards[-1] = np.sum(rewards_)[0]
            #rewards_ = rewards.copy()
            if self.use_varout:
                rewards_ = np.repeat(rewards_, ep_history[0][0].size)
            rewards = rewards_.tolist()

            rewhouse.append(rewards)

        rewards = np.concatenate(rewhouse).reshape([-1])
        if self.action_type == 'select':
            same_penalty = ep_history[:,9].reshape([-1]).copy().astype(np.float64)
            rewards += same_penalty
        if self.use_varout:
            rewards = rewards.reshape([-1, ep_history[0][0].size])
            rewards = rewards.tolist()
            rewards.append(bootstrap_value.reshape([-1]))
            discounted_rewards = discount(np.asarray(rewards), gamma)[:-1]
            discounted_rewards = discounted_rewards.T
            #discounted_rewards = discounted_rewards.reshape([-1])
            #advantages = discounted_rewards - bootstrap_value.reshape([-1])

            vv = np.concatenate(ep_history[:,4], axis=1)
            advantages = discounted_rewards - vv
            advantages = (advantages - advantages.mean(axis=1,keepdims=True)) / (advantages.std(axis=1,keepdims=True) + 1e-8)
            advantages = advantages.reshape([-1])
            discounted_rewards = discounted_rewards.reshape([-1])
        else:
            rewards = rewards.tolist()
            rewards.append(bootstrap_value)
            discounted_rewards = discount(np.asarray(rewards), gamma)[:-1]
            discounted_rewards = discounted_rewards.reshape([-1])
            #advantages = discounted_rewards - bootstrap_value.reshape([-1])

            advantages = discounted_rewards - ep_history[:,4].reshape([-1])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.reshape([-1])


        '''
        # diff-discount
        prewards = ep_history[:,1].copy()
        rewards = ep_history[:,3].copy()
        rewoi = np.zeros_like(rewards)
        rewoi[-1] = np.sum(rewards - prewards)
        rewoi = rewoi.tolist()
        rewoi.append(bootstrap_value)
        rewoi = discount(np.asarray(rewoi), gamma)[:-1]
        rewards = np.asarray(rewoi)
        discounted_rewards = rewards.reshape([-1])
        advantages = discounted_rewards - ep_history[:,4].reshape([-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.reshape([-1])
        '''

        '''
        # last reward(OI)-discount
        rewards = ep_history[:,3].copy()
        rewoi = np.zeros_like(rewards)
        rewoi[-1] = np.sum(OI(rewards))
        rewoi = rewoi.tolist()
        rewoi.append(bootstrap_value)
        rewoi = discount(np.asarray(rewoi), gamma)[:-1]
        rewards = np.asarray(rewoi)
        discounted_rewards = rewards.reshape([-1])
        advantages = discounted_rewards - ep_history[:,4].reshape([-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.reshape([-1])
        '''

        '''
        values = np.asarray(ep_history[:,4])
        mb_advs = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(self.max_ep)):
            if t == self.max_ep - 1:
                nextvalues = bootstrap_value
            else:
                nextvalues = values[t+1]
            delta = rewards[t] + gamma * nextvalues - values[t]
            mb_advs[t] = lastgaelam = delta + gamma * 0.97 * lastgaelam
        discounted_rewards = mb_advs + values
        discounted_rewards = discounted_rewards.reshape([-1])
        advantages = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-5)
        advantages = advantages.reshape([-1])
        '''

        processed_exp = {}

        processed_exp['prevAveract'] = np.vstack(ep_history[:,6])
        if self.use_context or self.use_context_v2:
            if self.use_randsign:
                scale = self.action_width[0] // 2
                if self.incremental_action:
                    scale = self.action_width_real[0] // 2
                processed_exp['dummyParam'] = np.vstack(ep_history[:,10]) + scale
                if self.stay_prev:
                    scale = (self.action_width[0] - 1) // 2
                    processed_exp['dummyParam'] = np.vstack(ep_history[:,10]) + scale
            else:
                processed_exp['dummyParam'] = np.vstack(ep_history[:,10])
        if self.use_varout:
            processed_exp['dummyListIdx'] = ep_history[:,11][0]
        processed_exp['prevAction'] = np.vstack(ep_history[:,0])
        if self.input_type == 'graph_net':
            processed_exp['prevAction_graph'] = np.vstack(ep_history[:,13])
        if self.use_gs_estimator:
            gs_state = np.transpose(np.stack(ep_history[:,14]), [1,0,2])
            processed_exp['gs_state'] = np.reshape(gs_state, [-1, gs_state.shape[-1]])
        if self.use_message:
            processed_exp['message'] = np.vstack(ep_history[:,15])
        processed_exp['prevReward'] = np.vstack(ep_history[:,1])
        processed_exp['prevReward2'] = np.vstack(ep_history[:,12])
        processed_exp['action_holder'] = np.hstack(ep_history[:,2]).reshape([-1,1])
        if self.use_varout:
            processed_exp['action_probs'] = np.concatenate([np.expand_dims(ap, axis=1) for ap in ep_history[:,5]], axis=1).reshape([-1,int(self.action_width[0])])
        else:
            processed_exp['action_probs'] = np.vstack(ep_history[:,5])
        processed_exp['value_estimates'] = np.hstack(ep_history[:,4]).reshape([-1,1])
        if self.use_varout:
            processed_exp['discounted_rewards'] = discounted_rewards
        else:
            processed_exp['discounted_rewards'] = np.vstack(discounted_rewards)
        processed_exp['advantages'] = advantages
        if self.action_type == 'select':
            processed_exp['prevSelect'] = np.vstack(ep_history[:,7])
            processed_exp['prevNewval'] = np.vstack(ep_history[:,8])

        self.training_buffer.add(processed_exp)

    def is_ready_update(self):
        size_of_buffer = len(self.training_buffer.buffer['prevAction'])
        return (size_of_buffer > 0) and (size_of_buffer % self.buffer_ready_size == 0)

    def set_update_local_ops(self):
        self.update_local_ops = update_target_graph('global',self.name)

    def make2dinput(self, s):
        netwidth = self.node_size+self.env.num_input_node
        netheight = self.node_size+self.env.num_input_node + 1
        ob = np.zeros((netwidth, netwidth))
        ob[self.env.weightmat!=0] = np.round(s[:-self.node_size].reshape([-1]))
        ob *= self.env.sign_mask
        #weightmat clustermap
        #colidx = np.array([25,  7, 21,  9,  4, 15,  5, 10, 13, 18, 24, 17,  6, 11, 12,  8, 16,
       #29, 23,  3,  0, 28, 27, 26, 20, 22, 14, 19,  1,  2, 30])
        #rowidx = np.array([ 7, 13, 26, 29,  5, 18, 23, 12, 14,  6,  8, 25, 19, 27, 17, 16, 15,
       #21, 11, 22,  9, 20, 28, 24, 10,  4,  3,  2,  0,  1])
        #abs(weightmat) clustermap
        #colidx = np.array([ 7, 21,  9,  4, 15, 25,  5, 10, 14, 19,  1,  2, 24, 17,  6, 11, 12,
       #18, 13, 23,  8, 16, 29,  3,  0, 28, 27, 26, 20, 22, 30])
        #rowidx = np.array([ 7, 13, 23, 12, 14, 25, 27, 22, 11, 16, 17, 15, 21, 19, 24,  9, 20,
       #28, 10,  4,  3,  2,  0,  1,  6,  8, 26, 29,  5, 18])
        bas = np.zeros(self.node_size+self.env.num_input_node)
        bas[self.env.num_input_node:] = np.round(s[-self.node_size:])
        ob = np.concatenate((ob, bas.reshape([-1,1])), axis=1)
        #ob = ob[rowidx,:]
        #ob = ob[:,colidx]
        return ob

    def s_setting(self, s, selector, acttype):
        if acttype == 0:#shift 1 to right
            s[self.idx_house[selector]] = np.roll(s[self.idx_house[selector]], 1)
        elif acttype == 1:#random initialize
            s[self.idx_house[selector]] = np.random.choice(int(self.action_width[0]), self.idx_house[selector].size)
        elif acttype == 2:#flatten
            s[self.idx_house[selector]] = 1
            #s = np.clip(s, 0, self.action_width[0]-1)
        elif acttype == 3:#plus 1
            s[self.idx_house[selector]] += 1
            s = np.clip(s, 0, self.action_width[0]-1)
        elif acttype == 4:#reverse sort
            st = s[self.idx_house[selector]]
            si = np.argsort(st)
            st[si] = st[si[::-1]]
            s[self.idx_house[selector]] = st

        return s


    def run_target_task(self,actiontemp):
        np.random.seed()
        s = np.zeros(self.a_size) + 0.5

        self.do_reset_initials.value = 1

        if self.action_type == 'discrete' or self.action_type == 'select':
            if self.stay_prev:
                s = np.round(s * (self.action_width - 2))
            else:
                s = np.round(s * 10)
            s_scaled = s - 5
            if self.incremental_action:
                s = np.zeros(a_size)
                s_scaled = s.copy()
        elif self.action_type == 'continuous':
            s_raw = s.copy()
            s = np.round(s * self.action_width)
            s_scaled = s -  self.action_center

        #s_scaled = actiontemp.copy().reshape([-1])

        signmat = np.sign(self.adj_mat[self.adj_mat!=0])
        s = np.random.choice(np.arange(1,self.action_center[0]+1), size=signmat.size)
        s_scaled = s * signmat
        s = s_scaled + self.action_center[0]
        self.params[:self.a_size] = s_scaled.reshape([-1])
        self.seed.value = int(self.name[-1])*5000
        self.tvt.value = 0
        self.START_CALC_EVENT.set()
        r = self.rq.get()
        #r -= 0.05 * np.mean(np.round(np.abs(s_scaled)))
        first_r = r
        spear = self.sq.get()
        mui = self.mi.get()
        first_s = spear
        first_mi = mui
        r = r - first_r
        r2 = np.array([spear-first_s, mui-first_mi, 0]).reshape([1,1,-1])
        self.do_reset_initials.value = 0
                    
        if np.isnan(r):
            r = np.array(-100.)
        r = r.reshape([1,1,1])
                
        state = self.local_AC.state_init #Reset the recurrent layer's hidden state
        context_state = self.local_AC.context_state_init #Reset the recurrent layer's hidden state
        if self.use_gs_estimator:
            gs_state = self.local_AC.gs_state_init
            gs_state = [np.tile(z,[self.a_size,1]) for z in gs_state]
            gs_est = np.zeros([int(self.a_size), int(128*3)])
        if self.use_context_v2:
            context_state = [np.tile(z,[self.a_size,1]) for z in context_state]
        if self.use_message:
            msg_nf = np.zeros((self.adj_mat.shape[0], self.message_dim))
            msg_lf = np.zeros((self.a_size, self.message_dim))
            msg_gf = np.zeros((1, self.message_dim))
        local_AC_running = self.local_AC

        cur_link_idx = np.array(self.weightmat_link_idx[:].copy())[:int(self.a_size*2)]
        cur_link_idx = cur_link_idx.ravel().reshape([-1,1])
        cur_link_idx = np.concatenate((np.zeros_like(cur_link_idx), cur_link_idx), axis=1)
        state = [[np.tile(z,[self.a_size,1]), np.tile(x,[self.a_size,1])] for z,x in state]

        er = -np.inf
        his = []
        sp_house = []
        rhis = []
        #r = OI(first_r, np.array([first_r]))[-1]
        for j in range(self.max_ep):
        #for j in range(256):
            feed_dict = {local_AC_running.prevReward:r.reshape([1,1,1]),
                    local_AC_running.prevReward2:r2,
                    local_AC_running.is_rollout:True,
                    #local_AC_running.prevAveract:averact.reshape([1,1,-1]),
                    local_AC_running.iidx:cur_link_idx.reshape([-1,2,2]),
                    #local_AC_running.state_in[0]:state[0], local_AC_running.state_in[1]:state[1]}
                    local_AC_running.state_in[0][0]:state[0][0], local_AC_running.state_in[0][1]:state[0][1],
                    local_AC_running.state_in[1][0]:state[1][0], local_AC_running.state_in[1][1]:state[1][1]}
            if self.use_context or self.use_context_v2:
                feed_dict[local_AC_running.context_state_in[0]] = context_state[0]
                feed_dict[local_AC_running.context_state_in[1]] = context_state[1]
            if self.use_gs_estimator:
                feed_dict[local_AC_running.gs_state_in[0]] = gs_state[0]
                feed_dict[local_AC_running.gs_state_in[1]] = gs_state[1]
                #feed_dict[local_AC_running.gs_est] = np.concatenate(gs_state, -1)
                feed_dict[local_AC_running.gs_est] = gs_est
                feed_dict[local_AC_running.gseinput_ph] = np.array([-1]).reshape([-1,1])

            if self.action_type == 'select':
                if j == 0:
                    a_dist = np.zeros((1,2)) - 1    #make zero vector
                    a_dist = a_dist.astype(np.int)
                feed_dict[local_AC_running.prevSelect] = a_dist[0,0].reshape([1,1,1])
                feed_dict[local_AC_running.prevNewval] = a_dist[0,1].reshape([1,1,1])
            if self.input_type == 'conv':
                ob = self.make2dinput(s)
                feed_dict[local_AC_running.prevAction] = ob.reshape([1,netwidth,netheight,1])
            elif self.input_type == 'fc':
                ob = s
                feed_dict[local_AC_running.prevAction] = ob.reshape([1,1,-1])
                feed_dict[local_AC_running.bs] = 1
            elif self.input_type == 'semi_graph' or self.input_type == 'gru_semi_graph' or self.input_type == 'attention':
                ob = s
                for ns in range(self.node_size):
                    feed_dict[local_AC_running.prevAction[ns]] = ob[self.idx_house[ns]].reshape([1,-1])
                if self.input_type == 'gru_semi_graph' or self.input_type == 'attention':
                    feed_dict[local_AC_running.bs] = 1
                feed_dict[local_AC_running.bst] = 1
            elif self.input_type == 'graph_net':
                ob = s
                obp = [make_graph_dict(ob, self.action_width[0], np.sign(self.adj_mat), np.concatenate((r.ravel(),r2.ravel())))]
                if self.use_message:
                    oblm = np.array([make_msg_graph_dict(ob, self.action_width[0], np.sign(self.adj_mat), msg_nf, msg_lf, msg_gf, np.concatenate((r.ravel(),r2.ravel())))], dtype=np.object)
                obp = utils_tf.get_feed_dict(local_AC_running.prevAction, utils_np.data_dicts_to_graphs_tuple(obp))
                if self.use_message:
                    obpm = utils_tf.get_feed_dict(local_AC_running.prevMsg, utils_np.data_dicts_to_graphs_tuple(oblm))
                    feed_dict.update(obpm)
                feed_dict.update(obp)
                feed_dict[local_AC_running.graph_iidx] = np.arange(s.size)
                feed_dict[local_AC_running.bs] = 1
                feed_dict[local_AC_running.bst] = 1

            if self.use_varout:
                feed_dict[local_AC_running.zpsize] = 1
                if j == 0:
                    feed_dict[local_AC_running.prevAction_raw] = np.zeros_like(ob).reshape([-1,1,1]) - 1
                else:
                    feed_dict[local_AC_running.prevAction_raw] = ob.reshape([-1,1,1])

            #Probabilistically pick an action given our network outputs.
            if self.use_attention or self.input_type=='attention':
                #a_dist, v, state, entropy_each, aw = sess.run([local_AC_running.mode_policy, local_AC_running.value,
                a_dist, v, state, entropy_each, aw = sess.run([local_AC_running.policy, local_AC_running.value,
                                                 local_AC_running.rnn_state, local_AC_running.entropy_each, local_AC_running.attention_weights],
                                          feed_dict=feed_dict)
            else:
                #a_dist, v, state, entropy_each = sess.run([local_AC_running.mode_policy, local_AC_running.value,
                a_dist, v, state, context_state, gs_state_new, gs_est_new, msg_nf_new, msg_lf_new, msg_gf_new, entropy_each = sess.run([local_AC_running.policy, local_AC_running.value,
                                                 local_AC_running.rnn_state, local_AC_running.context_rnn_state, local_AC_running.gs_rnn_state, local_AC_running.gs_estimate, local_AC_running.msg_nf, local_AC_running.msg_lf, local_AC_running.msg_gf, local_AC_running.entropy_each],
                                          feed_dict=feed_dict)

            if self.action_type == 'discrete':
                if self.incremental_action and not self.use_varout:
                    a_inc = a_dist - (self.inc_bound//2)
                    s_ = np.clip(s + a_inc.reshape([-1]), 0, self.action_width-1)
                    a_dist_scaled = s_ - self.action_center
                elif self.use_varout:
                    if self.incremental_action:
                        a_inc = a_dist.ravel() - (self.inc_bound//2)
                        s_ = np.clip(s + a_inc.reshape([-1]), 0, self.action_width_real-1)
                        a_dist_scaled = s_.reshape([-1]) - self.action_center
                    elif self.stay_prev:
                        s_ = s.copy()
                        cidx = a_dist.ravel()!=0
                        s_[cidx] = a_dist.ravel()[cidx] - 1
                        a_dist_scaled = s_.reshape([-1]) - self.action_center
                    else:
                        a_dist_scaled = a_dist.reshape([-1]) - 5
                else:
                    a_dist_scaled = a_dist - self.action_center
            if self.action_type == 'continuous':
                if self.incremental_action:
                    a_inc = a_dist - 0.5
                    s_raw = np.clip(s_raw + a_inc.reshape([-1]), 0, 1)
                    s_ = np.round(s_raw * self.action_width)
                    a_dist_scaled = s_ - self.action_center
                else:
                    a_dist = np.round(a_dist * self.action_width)
                    a_dist_scaled = a_dist - self.action_center
            if self.action_type == 'select':
                s_ = s.copy()
                if self.incremental_action:
                    a_inc = a_dist[0][1] - (self.inc_bound//2)
                    s_[a_dist[0][0]] = np.clip(s[a_dist[0][0]] + a_inc, 0, self.action_width[0]-1)
                else:
                    if s_[a_dist[0][0]] == a_dist[0][1]:
                        same_penalty = -0.01
                    else:
                        same_penalty = 0
                    sp_house.append(same_penalty)
                    s_[a_dist[0][0]] = a_dist[0][1]
                    #s_ = self.s_setting(s_, a_dist[0][0], a_dist[0][1])
                if self.use_randsign:
                    a_dist_scaled = s_.copy() - self.action_center
                else:
                    a_dist_scaled = s_.copy()

            self.params[:self.a_size] = a_dist_scaled.reshape([-1])
            self.seed.value = int(self.name[-1])*5000
            self.tvt.value = 0
            self.START_CALC_EVENT.set()
            r_ = self.rq.get()
            #r_ -= 0.05 * np.mean(np.round(np.abs(a_dist_scaled)))
            spear = self.sq.get()
            mui = self.mi.get()

            if np.isnan(r_):
                r_ = np.array(-100.)
            if np.isnan(spear):
                spear = np.array(0)

            if self.incremental_action or self.action_type == 'select' or (self.action_type == 'discrete' and self.stay_prev):
                s = s_.copy()
            else:
                s = a_dist.copy().reshape([-1])
            #s = np.round(a_dist_scaled.reshape([-1]))
            rhis.append(r_)
            roi = OI(first_r, np.array(rhis))[-1]
            r = r_ - first_r
            r2 = np.array([spear-first_s, mui-first_mi, roi]).reshape([1,1,-1])
            if self.use_gs_estimator:
                gs_state = gs_state_new
                gs_est = gs_est_new.reshape([-1,128*3])
            if self.use_message:
                msg_nf = msg_nf_new
                msg_lf = msg_lf_new
                msg_gf = msg_gf_new
            his.append(r_)


            if er < r_:
                running_action = a_dist_scaled
                er = r_
        #print(aw)

        return first_r, his, running_action, sp_house

    def work(self,gamma,sess,coord,saver,pretrain_saver,total_episodes, TRAIN_EVENT, COLLECT_EVENT, pretrain, s_init=None):
        episode_count = sess.run(self.global_episodes)
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                starttime = time()
                sess.run(self.update_local_ops)
                ep_history = []
                running_reward = []
                running_spear = 0
                running_action = 0
                running_entropy = 0
                running_intrinsic = 0
                
                state = self.local_AC.state_init #Reset the recurrent layer's hidden state
                #state = [[np.tile(z,[self.a_size,1]), np.tile(x,[self.a_size,1])] for z,x in state]
                context_state = self.local_AC.context_state_init #Reset the recurrent layer's hidden state
                if self.use_gs_estimator:
                    gs_state = self.local_AC.gs_state_init
                '''
                if self.use_context_v2:
                    context_state = [np.tile(z,[self.a_size,1]) for z in context_state]
                '''

                local_AC_running = self.local_AC
                
                netwidth = self.node_size+self.env.num_input_node
                netheight = self.node_size+self.env.num_input_node + 1
                er = -np.inf
                actiontemp = []
                first_r = []
                first_s = []
                first_mi = []
                sp_house = []
                for j in range(self.max_ep):
                    if j % self.reset_step == 0:
                        reset_epi = True
                    else:
                        reset_epi = False

                    if reset_epi:
                        if s_init is None:
                            s = np.random.rand(1500)
                            #s = np.zeros(1500) + 0.5
                        else:
                            s = s_init

                        self.do_reset_initials.value = 0
                        self.do_initialize_task.value = 1

                        if self.action_type == 'discrete' or self.action_type == 'select':
                            if self.stay_prev:
                                s = np.round(s * (self.action_width - 2))
                            else:
                                s = np.round(s * (self.action_width[0] - 1))
                            s_scaled = s - self.action_center[0]
                            if self.incremental_action:
                                s = np.zeros(a_size)
                                s_scaled = s.copy()
                        if self.action_type == 'continuous':
                            s_raw = s.copy()
                            s = np.round(s * self.action_width)
                            s_scaled = s - self.action_center

                        self.params[:1500] = s_scaled.reshape([-1])
                        self.seed.value = episode_count+int(self.name[-1])*5000
                        self.tvt.value = 3
                        #self.tvt.value = 0
                        self.START_CALC_EVENT.set()
                        r = self.rq.get()
                        first_r.append(r)
                        spear = self.sq.get()
                        mui = self.mi.get()
                        first_s.append(spear)
                        first_mi.append(mui)
                        r = r - first_r[j//self.reset_step]
                        r2 = np.array([spear-first_s[j//self.reset_step], mui-first_mi[j//self.reset_step], 0]).reshape([1,1,-1])
                        averact = np.array(self.averact[:].copy())
                        self.do_reset_initials.value = 0
                        self.do_initialize_task.value = 0

                        cur_dummy_params_num = int(self.dummy_params_num.value)
                        cur_dummy_params = np.array(self.dummy_params[:].copy())[:cur_dummy_params_num]
                        cur_dummy_link_idx = np.array(self.dummy_link_idx[:].copy())[:int(cur_dummy_params_num*2)]
                        cur_dummy_link_idx = cur_dummy_link_idx.ravel().reshape([-1,1])
                        cur_dummy_link_idx = np.concatenate((np.zeros_like(cur_dummy_link_idx), cur_dummy_link_idx), axis=1)

                        #s = s[:cur_dummy_params_num]
                        s = self.params[:cur_dummy_params_num]
                        s = s + self.action_center[0]
                        cur_numnode = (cur_dummy_link_idx.max()+1).astype(np.int)
                        averact = averact[:cur_numnode]
                        if j == 0:
                            state = [[np.tile(z,[cur_dummy_params_num,1]), np.tile(x,[cur_dummy_params_num,1])] for z,x in state]
                            context_state = [np.tile(z,[cur_dummy_params_num,1]) for z in context_state]
                            if self.use_gs_estimator:
                                gs_state = [np.tile(z,[cur_dummy_params_num,1]) for z in gs_state]
                                gs_est = np.zeros([int(cur_dummy_params_num), int(128*3)])

                            if self.use_message:
                                msg_nf = np.zeros((cur_numnode, self.message_dim))
                                msg_lf = np.zeros((cur_dummy_params_num, self.message_dim))
                                msg_gf = np.zeros((1, self.message_dim))

                        recon_adj = np.zeros([cur_numnode, cur_numnode])
                        di = cur_dummy_link_idx[:,1].reshape([-1,2]).astype(np.int)
                        recon_adj[di[:,0], di[:,1]] = np.sign(cur_dummy_params)

                        if np.isnan(r):
                            r = np.array(-100.)
                        r = r.reshape([1,1,1])

                        rhis = []
                        #r = OI(first_r, np.array([first_r]))[-1]


                    feed_dict = {local_AC_running.prevReward:r.reshape([1,1,1]),
                            local_AC_running.is_rollout:True,
                            local_AC_running.prevReward2:r2,
                            #local_AC_running.prevAveract:averact.reshape([1,1,-1]),
                            #local_AC_running.dummy_prevAction:cur_dummy_params.reshape([1,1,-1]),
                            local_AC_running.iidx:cur_dummy_link_idx.reshape([-1,2,2]),
                            #local_AC_running.state_in[0]:state[0], local_AC_running.state_in[1]:state[1]}
                            local_AC_running.state_in[0][0]:state[0][0], local_AC_running.state_in[0][1]:state[0][1],
                            local_AC_running.state_in[1][0]:state[1][0], local_AC_running.state_in[1][1]:state[1][1]}
                    if self.use_context or self.use_context_v2:
                        feed_dict[local_AC_running.context_state_in[0]] = context_state[0]
                        feed_dict[local_AC_running.context_state_in[1]] = context_state[1]
                    if self.use_gs_estimator:
                        feed_dict[local_AC_running.gs_state_in[0]] = gs_state[0]
                        feed_dict[local_AC_running.gs_state_in[1]] = gs_state[1]
                        #feed_dict[local_AC_running.gs_est] = np.concatenate(gs_state, -1)
                        feed_dict[local_AC_running.gs_est] = gs_est
                        feed_dict[local_AC_running.gseinput_ph] = np.array([-1]).reshape([-1,1])
                    if self.action_type == 'select':
                        if j % self.reset_step == 0:
                            a_dist = np.zeros((1,2)) - 1
                            a_dist = a_dist.astype(np.int)
                        feed_dict[local_AC_running.prevSelect] = a_dist[0,0].reshape([1,1,1])
                        feed_dict[local_AC_running.prevNewval] = a_dist[0,1].reshape([1,1,1])
                    if self.input_type == 'conv':
                        ob = self.make2dinput(s)
                        feed_dict[local_AC_running.prevAction] = ob.reshape([1,netwidth,netheight,1])
                    elif self.input_type == 'fc':
                        ob = s
                        feed_dict[local_AC_running.prevAction] = ob.reshape([1,1,-1])
                        if self.use_context:
                            feed_dict[local_AC_running.dummy_prevAction] = cur_dummy_params.reshape([1,1,-1])
                        feed_dict[local_AC_running.bs] = 1
                    elif self.input_type == 'semi_graph' or self.input_type == 'gru_semi_graph' or self.input_type == 'attention' :
                        ob = s
                        for ns in range(self.node_size):
                            feed_dict[local_AC_running.prevAction[ns]] = ob[self.idx_house[ns]].reshape([1,-1])
                            if self.use_context:
                                feed_dict[local_AC_running.dummy_prevAction[ns]] = cur_dummy_params[self.idx_house[ns]].reshape([1,-1])
                        if self.input_type == 'gru_semi_graph' or self.input_type == 'attention' :
                            feed_dict[local_AC_running.bs] = 1
                        feed_dict[local_AC_running.bst] = 1
                    elif self.input_type == 'graph_net':
                        ob = s
                        obl = np.array([make_graph_dict(ob, self.action_width[0], recon_adj, np.concatenate((r.ravel(),r2.ravel())))], dtype=np.object)
                        obp = utils_tf.get_feed_dict(local_AC_running.prevAction, utils_np.data_dicts_to_graphs_tuple(obl))
                        feed_dict.update(obp)
                        if self.use_message:
                            oblm = np.array([make_msg_graph_dict(ob, self.action_width[0], recon_adj, msg_nf, msg_lf, msg_gf, np.concatenate((r.ravel(),r2.ravel())))], dtype=np.object)
                            obpm = utils_tf.get_feed_dict(local_AC_running.prevMsg, utils_np.data_dicts_to_graphs_tuple(oblm))
                            feed_dict.update(obpm)
                        feed_dict[local_AC_running.graph_iidx] = np.arange(s.size)
                        feed_dict[local_AC_running.bs] = 1
                        feed_dict[local_AC_running.bst] = 1

                    if self.use_varout:
                        feed_dict[local_AC_running.zpsize] = 1
                        if j % self.reset_step == 0:
                            feed_dict[local_AC_running.prevAction_raw] = np.zeros_like(ob).reshape([-1,1,1]) - 1
                        else:
                            feed_dict[local_AC_running.prevAction_raw] = ob.reshape([-1,1,1])

                    #Probabilistically pick an action given our network outputs.
                    a_dist, a_log_prob, v, state, context_state, gs_state_new, gs_est_new, entropy_each, msg_nf_new, msg_lf_new, msg_gf_new = sess.run([local_AC_running.policy, local_AC_running.all_log_probs, local_AC_running.value, local_AC_running.rnn_state, local_AC_running.context_rnn_state, local_AC_running.gs_rnn_state, local_AC_running.gs_estimate, local_AC_running.entropy_each, local_AC_running.msg_nf, local_AC_running.msg_lf, local_AC_running.msg_gf],
                                      feed_dict=feed_dict)

                    
                    if pretrain.value == 1:
                        if np.random.rand() < 0.05:
                            replace_idx = np.random.choice(np.arange(cur_dummy_params_num), size=int(cur_dummy_params_num*0.5), replace=False)
                            a_dist[replace_idx,0] = cur_dummy_params[replace_idx] + self.action_center[0]
                            feed_dict[local_AC_running.is_rollout] = False
                            feed_dict[local_AC_running.gseinput_ph] = a_dist
                            gs_state_new, gs_est_new = sess.run([local_AC_running.gs_rnn_state, local_AC_running.gs_estimate], feed_dict=feed_dict)
                    else:
                        replace_idx = np.random.choice(np.arange(cur_dummy_params_num), size=int(cur_dummy_params_num*0.05), replace=False)
                        a_dist[replace_idx,0] = cur_dummy_params[replace_idx] + self.action_center[0]
                        feed_dict[local_AC_running.is_rollout] = False
                        feed_dict[local_AC_running.gseinput_ph] = a_dist
                        gs_state_new, gs_est_new = sess.run([local_AC_running.gs_rnn_state, local_AC_running.gs_estimate], feed_dict=feed_dict)

                    normalized_entropy = entropy_each.reshape([-1])
                    if self.action_type == 'discrete':
                        if self.incremental_action and not self.use_varout:
                            a_inc = a_dist - (self.inc_bound//2)
                            s_ = np.clip(s + a_inc.reshape([-1]), 0, self.action_width-1)
                            a_dist_scaled = s_ - self.action_center
                        elif self.use_varout:
                            if self.incremental_action:
                                a_inc = a_dist.ravel() - (self.inc_bound//2)
                                s_ = np.clip(s + a_inc.reshape([-1]), 0, self.action_width_real-1)
                                a_dist_scaled = s_.reshape([-1]) - self.action_center
                            elif self.stay_prev:
                                s_ = s.copy()
                                cidx = a_dist.ravel()!=0
                                s_[cidx] = a_dist.ravel()[cidx] - 1
                                a_dist_scaled = s_.reshape([-1]) - self.action_center
                            else:
                                a_dist_scaled = a_dist.reshape([-1]) - self.action_center[0]
                        else:
                            a_dist_scaled = a_dist - self.action_center
                    elif self.action_type == 'continuous':
                        if self.incremental_action:
                            a_inc = a_dist - 0.5
                            s_raw = np.clip(s_raw + a_inc.reshape([-1]), 0, 1)
                            s_ = np.round(s_raw * self.action_width)
                            a_dist_scaled = s_ - self.action_center
                        else:
                            a_dist_ = np.round(a_dist * self.action_width)
                            a_dist_scaled = a_dist_ - self.action_center
                    if self.action_type == 'select':
                        s_ = s.copy()
                        if self.incremental_action:
                            a_inc = a_dist[0][1] - (self.inc_bound//2)
                            s_[a_dist[0][0]] = np.clip(s[a_dist[0][0]] + a_inc, 0, self.action_width[0]-1)
                        else:
                            if s_[a_dist[0][0]] == a_dist[0][1]:
                                same_penalty = -0.01
                            else:
                                same_penalty = 0
                            sp_house.append(same_penalty)
                            s_[a_dist[0][0]] = a_dist[0][1]
                            #s_ = self.s_setting(s_, a_dist[0][0], a_dist[0][1])
                        if self.use_randsign:
                            a_dist_scaled = s_.copy() - self.action_center
                        else:
                            a_dist_scaled = s_.copy()

                    self.params[:cur_dummy_params_num] = a_dist_scaled.reshape([-1])
                    self.seed.value = episode_count+int(self.name[-1])*5000
                    self.tvt.value = 3
                    #self.tvt.value = 0
                    self.START_CALC_EVENT.set()
                    r_ = self.rq.get()
                    #r_ -= 0.05 * np.mean(np.round(np.abs(a_dist_scaled)))
                    spear = self.sq.get()
                    mui = self.mi.get()
                    #averact = np.array(self.averact[:].copy())

                    if np.isnan(r_):
                        r_ = np.array(-100.)
                    if np.isnan(spear):
                        spear = np.array(0)
                    rhis.append(r_)
                    roi = OI(first_r, np.array(rhis))[-1]

                    if self.use_curiosity:
                        feed_dict = {local_AC_running.prevReward:r.reshape([1,1,1]),
                                local_AC_running.nextReward:r_.reshape([1,1,1]),
                                local_AC_running.action_holder:a_dist.reshape([1,self.a_size])}
                        if self.input_type == 'conv':
                            neob = self.make2dinput(a_dist_scaled.reshape([-1]))
                            feed_dict[local_AC_running.prevAction] = ob.reshape([1,netwidth,netheight,1])
                            feed_dict[local_AC_running.nextAction] = neob.reshape([1,netwidth,netheight,1])
                        elif self.input_type == 'semi_graph':
                            neob = a_dist_scaled.reshape([-1])
                            for ns in range(self.node_size):
                                feed_dict[local_AC_running.nextAction[ns]] = neob[self.idx_house[ns]].reshape([1,-1])
                                feed_dict[local_AC_running.prevAction[ns]] = ob[self.idx_house[ns]].reshape([1,-1])
                        intrinsic_reward = sess.run([local_AC_running.intrinsic_reward], feed_dict=feed_dict)[0]

                    if self.input_type == 'conv':
                        his = [ob.reshape([-1,netwidth,netheight,1]),r.reshape([-1]),a_dist,r_.reshape([-1]),v[0,0], a_log_prob]
                    elif self.action_type == 'select':
                        his = [ob.reshape([-1,self.a_size]),r.reshape([-1]),a_dist,r_.reshape([-1]),v[0,0], a_log_prob,averact,a_dist[0,0],a_dist[0,1],same_penalty,cur_dummy_params.reshape([-1,self.a_size])]
                    else:
                        obtohis = ob.reshape([-1,cur_dummy_params_num])
                        his = [obtohis,r.reshape([-1]),a_dist,r_.reshape([-1]),v, a_log_prob,averact,0,0,0,cur_dummy_params.reshape([-1,cur_dummy_params_num]),cur_dummy_link_idx, r2.reshape([-1])]
                    if self.input_type == 'graph_net':
                        obtohis = obl.reshape([-1,1])
                        his += [obtohis]
                    if self.use_gs_estimator:
                        #his += [np.concatenate(gs_state, axis=-1)]
                        his += [gs_est]
                    if self.use_message:
                        msgobtohis = oblm.reshape([-1,1])
                        his += [msgobtohis]
                    if self.use_curiosity:
                        his += [intrinsic_reward.reshape([-1])]
                    ep_history.append(his) # (s,r) - state, action - action, r_ - reward, v - value, a_dist - action_probs
                    if self.incremental_action or self.action_type == 'select' or (self.action_type == 'discrete' and self.stay_prev):
                        s = s_.copy()
                    else:
                        s = a_dist.copy().reshape([-1])
                        #s = np.round(a_dist_scaled.reshape([-1]))
                        if self.action_type == 'continuous':
                            s = a_dist_.copy().reshape([-1])
                    r = r_ - first_r[j//self.reset_step]
                    r2 = np.array([spear-first_s[j//self.reset_step], mui-first_mi[j//self.reset_step], roi]).reshape([1,1,-1])
                    gs_state = gs_state_new
                    gs_est = gs_est_new.reshape([-1,128*3])
                    if self.use_message:
                        msg_nf = msg_nf_new
                        msg_lf = msg_lf_new
                        msg_gf = msg_gf_new
                    
                    running_reward.append(r_)
                    running_spear += spear
                    if self.use_curiosity:
                        running_intrinsic += intrinsic_reward
                    if er < r_:
                        running_action = a_dist_scaled
                        running_entropy = normalized_entropy
                        actiontemp = a_dist
                        er = r_
#                print('Episode time ', time()-starttime)
                
                feed_dict = {local_AC_running.prevReward:r.reshape([1,1,1]),
                        local_AC_running.is_rollout:True,
                        local_AC_running.prevReward2:r2,
                        #local_AC_running.prevAveract:averact.reshape([1,1,-1]),
                        local_AC_running.iidx:cur_dummy_link_idx.reshape([-1,2,2]),
                        #local_AC_running.state_in[0]:state[0], local_AC_running.state_in[1]:state[1]}
                        local_AC_running.state_in[0][0]:state[0][0], local_AC_running.state_in[0][1]:state[0][1],
                        local_AC_running.state_in[1][0]:state[1][0], local_AC_running.state_in[1][1]:state[1][1]}
                if self.use_context or self.use_context_v2:
                    feed_dict[local_AC_running.context_state_in[0]] = context_state[0]
                    feed_dict[local_AC_running.context_state_in[1]] = context_state[1]
                if self.use_gs_estimator:
                    feed_dict[local_AC_running.gs_state_in[0]] = gs_state[0]
                    feed_dict[local_AC_running.gs_state_in[1]] = gs_state[1]
                    #feed_dict[local_AC_running.gs_est] = np.concatenate(gs_state, -1)
                    feed_dict[local_AC_running.gs_est] = gs_est
                    feed_dict[local_AC_running.gseinput_ph] = np.array([-1]).reshape([-1,1])

                if self.action_type == 'select':
                    feed_dict[local_AC_running.prevSelect] = a_dist[0,0].reshape([1,1,1])
                    feed_dict[local_AC_running.prevNewval] = a_dist[0,1].reshape([1,1,1])
                if self.input_type == 'conv':
                    ob = self.make2dinput(s)
                    feed_dict[local_AC_running.prevAction] = ob.reshape([1,netwidth,netheight,1])
                elif self.input_type == 'fc':
                    ob = s
                    feed_dict[local_AC_running.prevAction] = ob.reshape([1,1,-1])
                    feed_dict[local_AC_running.bs] = 1
                elif self.input_type == 'semi_graph' or self.input_type == 'gru_semi_graph' or self.input_type == 'attention':
                    ob = s
                    for ns in range(self.node_size):
                        feed_dict[local_AC_running.prevAction[ns]] = ob[self.idx_house[ns]].reshape([1,-1])
                    if self.input_type == 'gru_semi_graph' or self.input_type == 'attention':
                        feed_dict[local_AC_running.bs] = 1
                    feed_dict[local_AC_running.bst] = 1
                elif self.input_type == 'graph_net':
                    ob = s
                    obp = [make_graph_dict(ob, self.action_width[0], recon_adj, np.concatenate((r.ravel(),r2.ravel())))]
                    obp = utils_tf.get_feed_dict(local_AC_running.prevAction, utils_np.data_dicts_to_graphs_tuple(obp))
                    feed_dict.update(obp)
                    if self.use_message:
                        oblm = np.array([make_msg_graph_dict(ob, self.action_width[0], recon_adj, msg_nf, msg_lf, msg_gf, np.concatenate((r.ravel(),r2.ravel())))], dtype=np.object)
                        obpm = utils_tf.get_feed_dict(local_AC_running.prevMsg, utils_np.data_dicts_to_graphs_tuple(oblm))
                        feed_dict.update(obpm)
                    feed_dict[local_AC_running.graph_iidx] = np.arange(s.size)
                    feed_dict[local_AC_running.bs] = 1
                    feed_dict[local_AC_running.bst] = 1

                if self.use_varout:
                    feed_dict[local_AC_running.zpsize] = 1
                    feed_dict[local_AC_running.prevAction_raw] = ob.reshape([-1,1,1])


                v1 = sess.run([local_AC_running.value],
                              feed_dict=feed_dict)[0]

                
                trainoi = []
                for epi in range(self.max_ep // self.reset_step):
                    curfitness = running_reward[epi*self.reset_step:(epi+1)*self.reset_step]
                    trainoi.append(np.sum(OI(first_r[epi],curfitness).reshape([-1])))
                self.total_reward.append(np.sum(trainoi))
                self.total_spear.append(running_spear/(j+1))
                if self.use_curiosity:
                    self.total_intrinsic.append(running_intrinsic/(j+1))
                
                if self.name == 'worker_0':
                    sess.run(self.increment)
                    if episode_count % 100 == 0:
                        maxr_arg = np.argmax(self.total_reward[-100:])
                        print(np.nanmean(self.total_reward[-100:]),
                              self.total_reward[-100:][maxr_arg], episode_count)
                    if episode_count > 16:
                        cr = np.mean(self.total_reward[-16:])
                        if self.initial_reward < cr:
                            self.initial_reward = cr
#                            saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                            print ("Saved Model", episode_count, cr, a_dist_scaled)
                    #if episode_count > total_episodes:
                    #    coord.request_stop()

                    if episode_count > self.pretrain_epi:
                        if pretrain.value == 1:
                            pretrain_saver.save(sess,self.model_path+'/pretrain/model-'+str(episode_count)+'.cptk')
                            pretrain.value = 0

                self.process_and_add_experience(ep_history,gamma,v1)
                if self.is_ready_update():
                    self.memory_dict[self.name].append(self.training_buffer)

                    self.barrier.wait()
                    if self.name == 'worker_0':
                        self.global_epi.value = episode_count
                        TRAIN_EVENT.set()
                        COLLECT_EVENT.clear()
                    self.barrier.wait()
                    COLLECT_EVENT.wait()
                    #self.training_buffer.reset()

                if episode_count % 10 == 0:
                    if episode_count % 20 == 0:
                        original_mb = self.mb_size.value
                        self.mb_size.value = self.mblimit
                        first_rt, target_rew, target_action, sp_t = self.run_target_task(actiontemp)
                        self.mb_size.value = original_mb
                        self.params[:self.a_size] = target_action.reshape([-1])
                        self.tvt.value = 2
                        self.START_CALC_EVENT.set()
                        rtest = self.rq.get()
                        speartest = self.sq.get()
                        muitest = self.mi.get()

                        rtraining_whole = np.max(target_rew)
                        self.total_target_reward.append(rtraining_whole)

                        cr = np.mean(self.total_target_reward[-16:])
                        if self.initial_reward < cr:
                            self.initial_reward = cr
                            print ("PKN Saved", self.number, episode_count, cr)
                            np.savetxt(self.model_path+'/pkn_params_'+str(self.number)+'.txt',np.array(target_action.reshape([1,-1])),footer=str(np.round(cr,4))+'/'+str(np.round(rtest,4)))
                            saver.save(sess,self.model_path+'/train_'+str(self.number)+'/model-'+str(episode_count)+'.cptk')
                    if episode_count > 0 and episode_count % 5e3 == 0:
                        saver.save(sess,self.model_path+'/train_'+str(self.number)+'/peri_model-'+str(episode_count)+'.cptk')
                    
                    mean_reward = np.mean(self.total_reward[-1])
                    mean_spear = np.mean(self.total_spear[-1])
                    summary = tf.compat.v1.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Spear', simple_value=float(mean_spear))
                    summary.value.add(tag='Perf/max_fit', simple_value=float(np.max(running_reward)))
                    if episode_count % 20 == 0:
                        summary.value.add(tag='Perf/R_test', simple_value=float(rtest))
                        summary.value.add(tag='Perf/Spear_test', simple_value=float(speartest))
                        summary.value.add(tag='Perf/R_train_whole', simple_value=float(rtraining_whole))
                        summary.value.add(tag='Perf/R_OI_train', simple_value=float(np.sum(OI(first_rt, np.array(target_rew)))))
                    self.summary_writer.add_summary(summary, episode_count)
                    summ = sess.run([self.local_AC.entropy_histogram], feed_dict={self.local_AC.entph:normalized_entropy.reshape([-1])})[0]
                    self.summary_writer.add_summary(summ, episode_count)

                    self.summary_writer.flush()

                episode_count += 1
                #if episode_count > 0 and episode_count % 30 == 0:
                #    self.mb_size.value += 1
                #    if self.mb_size.value > self.mblimit:
                #        self.mb_size.value = self.mblimit
                if episode_count % 30016 == 0 and self.max_ep < 80:
                #if episode_count % (self.buffer_ready_size*20000) == 0 and self.max_ep < 30:
                    self.max_ep += 0
                    self.reset_step += 0
                #if episode_count > self.pretrain_epi and episode_count % 5e3 == 0 and self.max_task_num.value < 1e4:
                if episode_count % 3e4 == 0 and self.max_task_num.value < 5e3:
                    self.max_task_num.value += 0 
                    

                #self.epsilon *= 0.999
                sys.stdout.flush()

     
if __name__ == '__main__':           
    tf.compat.v1.reset_default_graph()
    
    use_randsign = True
    max_task_num = 100000
    scale = 5
    env = test_prob.simul(num_initial=128, start_process=False, max_task_num=max_task_num, use_randsign=use_randsign, scale=scale)
    link_size, node_size = env.get_num_params()
    a_size = link_size# + node_size
    
    aw, ac = env.get_link_scale(level=3, scale= scale)    
    #action_width = np.concatenate((aw, np.zeros(node_size) + scale*2))
    #action_center = np.concatenate((ac, np.zeros(node_size) + scale))
    action_width = aw
    action_center = ac
    #aw, ac, bw, bc = env.get_link_scale(level=3, scale=scale, basal_scale=3)
    ##action_width = np.concatenate((aw, bw))
    ##action_center = np.concatenate((ac, bc))
    #action_width = np.concatenate((aw, np.zeros(0) + 0*2))
    #action_center = np.concatenate((ac, np.zeros(0) + 0))
    
    
    epsilon = 1 - (10 ** (np.log10(0.95) / a_size)) # NOT USED
    total_episodes = 3001 # NOT USED
    savedir = 'GREY'

    if not os.path.exists(savedir):
        os.mkdir(savedir)
    
    parameter_dict = {
            'a_size': a_size,
            'h_size': 512,
            'node_size': node_size,
            'link_size': link_size,
            'num_input_node': env.num_input_node,
            'use_randsign': use_randsign,
            'max_task_num': max_task_num,
            'input_type': 'graph_net', #fc, conv, semi_graph, gru_semi_graph, attention, graph_net
            'output_type': 'fc', #fc, semi_graph, gru_semi_graph
            'sg_unit': '64-3', # num_unit-num_layer
            'action_type': 'discrete', #continuous, discrete, select
            'adj_mat': np.sign(env.weightmat),
            'start_lr': 1e-5,
            'use_update_noise': False,
            'noise_start_scale': 0.01,
            'distance_threshold': 0.3,
            'gamma': 0.95,
            'pretrain_epi': 8000,
            'max_ep': 80,
            'reset_step': 80,
            'model_path': savedir,
            'mb_size':16,
            'use_curiosity': False,
            'use_noisynet': False,
            'use_attention': False,
            'use_context': False,
            'use_context_v2': True,
            'averact_context_target': False,
            'use_gs_estimator': True,
            'use_message': True,
            'message_dim': 32,
            'gs_attention': False,
            'use_ib': False,
            'probabilistic_context': False,
            'use_varout': True,
            'curiosity_encode_size': 128,
            'curiosity_strength': 0.05,
            'buffer_size': 64,
            'buffer_ready_size': 64,
            'update_batch_size': 4,
            'num_epoch': 3,
            'action_branching': False,
            'incremental_action': False,
            'stay_prev': False,
            'inc_bound':3,
            'autoregressive':False
            }
    assert(not (parameter_dict['use_update_noise'] and parameter_dict['use_noisynet']))
    assert(not (parameter_dict['use_context'] and parameter_dict['use_context_v2']))
    if parameter_dict['action_type'] == 'discrete' or parameter_dict['action_type'] == 'select':
        action_width += 1
        if parameter_dict['stay_prev']:
            action_width += 1
    
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        learning_rate = parameter_dict['start_lr']
        #learning_rate = tf.train.cosine_decay_restarts(parameter_dict['start_lr'], global_step,
        #                                       20000, t_mul=1, m_mul=0.9, alpha=0.1)
        trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        memory_dict = {}
    #    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
        num_workers = 10 # Set workers to number of available CPU threads
        barrier = multiprocessing.Barrier(num_workers)
        global_epi = multiprocessing.Value('d',0)

        workers = []
        # Create worker classes
        for i in range(num_workers):
            memory_dict['worker_%i'%i] = deque()
            workers.append(Worker(i, parameter_dict, epsilon, trainer, num_workers, global_episodes, global_step, memory_dict, barrier, global_epi))

        master_network = AC_Network(parameter_dict=parameter_dict, action_width=action_width, scope='global', trainer=trainer,
                                    num_workers=num_workers,global_step=global_step, worker_lists=workers) # Generate global network
        master = Master(env, parameter_dict, master_network, trainer, global_step, memory_dict, workers, global_epi)
        for w in workers:
            w.set_update_local_ops()
        saver = tf.compat.v1.train.Saver(max_to_keep=5)
        pretrain_saver = tf.compat.v1.train.Saver(max_to_keep=1)
        
    #seconfig = tf.compat.v1.ConfigProto(allow_soft_placement = True, intra_op_parallelism_threads=15, inter_op_parallelism_threads=15)
    seconfig = tf.compat.v1.ConfigProto(allow_soft_placement = True)
    seconfig.gpu_options.allow_growth = True
    # Launch the tensorflow graph
    with tf.compat.v1.Session(config=seconfig) as sess:
        coord = tf.train.Coordinator()
        
        sess.run(tf.compat.v1.global_variables_initializer())

        pretrain = multiprocessing.Value('d',0)
        TRAIN_EVENT = threading.Event()
        COLLECT_EVENT = threading.Event()
        TRAIN_EVENT.clear()
        COLLECT_EVENT.clear()
    #    s_init = np.random.rand(1,1,a_size)
        s_init = None
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(parameter_dict['gamma'], sess, coord, saver, pretrain_saver, total_episodes, TRAIN_EVENT, COLLECT_EVENT, pretrain, s_init=s_init)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)

        threads = []
        threads.append(threading.Thread(target=master.check(sess, coord, TRAIN_EVENT, COLLECT_EVENT, pretrain)))
        threads[0].start()

        threads += worker_threads

        coord.join(threads)
