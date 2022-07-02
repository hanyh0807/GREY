import os
import sys
import warnings
os.environ["OPENBLAS_NUM_THREADS"] = "10"
#os.environ["NUMBA_NUM_THREADS"] = "5"

if not sys.warnoptions:
    warnings.simplefilter("ignore")
sys.path.append(os.path.dirname(__file__))

import threading
from multiprocessing import Lock, Array, Value, Barrier
import multiprocessing
import tensorflow as tf
import tensorflow_probability as tfp
from graph_nets import blocks, graphs, modules, utils_np, utils_tf
import sonnet as snt
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from time import sleep, time
from collections import deque
import copy
import pickle
import networkx as nx
import termplotlib as tpl

import python_networks.synthetic.Si_restore as test_prob

from restoremodel import *

class GREY():
    def __init__(self, params):
        use_randsign = True
        max_task_num = 100000
        scale = 5
        env = test_prob.simul(num_initial=1, start_process=False, max_task_num=max_task_num, use_randsign=use_randsign, scale=scale)
        link_size, node_size = env.get_num_params()
        a_size = link_size# + node_size

        aw, ac = env.get_link_scale(level=3, scale= scale)
        action_width = aw
        action_center = ac


        epsilon = 1 - (10 ** (np.log10(0.95) / a_size)) #NOT USED
        total_episodes = 3001 #NOT USED

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
                'gamma': 0.95,
                'pretrain_epi': -1,
                'max_ep': 80,
                'reset_step': 80,
                'model_path': '_',
                'mb_size':32,
                'use_curiosity': False,
                'curiosity_encode_size': 128,
                'curiosity_strength': 0.05,
                'use_noisynet': False,
                'use_attention': False,
                'use_context_v2': True,
                'use_gs_estimator': True,
                'use_message': True,
                'message_dim': 32,
                'gs_attention': False,
                'use_ib': False,
                'use_varout': True,
                'buffer_size': 64,
                'buffer_ready_size': 64,
                'update_batch_size': 4,
                'num_epoch': 3,
                'incremental_action': False,
                'inc_bound':3,
                'network_file':'example_data/trametinib_network.csv',
                'profile_file':'example_data/Core_Profile_MM.csv',
                'targets':['MAP2K1', 'MAP2K2']
                }
        parameter_dict.update(params)
        if parameter_dict['action_type'] == 'discrete' or parameter_dict['action_type'] == 'select':
            action_width += 1

        with tf.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False)
            global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            learning_rate = parameter_dict['start_lr']
            trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

            memory_dict = {}
            num_workers = 1 # Set workers to number of available CPU threads
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

        seconfig = tf.compat.v1.ConfigProto(allow_soft_placement = True, intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
        seconfig.gpu_options.allow_growth = True

        self.sess = tf.compat.v1.Session(config=seconfig)
        #self.sess.run(tf.compat.v1.global_variables_initializer())

        modelpath = 'checkpoint/model.ckpt'
        saver.restore(self.sess, modelpath)
        print('RESTORED')

        self.worker = workers[0]
        #self.max_ep = 500
        self.max_ep = 5

    def optimize(self, max_step, opt_try):
        '''
        Inputs
        max_step : Maximum step for each optimization trial
        opt_try : Number of trial for optimization

        Outputs
        qa : Optimized parameters for each optimization trial
        q : Fitness of each optimized parameters
        '''
        actrepeat = True
        repnum = 5

        qa, qt, q, qoi, qhis = [], [], [], [], []
        seed = 2457245
        for _ in range(opt_try):
            st = time()
            print('Trial ',_)
            self.worker.val_env.do_initialize_task.value = 0
            self.worker.val_env.seed.value = seed
            self.worker.val_env.mb_size.value = self.worker.val_env.train_y.shape[1]
            np.random.seed()
            asize = self.worker.val_env.get_num_params()[0]
            s = np.zeros(asize) + 0.5

            self.worker.val_env.do_reset_initials.value = 1

            if self.worker.action_type == 'discrete' or self.worker.action_type == 'select':
                s = np.round(s * (self.worker.action_width[0] - 1))
                s_scaled = s - self.worker.action_center[0]
            elif self.worker.action_type == 'continuous':
                s_scaled = np.round((s * self.worker.action_width) - self.worker.action_center)

            signmat = np.sign(self.worker.val_env.weightmat[self.worker.val_env.weightmat!=0])
            s = np.random.choice(np.arange(1,self.worker.action_center[0]+1), size=signmat.size)
            s_scaled = s * signmat
            s = s_scaled + self.worker.action_center[0]

            self.worker.val_env.params[:asize] = s_scaled.reshape([-1])
            self.worker.val_env.seed.value = seed
            self.worker.val_env.tvt.value = 0
            self.worker.val_env.START_CALC_EVENT.set()
            r = self.worker.val_env.rq.get()
            first_r = r
            spear = self.worker.val_env.sq.get()
            mui = self.worker.val_env.zc.get()
            first_s = spear
            first_mi = mui
            r = r - first_r
            r2 = np.array([spear-first_s, mui-first_mi, 0]).reshape([1,1,-1])
            self.worker.val_env.do_reset_initials.value = 0
            self.worker.val_env.do_initialize_task.value = 0

            cur_dummy_params_num = asize
            cur_dummy_link_idx = np.array(self.worker.val_env.weightmat_link_idx[:].copy())[:int(asize*2)]
            cur_dummy_link_idx = cur_dummy_link_idx.ravel().reshape([-1,1])
            cur_dummy_link_idx = np.concatenate((np.zeros_like(cur_dummy_link_idx), cur_dummy_link_idx), axis=1)

            cur_numnode = (cur_dummy_link_idx.max()+1).astype(np.int)

            state = self.worker.local_AC.state_init #Reset the recurrent layer's hidden state
            context_state = self.worker.local_AC.context_state_init #Reset the recurrent layer's hidden state
            if self.worker.use_gs_estimator:
                gs_state = self.worker.local_AC.gs_state_init

            state = [[np.tile(z,[cur_dummy_params_num,1]), np.tile(x,[cur_dummy_params_num,1])] for z,x in state]
            context_state = [np.tile(z,[cur_dummy_params_num,1]) for z in context_state]
            if self.worker.use_gs_estimator:
                gs_state = [np.tile(z,[cur_dummy_params_num,1]) for z in gs_state]
                gs_est = np.zeros([int(cur_dummy_params_num), int(128*3)])
            if self.worker.use_message:
                msg_nf = np.zeros((cur_numnode, self.worker.message_dim))
                msg_lf = np.zeros((cur_dummy_params_num, self.worker.message_dim))
                msg_gf = np.zeros((1, self.worker.message_dim))

            recon_adj = np.sign(self.worker.val_env.weightmat)

            if np.isnan(r):
                r = np.array(-100.)
            r = r.reshape([1,1,1])

            local_AC_running = self.worker.local_AC

            er = -np.inf
            actiontemp = []
            his = []
            rhis = []
            r2his = []
            roihis = []
            ahis = []
            rnnhis = []
            statehis = []
            gshis = []

            for j in range(max_step):
                feed_dict = {local_AC_running.prevReward:r.reshape([1,1,1]),
                                local_AC_running.is_rollout:True,
                                local_AC_running.prevReward2:r2,
                                local_AC_running.iidx:cur_dummy_link_idx.reshape([-1,2,2]),
                                local_AC_running.state_in[0][0]:state[0][0], local_AC_running.state_in[0][1]:state[0][1],
                                local_AC_running.state_in[1][0]:state[1][0], local_AC_running.state_in[1][1]:state[1][1]}
                if self.worker.use_context_v2:
                            feed_dict[local_AC_running.context_state_in[0]] = context_state[0]
                            feed_dict[local_AC_running.context_state_in[1]] = context_state[1]
                if self.worker.use_gs_estimator:
                            feed_dict[local_AC_running.gs_state_in[0]] = gs_state[0]
                            feed_dict[local_AC_running.gs_state_in[1]] = gs_state[1]
                            feed_dict[local_AC_running.gs_est] = gs_est
                            feed_dict[local_AC_running.gseinput_ph] = np.array([-1]).reshape([-1,1])
                ob = s
                if self.worker.use_message:
                    oblm = np.array([make_msg_graph_dict(ob, self.worker.action_width[0], recon_adj, msg_nf, msg_lf, msg_gf, np.concatenate((r.ravel(),r2.ravel())))], dtype=np.object)
                    obpm = utils_tf.get_feed_dict(local_AC_running.prevMsg, utils_np.data_dicts_to_graphs_tuple(oblm))
                    feed_dict.update(obpm)
                obl = np.array([make_graph_dict(ob, self.worker.action_width[0], recon_adj, np.concatenate((r.ravel(),r2.ravel())))], dtype=np.object)
                obp = utils_tf.get_feed_dict(local_AC_running.prevAction, utils_np.data_dicts_to_graphs_tuple(obl))
                feed_dict.update(obp)
                feed_dict[local_AC_running.graph_iidx] = np.arange(s.size)
                feed_dict[local_AC_running.bs] = 1
                feed_dict[local_AC_running.bst] = 1

                if self.worker.use_varout:
                    feed_dict[local_AC_running.zpsize] = 1
                    if j  == 0:
                        feed_dict[local_AC_running.prevAction_raw] = np.zeros_like(ob).reshape([-1,1,1]) - 1
                    else:
                        feed_dict[local_AC_running.prevAction_raw] = ob.reshape([-1,1,1])

                _a_dist, _state, _context_state, _gs_state_new, _gs_est_new, _msg_nf_new, _msg_lf_new, _msg_gf_new = [], [], [], [], [], [], [], []
                rtemp, stemp, mtemp = [], [], []
                _a_dist_scaled = []
                for rep in range(repnum):
                    rst = time()
                    feed_dict[local_AC_running.is_rollout] = True
                    a_dist_c, state_c, context_state_c, gs_state_new_c, gs_est_new_c, msg_nf_new_c, msg_lf_new_c, msg_gf_new_c = self.sess.run([local_AC_running.policy, local_AC_running.rnn_state, local_AC_running.context_rnn_state, local_AC_running.gs_rnn_state, local_AC_running.gs_estimate, local_AC_running.msg_nf, local_AC_running.msg_lf, local_AC_running.msg_gf],
                                              feed_dict=feed_dict)

                    
                    if j > 10:
                        if actrepeat & (np.random.rand()<100):
                            if (np.random.rand()<-0.5):
                                replace_idx = np.random.choice(np.arange(cur_dummy_params_num), size=int(cur_dummy_params_num*0.5), replace=False)
                                maxadist = ahis[np.argmax(rhis)].copy()
                                a_dist_c[replace_idx,0] = maxadist.ravel()[replace_idx]
                            else:
                                ur, ui = np.unique(rhis, return_index=True)
                                for uiui in range(1,0,-1):
                                    replace_idx = np.random.choice(np.arange(cur_dummy_params_num), size=int(cur_dummy_params_num*0.5), replace=False)
                                    maxadist = ahis[int(ui[np.argsort(ur)[-uiui]])].copy()
                                    a_dist_c[replace_idx,0] = maxadist.ravel()[replace_idx]

                            feed_dict[local_AC_running.is_rollout] = False
                            feed_dict[local_AC_running.gseinput_ph] = a_dist_c
                            gs_state_new_c, gs_est_new_c = self.sess.run([local_AC_running.gs_rnn_state, local_AC_running.gs_estimate], feed_dict=feed_dict)

                    _a_dist.append(a_dist_c)
                    _state.append(state_c)
                    _context_state.append(context_state_c)
                    _gs_state_new.append(gs_state_new_c)
                    _gs_est_new.append(gs_est_new_c)
                    _msg_nf_new.append(msg_nf_new_c)
                    _msg_lf_new.append(msg_lf_new_c)
                    _msg_gf_new.append(msg_gf_new_c)

                    if self.worker.action_type == 'discrete':
                        a_dist_scaled = a_dist_c.reshape([-1]) - self.worker.action_center[0]

                    self.worker.val_env.params[:cur_dummy_params_num] = a_dist_scaled.reshape([-1])
                    self.worker.val_env.seed.value = seed
                    self.worker.val_env.tvt.value = 0
                    self.worker.val_env.START_CALC_EVENT.set()
                    r_ = self.worker.val_env.rq.get()
                    spear = self.worker.val_env.sq.get()
                    mui = self.worker.val_env.zc.get()

                    sys.stdout.flush()
                    rtemp.append(r_)
                    stemp.append(spear)
                    mtemp.append(mui)
                    _a_dist_scaled.append(a_dist_scaled)

                isreplaced = False
                if (er < np.max(rtemp)) | (np.random.rand() < 0.7):
                    if (er < np.max(rtemp)):
                        maxidx = np.argmax(rtemp)
                    elif actrepeat & (np.random.rand()<0.5) & (j > 0):
                        ur, ui = np.unique(rhis, return_index=True)
                        hismax = int(np.random.choice(ui[np.argsort(ur)[-3:]]))
                        a_dist = ahis[hismax].copy()
                        feed_dict[local_AC_running.is_rollout] = False
                        feed_dict[local_AC_running.gseinput_ph] = a_dist
                        gs_state_new, gs_est_new = self.sess.run([local_AC_running.gs_rnn_state, local_AC_running.gs_estimate], feed_dict=feed_dict)

                        a_dist_scaled = a_dist.reshape([-1]) - self.worker.action_center[0]
                        r_ = rhis[hismax]
                        spear = r2his[hismax][0]
                        mui = r2his[hismax][1]
                        maxidx = np.argmax(rtemp)
                        isreplaced = True
                    else:
                        maxidx = np.argmax(rtemp)
                else:
                    maxidx = np.random.choice(repnum, 1)[0]

                if not isreplaced:
                    a_dist = _a_dist[maxidx]
                    gs_state_new = _gs_state_new[maxidx]
                    gs_est_new = _gs_est_new[maxidx]
                    r_ = rtemp[maxidx]
                    spear = stemp[maxidx]
                    mui = mtemp[maxidx]
                    a_dist_scaled = _a_dist_scaled[maxidx]

                state = _state[maxidx]
                context_state = _context_state[maxidx]
                msg_nf_new = _msg_nf_new[maxidx]
                msg_lf_new = _msg_lf_new[maxidx]
                msg_gf_new = _msg_gf_new[maxidx]

                print('step:',j,' fitness',r_)
                print('step:',j,' param',a_dist_scaled)

                if np.isnan(r_):
                    r_ = np.array(-100.)
                if np.isnan(spear):
                    spear = np.array(0)

                s = a_dist.copy().reshape([-1])

                rhis.append(r_)
                r2his.append([spear, mui])
                roi = OI(first_r, np.array(rhis))[-1]
                r = r_ - first_r
                r2 = np.array([spear-first_s, mui-first_mi, roi]).reshape([1,1,-1])
                gs_state = gs_state_new
                gs_est = gs_est_new.reshape([-1,128*3])
                if self.worker.use_message:
                    msg_nf = msg_nf_new
                    msg_lf = msg_lf_new
                    msg_gf = msg_gf_new


                his.append(r_)
                ahis.append(a_dist)

                if er < r_:
                    running_action = a_dist_scaled
                    er = r_

            optmat = self.worker.val_env.weightmat.copy()
            optmat[optmat!=0] = running_action
            qa.append(pd.DataFrame(optmat, columns=self.worker.val_env.Nodes, index=self.worker.val_env.Nodes))
            q.append(np.max(his))
            print(_, time() - st, 'Elapsed')

            x = np.arange(max_step)
            fig = tpl.figure()
            fig.plot(x, his)
            fig.show()

        return qa, q

    def close(self):
        pid = os.getpid()
        os.system('pkill -P %d'%pid)
