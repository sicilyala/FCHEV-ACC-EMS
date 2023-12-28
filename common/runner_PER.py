import torch
from tqdm import tqdm
import os
import scipy.io as scio
import numpy as np
from numpy.random import normal  # normal distribution
from common.Priority_Replay import Memory_PER
from common.ddpg_PER import DDPG

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.Memory = Memory_PER(args)
        self.Brain = DDPG(args)
        # configuration
        self.episode_num = args.max_episodes
        self.episode_step = args.episode_steps
        self.save_path = self.args.save_dir+'/'+self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path+'/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)
    
    def seperate_transitions(self, transition):
        s_dim = self.args.obs_shape
        a_dim = self.args.action_shape
        s = transition[:, :s_dim]
        a = transition[:, s_dim:s_dim+a_dim]
        r = transition[:, s_dim+a_dim:s_dim+a_dim+1]
        s_next = transition[:, s_dim+a_dim+1:s_dim+a_dim+1+s_dim]
        return s, a, r, s_next
    
    def run(self):
        average_reward = []  # average_reward of each episode
        noise_decrease = False
        noise_rate = self.args.noise_rate
        DONE = {}
        c_loss_average = []
        a_loss_average = []
        fuel = []
        lr_recorder = {'lra': [], 'lrc': []}
        for episode in tqdm(range(self.episode_num)):
            state = self.env.reset()  # reset the environment
            if noise_decrease:
                noise_rate *= self.args.noise_discount_rate
            setp_reward = []
            c_loss_one_episode = []
            a_loss_one_episode = []
            # data being saved in .mat
            episode_info_ACC = {'spd': [], 'acc': [], 'distance': [], 'collision': [],
                                'TTC': [], 'r_jerk': [], 'r_speed': [], 'r_safe': [],
                                'ACC_reward': [], 'jerk_value': [], 'follow_x': []}
            episode_info_EMS = {'P_fc': [], 'P_fce': [], 'P_dcdc': [],
                                'fce_eff': [], 'dcdc_eff': [], 'mot_eff': [],
                                'P_mot': [], 'T_mot': [], 'W_mot': [],
                                'h2_fcs': [], 'h2_batt': [], 'h2_equal': [],
                                'P_batt_out': [], 'V_batt': [], 'I': [], 'I_C': [], 'r': [],
                                'SOC': [], 'h2_cost': [], 'soc_cost': [], 'P_FCS_cost': [],
                                'EMS_reward': []}
                                
            for episode_step in range(self.episode_step):
                with torch.no_grad():
                    raw_action = self.Brain.choose_action(state)
                action = np.clip(normal(raw_action, noise_rate), -1, 1)
                state_next, reward, done, info_ACC, info_EMS = self.env.step(action)
                transition = np.concatenate((state, action, np.array([reward]), state_next))
                self.Memory.store(transition)
                if any(done):
                    # print('failure in step %d of episode %d'%(episode_step, episode))
                    if episode not in DONE.keys():
                        DONE.update({episode: episode_step})
                    # break
                state = state_next
                # save data
                for key in episode_info_ACC.keys():
                    episode_info_ACC[key].append(info_ACC[key])
                for key in episode_info_EMS.keys():
                    episode_info_EMS[key].append(info_EMS[key])
                setp_reward.append(reward)
                # learn
                if self.Memory.currentSize >= 10*self.args.batch_size:
                    noise_decrease = True
                    # s, a, r, s_next = self.Memory.sample()
                    tree_index, transitions, ISWeights = self.Memory.sample()
                    s, a, r, s_next = self.seperate_transitions(transitions)
                    td_error_up = self.Brain.train(s, a, r, s_next, ISWeights)
                    c_loss_one_episode.append(self.Brain.c_loss)
                    a_loss_one_episode.append(self.Brain.a_loss)
                    self.Memory.batch_update(tree_index, td_error_up)
                # save data in .mat
                if episode_step+1 == self.episode_step:
                    datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                    
                    colli_times = int(sum(episode_info_ACC['collision']))
                    episode_info_ACC.update({'colli_times': colli_times})
                    f_travel = episode_info_ACC['follow_x'][-1]/1000
                    
                    h2_equal = float(sum(episode_info_EMS['h2_equal']))
                    h2_g_100km = h2_equal/f_travel*100
                    episode_info_EMS.update({'h2_g_100km': h2_g_100km})
                    SOC = episode_info_EMS['SOC'][-1]
                    fuel.append(h2_g_100km)

                    print('episode %d: f_travel %.3fkm, '
                          'colli_times %d, SOC %.3f, h2_g_100km %.3fg'
                            % (episode, f_travel, colli_times, SOC, h2_g_100km))
                    
                    episode_info_ACC.update(episode_info_EMS)
                    scio.savemat(datadir, mdict=episode_info_ACC)
                    self.Brain.save_model(episode)
            # lr scheduler
            lra, lrc = self.Brain.lr_scheduler()
            lr_recorder['lra'].append(lra)
            lr_recorder['lrc'].append(lrc)
            # save loss data
            c_loss_mean = np.mean(c_loss_one_episode)
            a_loss_mean = np.mean(a_loss_one_episode)
            c_loss_average.append(c_loss_mean)
            a_loss_average.append(a_loss_mean)
            # save reward
            ep_r_mean = np.mean(setp_reward)
            average_reward.append(ep_r_mean)
            # print
            print('episode %d: c_loss %.3f, a_loss %.3f, ep_r %.3f'
                  % (episode, c_loss_mean, a_loss_mean, ep_r_mean))
        
        scio.savemat(self.save_path+'/c_loss_average.mat', mdict={'c_loss': c_loss_average})
        scio.savemat(self.save_path+'/a_loss_average.mat', mdict={'a_loss': a_loss_average})
        scio.savemat(self.save_path+'/average_reward.mat', mdict={'ep_r': average_reward})
        scio.savemat(self.save_path+'/lr_recorder.mat', mdict=lr_recorder)
        scio.savemat(self.save_path+'/fuel.mat', mdict={'fuel': fuel})
        
        print('buffer counter:', self.Memory.counter)
        print('buffer current size:', self.Memory.currentSize)
        print('replay ratio: %.3f'%(self.Memory.counter/self.Memory.currentSize))
        print('done:', DONE)