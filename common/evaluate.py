from tqdm import tqdm           # 进度条
import os
import torch
import numpy as np
import time
import scipy.io as scio
from common.ddpg import DDPG

class Evaluator:
    def __init__(self, args, env):
        self.args = args
        self.eva_episode = args.evaluate_episode
        self.episode_step = args.episode_steps
        self.env = env
        self.brain = DDPG(args)
        
        self.save_path = self.args.eva_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path+'/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)

    def evaluate(self):
        average_reward = []  # average_reward of each episode
        fuel = []
        electric = []
        money = []
        for episode in tqdm(range(self.eva_episode)):
            state = self.env.reset()  # reset the environment

            setp_reward = []
            # data being saved in .mat
            episode_info_ACC = {'spd': [], 'acc': [], 'distance': [], 'collision': [],
                                'TTC': [], 'r_jerk': [], 'r_speed': [], 'r_safe': [],
                                'ACC_reward': [], 'jerk_value': [], 'follow_x': [], 'acc_leading': []}
            episode_info_EMS = {'P_fc': [], 'P_fce': [], 'P_dcdc': [],
                                'fce_eff': [], 'dcdc_eff': [], 'mot_eff': [],
                                'P_mot': [], 'T_mot': [], 'W_mot': [],
                                'h2_fcs': [], 'h2_batt': [], 'h2_equal': [],
                                'V_batt': [], 'I': [], 'I_C': [], 'r': [],
                                'SOC': [], 'h2_cost': [], 'soc_cost': [], 'P_FCS_cost': [],
                                'EMS_reward': [], 'P_batt_need': [], 'h2_money': [],
                                'elec_money_spent': [], 'elec_money_revised': [],
                                'total_money_spent': [], 'total_money_revised': []}
            start_time = time.time()
            for episode_step in range(self.episode_step):
                with torch.no_grad():
                    raw_action = self.brain.choose_action(state)
                action = raw_action
                state_next, reward, done, info_ACC, info_EMS = self.env.step(action)
                state = state_next
                # save data
                for key in episode_info_ACC.keys():
                    episode_info_ACC[key].append(info_ACC[key])
                for key in episode_info_EMS.keys():
                    episode_info_EMS[key].append(info_EMS[key])
                setp_reward.append(reward)
                # save data in .mat
                if episode_step+1 == self.episode_step:
                    datadir = self.save_path_episode+'/data_ep%d.mat'%episode
    
                    colli_times = int(sum(episode_info_ACC['collision']))
                    episode_info_ACC.update({'colli_times': colli_times})
                    f_travel = episode_info_ACC['follow_x'][-1]/1000
    
                    # h2_equal = float(sum(episode_info_EMS['h2_equal']))
                    h2_equal = float(sum(episode_info_EMS['h2_fcs']))
                    h2_g_100km = h2_equal/f_travel*100
                    episode_info_EMS.update({'h2_g_100km': h2_g_100km})
                    fuel.append(h2_g_100km)
    
                    SOC_0 = episode_info_EMS['SOC'][0]
                    SOC_end = episode_info_EMS['SOC'][-1]
                    elec_100km = 111.5*(SOC_0-SOC_end)/f_travel*100
                    electric.append(elec_100km)
                    episode_info_EMS.update({'elec_kWh_100km': elec_100km})
    
                    money_epi = float(sum(episode_info_EMS['total_money_revised']))
                    money_100km = money_epi/f_travel*100
                    episode_info_EMS.update({'money_100km': money_100km})
                    money.append(money_100km)
    
                    print('episode %d: f_travel %.3fkm, colli_times %d, SOC %.3f'
                          % (episode, f_travel, colli_times, SOC_end))
                    print('episode %d: h2_100km %.3fg, electric_100km %.3fkWh, money_100km ￥%.3f'
                          % (episode, h2_g_100km, elec_100km, money_100km))
    
                    episode_info_ACC.update(episode_info_EMS)
                    scio.savemat(datadir, mdict=episode_info_ACC)
            end_time = time.time()
            spent_time = end_time - start_time
            # save reward
            ep_r_mean = np.mean(setp_reward)
            average_reward.append(ep_r_mean)
            # print
            print('episode %d: ep_r %.3f, time spent: %.4fs'
                  % (episode, ep_r_mean, spent_time))
    
        scio.savemat(self.save_path+'/average_reward.mat', mdict={'ep_r': average_reward})
        scio.savemat(self.save_path+'/fuel.mat', mdict={'fuel': fuel})
        scio.savemat(self.save_path+'/electric.mat', mdict={'electric': electric})
        scio.savemat(self.save_path+'/money.mat', mdict={'money': money})
