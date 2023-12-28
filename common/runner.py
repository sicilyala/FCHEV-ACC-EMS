import torch
from tqdm import tqdm
import os
import scipy.io as scio
import numpy as np
from numpy.random import normal  # normal distribution
from common.memory import MemoryBuffer
from common.ddpg import DDPG


def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Random seeds have been set to %d!" % seed)


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.buffer = MemoryBuffer(args)
        self.brain = DDPG(args)
        # configuration
        self.episode_num = args.max_episodes
        self.episode_step = args.episode_steps
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path + '/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)

    def run(self):
        average_reward = []  # average_reward of each episode
        noise_decrease = False
        noise_rate = self.args.noise_rate
        DONE = {}
        c_loss_average = []
        a_loss_average = []
        H2_FCS = []
        H2_eq = []
        FCS_SOH = []
        electric = []
        money = []
        lr_recorder = {'lra': [], 'lrc': []}
        for episode in tqdm(range(self.episode_num)):
            state = self.env.reset()  # reset the environment
            if noise_decrease:
                noise_rate *= self.args.noise_discount_rate
            step_reward = []
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
                                'V_batt': [], 'I': [], 'I_C': [], 'r': [], 'h2_conv_coef': [],
                                'SOC': [], 'soc_cost': [], 'FCS_SOH': [], 'EMS_money': [],
                                'EMS_reward': [], 'P_batt_need': [], 'h2_money': [], 'FCS_money': []}

            for episode_step in range(self.episode_step):
                with torch.no_grad():
                    raw_action = self.brain.choose_action(state)
                action = np.clip(normal(raw_action, noise_rate), -1, 1)
                state_next, reward, done, info_ACC, info_EMS = self.env.step(action)
                self.buffer.add(state, action, reward, state_next)
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
                step_reward.append(reward)
                # learn
                if self.buffer.currentSize >= 10 * self.args.batch_size:
                    noise_decrease = True
                    s, a, r, s_next = self.buffer.sample()
                    self.brain.train(s, a, r, s_next)
                    c_loss_one_episode.append(self.brain.c_loss)
                    a_loss_one_episode.append(self.brain.a_loss)
                # save data in .mat
                if episode_step + 1 == self.episode_step:
                    datadir = self.save_path_episode + '/data_ep%d.mat' % episode

                    colli_times = int(sum(episode_info_ACC['collision']))
                    episode_info_ACC.update({'colli_times': colli_times})
                    f_travel = episode_info_ACC['follow_x'][-1] / 1000  # km

                    h2_equal = float(sum(episode_info_EMS['h2_equal']))
                    h2_eq_100km = h2_equal / f_travel * 100
                    episode_info_EMS.update({'h2_eq_100km': h2_eq_100km})
                    H2_eq.append(h2_eq_100km)

                    h2_fcs = float(sum(episode_info_EMS['h2_fcs']))
                    h2_fcs_100km = h2_fcs / f_travel * 100
                    episode_info_EMS.update({'h2_fcs_100km': h2_fcs_100km})
                    H2_FCS.append(h2_fcs_100km)

                    fcs_soh_end = episode_info_EMS['FCS_SOH'][-1]
                    FCS_SOH.append(fcs_soh_end)

                    SOC_0 = episode_info_EMS['SOC'][0]
                    SOC_end = episode_info_EMS['SOC'][-1]
                    elec_100km = 111.5 * (SOC_0 - SOC_end) / f_travel * 100
                    electric.append(elec_100km)
                    episode_info_EMS.update({'elec_kWh_100km': elec_100km})

                    money_epi = float(sum(episode_info_EMS['EMS_money']))
                    money_100km = money_epi / f_travel * 100
                    episode_info_EMS.update({'money_100km': money_100km})
                    money.append(money_100km)

                    print('epi %d: f_travel %.3fkm, colli_times %d, SOC %.3f, FCS-SOH %.6f'
                          % (episode, f_travel, colli_times, SOC_end, fcs_soh_end))
                    print('epi %d: h2_fcs_100km %.3fg, h2_eq_100km %.3fg, electric_100km %.3fkWh, money_100km '
                          '￥%.3f '
                          % (episode, h2_fcs_100km, h2_eq_100km, elec_100km, money_100km))

                    episode_info_ACC.update(episode_info_EMS)
                    scio.savemat(datadir, mdict=episode_info_ACC)
                    self.brain.save_model(episode)
            # lr scheduler
            lra, lrc = self.brain.lr_scheduler()
            lr_recorder['lra'].append(lra)
            lr_recorder['lrc'].append(lrc)
            # save loss data
            c_loss_mean = np.mean(c_loss_one_episode)
            a_loss_mean = np.mean(a_loss_one_episode)
            c_loss_average.append(c_loss_mean)
            a_loss_average.append(a_loss_mean)
            # save reward
            ep_r_mean = np.mean(step_reward)
            average_reward.append(ep_r_mean)
            # print
            print('epi %d: c_loss %.3f, a_loss %.3f, ep_r %.3f \n'
                  % (episode, c_loss_mean, a_loss_mean, ep_r_mean))

        scio.savemat(self.save_path + '/c_loss_average.mat', mdict={'c_loss': c_loss_average})
        scio.savemat(self.save_path + '/a_loss_average.mat', mdict={'a_loss': a_loss_average})
        scio.savemat(self.save_path + '/average_reward.mat', mdict={'ep_r': average_reward})
        scio.savemat(self.save_path + '/lr_recorder.mat', mdict=lr_recorder)
        scio.savemat(self.save_path + '/H2_FCS.mat', mdict={'H2_FCS': H2_FCS})
        scio.savemat(self.save_path + '/electric.mat', mdict={'electric': electric})
        scio.savemat(self.save_path + '/H2_eq.mat', mdict={'H2_eq': H2_eq})
        scio.savemat(self.save_path + '/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SOH})

        print('buffer counter:', self.buffer.counter)
        print('buffer current size:', self.buffer.currentSize)
        print('replay ratio: %.3f' % (self.buffer.counter / self.buffer.currentSize))
        print('done:', DONE)
