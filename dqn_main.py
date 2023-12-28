from tqdm import tqdm
import numpy as np
import torch
import os
import scipy.io as scio
from common.arguments import get_args
from common.dqn_model import DQN_model, Memory
from common.agentEMS import EMS
from common.env import make_DQN_env


def main_EMS(args):
    save_path = args.save_dir + '_DQN' + '/' + args.scenario_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_episode = save_path + '/episode_data'
    if not os.path.exists(save_path_episode):
        os.makedirs(save_path_episode)

    # action_space = [-5.0, 0, 5.0]
    action_num = 61
    action_space = np.linspace(0, 1, action_num, dtype=np.float32)
    ems = EMS()
    memory = Memory(memory_size=args.buffer_size, batch_size=args.batch_size)
    dqn_agent = DQN_model(args, s_dim=ems.obs_num, a_dim=action_num)

    average_reward = []  # average_reward of each episode
    DONE = {}
    H2_FCS = []
    H2_eq = []
    FCS_SOH = []
    electric = []
    money = []

    average_loss = []
    lr = []
    initial_epsilon = 1.0
    finial_epsilon = 0.01
    epsilon_decent = (initial_epsilon - finial_epsilon) / 450
    epsilon = initial_epsilon

    data = scio.loadmat("E:/cwqaq/ACC_EMS_FCHEV_PAPER/ACC_EMS_FCHEV/result1/WVUCITY_WVUINTER_v2_H2conv_w10"
                        "/episode_data/data_ep499.mat")

    SPD_LIST = data['spd'][0]
    ACC_LIST = data['acc'][0]
    episode_step_num = SPD_LIST.shape[0]  #
    MILE = np.sum(SPD_LIST) / 1000      # km
    print('episode_step_num: %d' % episode_step_num)
    print('mileage: %.3fkm' % MILE)

    for episode in tqdm(range(args.max_episodes)):
        state = ems.reset_obs()  # ndarray

        rewards = []
        loss = []
        episode_info_EMS = {'P_fc': [], 'P_fce': [], 'P_dcdc': [],
                                'fce_eff': [], 'dcdc_eff': [], 'mot_eff': [],
                                'P_mot': [], 'T_mot': [], 'W_mot': [],
                                'h2_fcs': [], 'h2_batt': [], 'h2_equal': [],
                                'V_batt': [], 'I': [], 'I_C': [], 'r': [], 'h2_conv_coef': [],
                                'SOC': [], 'soc_cost': [], 'FCS_SOH': [], 'EMS_money': [],
                                'EMS_reward': [], 'P_batt_need': [], 'h2_money': [], 'FCS_money': []}
        for episode_step in range(episode_step_num):
            with torch.no_grad():
                action_id, epsilon_using = dqn_agent.e_greedy_action(state, epsilon)
            action = action_space[action_id]  # float
            spd = SPD_LIST[episode_step]
            acc = ACC_LIST[episode_step]
            next_state = ems.execute(action, spd, acc)
            reward = ems.get_reward()
            done = ems.get_done()
            info = ems.get_info()

            memory.store_transition(state, action_id, reward, next_state)
            state = next_state

            rewards.append(reward)
            if done:
                # print('SOC failure in step %d of episode %d'%(episode_step, episode))
                if episode not in DONE.keys():
                    DONE.update({episode: episode_step})
            for key in episode_info_EMS.keys():
                episode_info_EMS[key].append(info[key])

            if memory.current_size > 100 * args.batch_size:
                minibatch = memory.uniform_sample()
                dqn_agent.train(minibatch)
                loss.append(dqn_agent.loss)

            # end of an episode: sava model params, save data, print info
            if episode_step + 1 == args.episode_step:
                dqn_agent.save_model(episode)
                datadir = save_path_episode + '/data_ep%d.mat' % episode

                f_travel = MILE
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

                print('\n epi %d: f_travel %.3fkm, SOC %.3f, FCS-SOH %.6f'
                      % (episode, f_travel, SOC_end, fcs_soh_end))
                print('epi %d: h2_fcs_100km %.3fg, h2_eq_100km %.3fg, electric_100km %.3fkWh, money_100km '
                      '￥%.3f '
                      % (episode, h2_fcs_100km, h2_eq_100km, elec_100km, money_100km))

                scio.savemat(datadir, mdict=episode_info_EMS)

                mean_r = np.mean(rewards)
                mean_loss = np.mean(loss)
                average_reward.append(mean_r)
                average_loss.append(mean_loss)
                print('epi %d: reward %.6f, loss %.6f, epsilon %.6f'
                      % (episode, mean_r, mean_loss, epsilon_using))

        epsilon -= float(epsilon_decent)
        lr0 = dqn_agent.scheduler_lr.get_last_lr()[0]
        print('epi %d: lr %.6f\n' % (episode, lr0))
        lr.append(lr0)
        dqn_agent.scheduler_lr.step()

    scio.savemat(save_path + '/ep_r.mat', mdict={'ep_r': average_reward})
    scio.savemat(save_path + '/ep_loss.mat', mdict={'ep_loss': average_loss})
    scio.savemat(save_path + '/H2_FCS.mat', mdict={'H2_FCS': H2_FCS})
    scio.savemat(save_path + '/H2_eq.mat', mdict={'H2_eq': H2_eq})
    scio.savemat(save_path + '/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SOH})
    scio.savemat(save_path + '/electric.mat', mdict={'electric': electric})
    scio.savemat(save_path + '/money.mat', mdict={'money': money})
    scio.savemat(save_path + '/lr.mat', mdict={'lr': lr})

    print('buffer counter:', memory.counter)
    print('buffer current size:', memory.current_size)
    print('replay ratio: %.3f' % (memory.counter / memory.current_size) + '\n')
    print('done:', DONE)


if __name__ == '__main__':
    args = get_args()
    args = make_DQN_env(args)
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Random seeds have been set to %d!" % seed)

    print('cycle name: ', args.scenario_name)

    main_EMS(args)
