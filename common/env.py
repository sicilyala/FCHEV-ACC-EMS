import numpy as np
from common.agentACC import ACC
from common.agentEMS import EMS
from common.utils import get_driving_cycle


class Env:
    def __init__(self, w_acc, w_ems):
        self.ACC = ACC()
        self.EMS = EMS()
        self.obs_num = int(self.ACC.obs_num+self.EMS.obs_num)
        self.action_num = int(self.ACC.action_num+self.EMS.action_num)
        self.w_ACC = w_acc
        self.w_EMS = w_ems
    
    def reset(self):
        ob1 = self.ACC.reset_obs()
        ob2 = self.EMS.reset_obs()
        obs = np.concatenate((ob1, ob2))
        return obs
    
    def step(self, action):
        ob1 = self.ACC.execute(action[0])
        car_spd = self.ACC.follow_speed
        car_acc = self.ACC.follow_acc
        ob2 = self.EMS.execute(action[1], car_spd, car_acc)
        obs = np.concatenate((ob1, ob2))
        
        rwd_ACC = self.ACC.get_reward()
        rwd_EMS = self.EMS.get_reward()
        reward = self.w_ACC*rwd_ACC + self.w_EMS*rwd_EMS
        
        done = [self.ACC.get_done(), self.EMS.get_done()]
        info_ACC = self.ACC.get_info()
        info_EMS = self.EMS.get_info()
        return obs, reward, done, info_ACC, info_EMS


def make_env(args):
    w_ems = args.w_EMS
    w_acc = args.w_ACC
    env = Env(w_acc, w_ems)
    args.obs_shape = env.obs_num
    args.action_shape = env.action_num
    speed_list = get_driving_cycle(cycle_name=args.scenario_name)
    args.episode_steps = len(speed_list)  # cycle length, be equal to args.episode_steps
    args.trip_length = sum(speed_list)/1000
    return env, args
