import numpy as np
from common.utils import get_driving_cycle, get_acc_limit
from common.arguments import get_args

args = get_args()
SPEED_LIST = get_driving_cycle(cycle_name=args.scenario_name)  # speed of leading car
ACC_LIST = get_acc_limit(SPEED_LIST, args.scenario_name, output_max_min=False)  # acceleration of leading car

class ACC:
    """
    ACC model
    """
    def __init__(self):
        self.obs_num = 5
        # observation: np.array([follow_speed, follow_acc, distance, leading_speed, leading_acc])
        self.action_num = 1  # action: [follow_acc]: [-1, 1]+noise
        self.leading_speed = 0.  # speed of leading car      m/s
        self.follow_speed = 0.  # speed of following car
        self.relative_speed = 0.
        self.leading_acc = 0.  # acceleration of leading car   m/s2
        self.follow_acc = 0.  # output by actor network
        self.jerk = 0.
        self.leading_x = 0.  # how far the leading car has traveled   m
        self.follow_x = 0.
        self.car_length = 12  # unit: m
        self.speed_limit = 69/3.6  # unit: m/s,
        self.distance = 0.  # distance between two cars
        # self.distance_optimal = 0.  # the optimal of distance for following
        self.distance_min = 3.  # minimum distance for safety
        self.distance_max = 10.
        self.acc_max = 0.7  # m/s2, this is a limit for output action
        self.acc_min = -1.5
        self.acc_limit = max(abs(self.acc_max), abs(self.acc_min))
        self.train_counter = 0  # counter of training times
        
        self.speed_penalty = False
        self.info = {}
        self.done = False
    
    def reset_obs(self):
        """reset observation"""
        self.info = {}
        self.done = False
        self.train_counter = 0
        self.jerk = 0
        self.speed_penalty = False
        self.distance_min = 3.
        self.distance_max = 10.
        self.follow_speed = 0.
        self.follow_acc = 0.
        self.leading_speed = 0.
        self.leading_acc = 0.
        self.leading_x = 35.
        self.follow_x = 0.
        self.distance = self.leading_x-self.follow_x-self.car_length
        self.relative_speed = self.leading_speed-self.follow_speed
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # maximum normalization
        obs[0] = self.follow_speed/self.speed_limit
        obs[1] = self.follow_acc/self.acc_limit
        obs[2] = self.distance/1000
        obs[3] = self.leading_speed/self.speed_limit
        obs[4] = self.leading_acc/self.acc_limit
        return obs
    
    def execute(self, action):
        # action = action[0]  # acceleration of following car
        action *= abs(self.acc_limit)
        if action > self.acc_max:
            action = self.acc_max
        if action < self.acc_min:
            action = self.acc_min
        sim_time = 1  # simulation time interval
        # actual speed and acc considering speed limit
        spd_tmp = self.follow_speed+action*sim_time
        # if spd_tmp < 0 or spd_tmp > self.speed_limit:
        if spd_tmp > self.speed_limit:
            self.speed_penalty = True
        else:
            self.speed_penalty = False
        if spd_tmp < 0:
            spd_tmp = 0
        if spd_tmp > self.speed_limit:
            spd_tmp = self.speed_limit
        acc_tmp = (spd_tmp-self.follow_speed)/sim_time  # true acc
        # update distance
        self.leading_x += (sim_time*self.leading_speed)
        self.follow_x += (self.follow_speed*sim_time+0.5*acc_tmp*pow(sim_time, 2))
        self.distance = self.leading_x-self.follow_x-self.car_length
        # update speed, acceleration, jerk
        self.follow_speed = spd_tmp
        self.leading_speed = SPEED_LIST[self.train_counter]
        self.leading_acc = ACC_LIST[self.train_counter]
        self.train_counter += 1
        self.relative_speed = self.leading_speed-self.follow_speed
        self.jerk = acc_tmp-self.follow_acc
        self.follow_acc = acc_tmp
        # optimal and safe distance
        # self.distance_optimal = 15.336+95.9*np.arctanh(0.02*self.leading_speed-0.008)
        self.distance_min = 3+self.follow_speed*0.8+pow(self.follow_speed, 2)/(7.5*2)
        self.distance_max = 10+self.follow_speed+0.0825*(self.follow_speed**2)
        if self.distance_max > 160:
            self.distance_max = 160
        self.info.update({'spd': self.follow_speed, 'acc': self.follow_acc, 'distance': self.distance,
                          'leading_x': self.leading_x, 'follow_x': self.follow_x, 'acc_leading': self.leading_acc})
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = self.follow_speed/self.speed_limit
        obs[1] = self.follow_acc/self.acc_limit
        obs[2] = self.distance/1000
        obs[3] = self.leading_speed/self.speed_limit
        obs[4] = self.leading_acc/self.acc_limit
        return obs
    
    def get_reward(self):
        # speed
        if self.speed_penalty:
            r_speed = - self.speed_limit  # 120/3.6=33.33
        else:
            r_speed = 0
        # safety, version 2
        collision = 0
        if self.distance <= 0:
            # print('step %d: distance <= 0, distance: %.1fm, lead_x: %.1fm, follow_x: %.1fm, spd: %.1fm/s, acc: %.1f' %
            #       (self.train_counter, self.distance, self.leading_x, self.follow_x, self.follow_speed, self.follow_acc))
            r_safe = - self.speed_limit - self.follow_acc + self.distance
            collision = 1
        elif 0 < self.distance < self.distance_min:
            r_safe = - self.follow_speed
        elif self.distance > self.distance_max:
            r_safe = -abs(self.distance-self.distance_max)/4
        else:
            r_safe = 1      # 阶梯高度小一点
        TTC = - self.distance/(self.relative_speed+0.001)  # time to collision
        if TTC < 0:
            TTC = -0.1
        if TTC > 10:
            TTC = 10
        # comfort
        r_jerk = - abs(self.jerk)/(self.acc_max-self.acc_min)  # jeck: [-1, 0]
        # total reward
        if r_safe > 0:
            # r_jerk *= 2   # v1
            # r_jerk *= 5   # v2
            r_jerk *= 10.0  # from v3
        r_speed *= 0
        reward = r_speed+r_safe+r_jerk
        self.info.update({'collision': collision, 'TTC': TTC, 'r_jerk': r_jerk,
                          'r_speed': r_speed, 'jerk_value': self.jerk, 'r_safe': r_safe,
                          'ACC_reward': reward})
        return float(reward)
    
    def get_info(self):
        return self.info
    
    def get_done(self):
        return self.done
