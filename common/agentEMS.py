import numpy as np
from common.FCHEV_SOH import FCHEV_SOH


class EMS:
    def __init__(self):
        self.done = False
        self.info = {}
        self.FCHEV = FCHEV_SOH()
        self.obs_num = 3  # soc, fcs-soh, P_mot
        self.action_num = 1  # P_FC
        self.SOC_init = 0.5
        self.SOC_target = self.SOC_init
        self.SOC = self.SOC_init
        self.P_mot_max = self.FCHEV.motor_max_power  # W
        self.P_mot = 0
        self.h2_fcs = 0     # g
        self.P_batt = 0
        self.P_FCS = 0
        self.FCS_SOH = 1.0
        self.dSOH_FCS = 0
        self.SOC_delta = 0
    
    def reset_obs(self):
        self.SOC = self.SOC_init
        self.FCS_SOH = 1.0
        self.P_mot = 0
        self.P_FCS = 0
        self.done = False
        self.info = {}
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = self.SOC
        obs[1] = self.FCS_SOH
        obs[2] = self.P_mot / self.P_mot_max
        return obs
    
    def execute(self, action, car_spd, car_acc):
        self.P_FCS = abs(action) * self.FCHEV.P_FC_max     # kW
        T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(car_spd, car_acc)
        T_mot, W_mot, mot_eff, self.P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)  # W
        P_dcdc, self.h2_fcs, info_fcs = self.FCHEV.run_fuel_cell(self.P_FCS)      # kW
        self.dSOH_FCS, info_fcs_soh = self.FCHEV.run_FC_SOH(self.P_FCS)
        self.FCS_SOH -= self.dSOH_FCS
        self.P_batt = self.P_mot - P_dcdc*1000        # W
        self.SOC_delta, self.SOC, self.done, info_batt = self.FCHEV.run_power_battery(self.P_batt, self.SOC)
        
        self.info = {}
        self.info.update({'T_axle': T_axle, 'W_axle': W_axle, 'P_axle': P_axle/1000,
                          'T_mot': T_mot, 'W_mot': W_mot, 'mot_eff': mot_eff,
                          'P_mot': self.P_mot/1000, 'FCS_SOH': self.FCS_SOH})
        self.info.update(info_fcs)
        self.info.update(info_batt)
        self.info.update(info_fcs_soh)
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = self.SOC
        obs[1] = self.FCS_SOH
        obs[2] = self.P_mot / self.P_mot_max
        return obs
    
    def get_reward(self):
        # H2
        varying_h2_conv_coef = self.FCHEV.get_h2_conv_coef(self.P_FCS)
        best_h2_conv_coef = 0.0164  # 取FCS效率最高点(0.5622)计算, 该系数为0.0164, 原为0.0167
        h2_conv_coef = varying_h2_conv_coef
        if self.P_batt > 0:
            h2_batt = self.P_batt / 1000 * h2_conv_coef      # g in one second
        else:
            h2_batt = 0
        h2_equal = float(self.h2_fcs + h2_batt)
        h2_price = 55/1000      # ￥ per g
        h2_money = h2_price*self.h2_fcs     # or *h2_equal
        eq_h2_money = h2_price * h2_equal
        
        # fcs-soh cost
        FCS_price = 300000
        # FCS_price = 0
        FCS_money = FCS_price * self.dSOH_FCS
        
        # SOC cost
        w_soc = 20.0
        soc_cost = w_soc * abs(self.SOC - self.SOC_target)
        
        # reward function
        reward = -(eq_h2_money + FCS_money + soc_cost)
        reward = float(reward)
        EMS_money = h2_money + FCS_money
        self.info.update({'EMS_reward': reward, 'h2_money': h2_money, 'FCS_money': FCS_money,
                          'h2_equal': h2_equal, 'h2_batt': h2_batt, 'soc_cost': soc_cost,
                          'EMS_money': EMS_money, 'eq_h2_money': eq_h2_money, 'h2_conv_coef': h2_conv_coef})
        return reward

    def get_info(self):
        return self.info

    def get_done(self):
        return self.done
    