# fuel cell engine, DC/DC converter, and motor model
# and health model of FCE

import math
from scipy.interpolate import interp1d, interp2d
import scipy.io as scio


class FCHEV_SOH:
    def __init__(self):
        self.simtime = 1.0        # length of a time step
        # fuel cell
        self.motor_max_power = 200000  # W
        self.P_FC_max = 60.0   # kW
        self.P_FC_high = 0.9 * self.P_FC_max        # 已有论文中的说法是82开
        self.P_FC_low = 0.1 * self.P_FC_max
        self.P_FC_off = 0.01 * self.P_FC_max
        self.P_FC_old = 0       # power of last time step
        self._func_init()
        # power battery
        self._func_init_batt()
        self.Q_batt = 176  # Ah
        self.C_batt = 111.5     # kWh, 176 Ah * 633.54 V / 1000 = 111.5 kWh
        self.I_max = 20*self.Q_batt  # Battery current limitation
    
    # fuel cell model
    def _func_init(self):
        data_dir = "E:/SEU2/Program2/ACC_EMS_FCHEV/common/data/"  
        # fuel cell engine
        data = scio.loadmat(data_dir+'P_fc.mat')
        P_fc = data['P_fc'][0]   # attention: kW
        data = scio.loadmat(data_dir+'P_fce.mat')
        P_fce = data['P_fce'][0]   # attention: kW
        data = scio.loadmat(data_dir+'h2_consumption.mat')
        h2_consumption = data['h2_consumption'][0]
        data = scio.loadmat(data_dir+'fce_eff.mat')
        fce_eff = data['fce_eff'][0]
        self.P_fce = interp1d(P_fc, P_fce, kind='linear', fill_value='extrapolate')
        self.fce_eff = interp1d(P_fc, fce_eff, kind='linear', fill_value='extrapolate')
        self.h2_consumption = interp1d(P_fc, h2_consumption, kind='linear', fill_value='extrapolate')
        # 电池功率到等效氢气消耗量的转换系数，动态变化
        convert_coefficient_data = scio.loadmat('./data/conversion.mat')
        convert_coefficient = convert_coefficient_data['conversion'][0]
        p_fcs_data = scio.loadmat('./data/P_fc_conv.mat')
        p_fcs = p_fcs_data['P_fc_conv'][0]        # kW 60-point
        self.h2_conv_coef = interp1d(p_fcs, convert_coefficient, kind='linear', fill_value='extrapolate')
        
        # motor
        data = scio.loadmat(data_dir+'mot_eff.mat')
        motor_eff = data['mot_eff']
        data = scio.loadmat(data_dir+'W_mot.mat')
        motor_spd = data['W_mot'][0]
        data = scio.loadmat(data_dir+'T_mot.mat')
        motor_trq = data['T_mot'][0]
        self.mot_eff_map = interp2d(motor_spd, motor_trq, motor_eff)
       
        data = scio.loadmat(data_dir+'mot_trq_min.mat')
        mot_trq_min = data['new']
        self.mot_trq_min = interp1d(mot_trq_min[:, 0], mot_trq_min[:, 1], kind='linear', fill_value='extrapolate')
        
        data = scio.loadmat(data_dir+'mot_trq_max.mat')
        mot_trq_max = data['new']
        self.mot_trq_max = interp1d(mot_trq_max[:, 0], mot_trq_max[:, 1], kind='linear', fill_value='extrapolate')
        
        # DC/DC
        data = scio.loadmat(data_dir+'e_dcdc.mat')
        DCDC_eff = data['e_dcdc'][0]
        data = scio.loadmat(data_dir+'P_fce.mat')
        P_DCDC = data['P_fce'][0]      # kW
        self.DCDC_eff = interp1d(P_DCDC, DCDC_eff, kind='linear', fill_value='extrapolate')
        
    # power battery model
    def _func_init_batt(self):
        # SoC
        SOC_list = [0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1]
        # Discharging resistance
        R_dis = [1.005333333, 1.005333333, 0.901333333, 0.8, 0.746666667, 0.733333333,
                 0.714666667, 0.717333333, 0.728, 0.754666667, 0.794666667, 0.832, 0.832]  # ohm
        # Charging resistance
        R_chg = [0.626666667, 0.626666667, 0.586666667, 0.546666667, 0.528, 0.528,
                 0.522666667, 0.528, 0.525333333, 0.541333333, 0.544, 0.544, 0.544]
        # Open circuit voltage
        V_oc = [578.96, 578.96, 592.376, 600.848, 606.984, 611.672, 615.272,
                618.352, 621.328, 624.624, 633.144, 669.16, 669.16]
        self.R_dis_func = interp1d(SOC_list, R_dis, kind='linear', fill_value='extrapolate')
        self.R_chg_func = interp1d(SOC_list, R_chg, kind='linear', fill_value='extrapolate')
        self.V_func = interp1d(SOC_list, V_oc, kind='linear', fill_value='extrapolate')
    
    def _FCE_is_ON(self, P_FCE):
        return P_FCE >= self.P_FC_off
    
    def get_h2_conv_coef(self, P_FCS):
        return self.h2_conv_coef(P_FCS).tolist()
    
    @staticmethod
    def T_W_axle(car_spd, car_acc):
        """计算传动轴/电机轴转速和扭矩, m/s, m/s2, 均采用国际标准单位制"""
        # parameters of car
        wheel_radius = 0.467  # 0.4671 m
        mass = 8400  # kg      # 14250 kg
        C_roll = 0.012       # 0.0085         # [0.012， 0.015]
        density_air = 1.226  # N*s2/m4
        area_frontal = 6.56  # 3.9 m2
        G = 9.81
        C_d = 0.55  # 0.65
        G_f = 6.2  # Final reducer ratio
        rads2rpm = 2*math.pi/60
        # calculate
        # W_axle = car_spd/wheel_radius*G_f  # 传动轴转速  r per second
        W_axle = car_spd/wheel_radius*G_f/rads2rpm      # 电机轴转速, rpm
        # torque        # Nm    电机轴
        T_axle = wheel_radius/G_f*(mass*car_acc+mass*G*C_roll+0.5*density_air*area_frontal*C_d*(car_spd**2))
        if T_axle >= 0:
            P_axle = rads2rpm*T_axle*W_axle/0.98     # in W  电机辅件效率0.98
        else:
            P_axle = rads2rpm*T_axle*W_axle*0.95     # 再生制动效率0.95
        return T_axle, W_axle, P_axle

    def run_motor(self, T_axle, W_axle, P_axle):
        T_mot = T_axle
        W_mot = W_axle
        T_mot_max = self.mot_trq_max(W_mot)
        T_mot_min = self.mot_trq_min(W_mot)
        if T_mot < T_mot_min:
            T_mot = T_mot_min
        if T_mot > T_mot_max:
            T_mot = T_mot_max
        mot_eff = self.mot_eff_map(W_mot, T_mot)
        mot_eff = mot_eff[0]
        rads2rpm = 2*math.pi/60
        if P_axle <= 0:
            # P_mot = P_axle*mot_eff
            P_mot = rads2rpm * T_mot * W_mot * mot_eff
        else:
            # P_mot = P_axle/mot_eff
            P_mot = rads2rpm*T_mot*W_mot/mot_eff
        return T_mot, W_mot, mot_eff, P_mot
        
    def run_fuel_cell(self, P_fc):
        P_fce = self.P_fce(P_fc).tolist()   # kW, from 0-dimension array to a float
        fce_eff = self.fce_eff(P_fc).tolist()
        h2_fcs = self.h2_consumption(P_fc).tolist()    # g/min
        h2_fcs /= 60                           # gram in one second
        dcdc_eff = self.DCDC_eff(P_fce).tolist()
        P_dcdc = dcdc_eff * P_fce                   # kW
        info = {'P_fc': P_fc, 'P_fce': P_fce, 'fce_eff': fce_eff, 'dcdc_eff': dcdc_eff,
                'h2_fcs': h2_fcs, 'P_dcdc': P_dcdc}
        return P_dcdc, h2_fcs, info
    
    def run_FC_SOH(self, P_fc):
        # degradation per second, in percentage
        # start-stop cycles
        if self._FCE_is_ON(self.P_FC_old) is False and self._FCE_is_ON(P_fc) is True:
            # d_s_s = self.simtime * 1.96 * 1e-3 / 3600
            d_s_s = self.simtime*1.96
        else:
            d_s_s = 0
        # low-power load condition
        if self._FCE_is_ON(P_fc) is True and P_fc < self.P_FC_low:
            # d_low = self.simtime * 1.26 * 1e-3 / 3600
            d_low = self.simtime*1.26
        else:
            d_low = 0
        # high-power load condition
        if P_fc >= self.P_FC_high:
            # d_high = self.simtime * 1.47 * 1e-3 / 3600      # 1e-3/3600 == 2.8e-7
            d_high = self.simtime*1.47
        else:
            d_high = 0
        # load changing cycles
        # d_l_c = 5.93*1e-5*abs(P_fc-self.P_FC_old)/(self.P_FC_high-self.P_FC_low)
        d_l_c = 5.93*abs(P_fc-self.P_FC_old)/(self.P_FC_high-self.P_FC_low)
        # d_l_c = self.simtime * 3.32 * 1e-3 / 3600
        
        # total degradation in current time step
        De_i = (d_s_s + d_low + d_high) * 1e-3 / 3600 + d_l_c*1e-5   # in percentage
        FCS_De = De_i / 100      # 10% is the allowed performance degradation percentages\
        out_info = {'d_s_s': d_s_s, 'd_low': d_low, 'd_high': d_high, 'd_l_c': d_l_c,
                    'FCS_De': FCS_De}
        self.P_FC_old = P_fc
        return FCS_De, out_info
    
    def run_power_battery(self, P_batt, SOC):
        # P_batt in W
        fail = False
        # Battery efficiency
        e_batt = (P_batt >= 0)+(P_batt < 0)*0.95
        # Battery internal resistance, on package level
        r = (P_batt >= 0)*self.R_dis_func(SOC)+(P_batt < 0)*self.R_chg_func(SOC)
        # r *= 0.5  # line 71
        # Battery voltage
        V_batt = float(self.V_func(SOC))  # OCV
        V_batt *= 1.03  # line 76
        # Battery current
        if V_batt**2 < 4*r*P_batt:
            # I_batt = e_batt*V_batt/(2*r)
            I_batt = P_batt/e_batt/V_batt
        else:
            I_batt = e_batt*(V_batt-math.sqrt(V_batt**2-4*r*P_batt))/(2*r)
        if I_batt > self.I_max:
            I_batt = self.I_max
        if I_batt < -self.I_max:
            I_batt = -self.I_max
        I_c = abs(I_batt/self.Q_batt)  # c-rate
        SOC_delta = -I_batt/self.Q_batt/3600
        SOC_new = SOC_delta+SOC
        if SOC_new >= 1:
            SOC_new = 1.0
            fail = True
        if SOC_new <= 0:
            SOC_new = 0.0
            fail = True
        # battery output power
        batt_pow = I_batt*V_batt  # in W        # total power including auxiliaries
        out_info = {'SOC': float(SOC_new), 'r': r, 'V_batt': V_batt,
                    'I': I_batt, 'I_C': I_c,
                    'P_batt_need': P_batt/1000, 'P_batt_out': batt_pow/1000}
        done = fail
        return SOC_delta, SOC_new, done, out_info
    

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    fchev = FCHEV_SOH()
    pfcs = np.linspace(0, 60, 120)
    yconv = []
    for ii in pfcs:
        y = fchev.get_h2_conv_coef(ii)
        yconv.append(y)

    plt.plot(pfcs, yconv)
    plt.show()
    
    
