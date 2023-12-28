# calculate the conversion coefficient of batter power to equivalent hydrogen

import numpy as np
import scipy.io as scio
from common.FCHEV import FCHEV_model

if __name__ == '__main__':
    FCB = FCHEV_model()
    P_fc = np.linspace(0, 60, 61, dtype=np.double)
    print(P_fc)
    conversion = np.zeros(P_fc.shape)
    for i, p in enumerate(P_fc):
        p_fce = FCB.P_fce(p).tolist()
        eff_fce = FCB.fce_eff(p).tolist()
        h2 = FCB.h2_consumption(p).tolist()
        h2 /= 60
        power_into_dcdc = p_fce*eff_fce
        dcdc_eff = FCB.DCDC_eff(power_into_dcdc).tolist()
        P_dcdc = dcdc_eff * p_fce
        if P_dcdc == 0:
            conversion[i] = 0
        else:
            conversion[i] = 0.95 * h2 / P_dcdc
    print(conversion)
    scio.savemat("E:/SEU2/CVCI2022/Program CVCI/common/data/P_fc_conv.mat", mdict={'P_fc_conv': P_fc})
    scio.savemat("E:/SEU2/CVCI2022/Program CVCI/common/data/conversion.mat", mdict={'conversion': conversion})