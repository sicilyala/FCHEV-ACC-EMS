clear;
SOC_list = [0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1];
R_dis = [1.005333333, 1.005333333, 0.901333333, 0.8, 0.746666667, 0.733333333,0.714666667, 0.717333333, 0.728, 0.754666667, 0.794666667, 0.832, 0.832]; 
R_chg = [0.626666667, 0.626666667, 0.586666667, 0.546666667, 0.528, 0.528,0.522666667, 0.528, 0.525333333, 0.541333333, 0.544, 0.544, 0.544];
V_oc = [578.96, 578.96, 592.376, 600.848, 606.984, 611.672, 615.272,618.352, 621.328, 624.624, 633.144, 669.16, 669.16];

xlabel('SoC','FontName','Times New Roman','FontSize',16)
yyaxis left
plot(SOC_list, R_chg,SOC_list, R_dis,'LineWidth',2)
ylim([0.5,1.1])
ylabel('Internal Resistance (Ω)','FontName','Times New Roman','FontSize',16)

yyaxis right
plot(SOC_list, V_oc,'LineWidth',2)
ylabel('Battery Voltage (V)','FontName','Times New Roman','FontSize',16)
legend('resistance of charge', 'resistance of discharge','battery voltage', 'FontName','Times New Roman','FontSize',16)
