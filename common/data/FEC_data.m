clear;
P_FC=[60.1, 56.9, 50.7, 46.0, 40.8, 35.6, 30.1, 24.3, 18.7, 12.9, 0];
P_FCE=[50.2, 48.1, 43.4, 39.6, 35.3, 31.1, 26.8, 21.8, 16.9, 11.8,0]; %kW
H2_consumption=[60.2, 55.0, 47.7, 42.3, 37.0, 31.8, 26.5, 21.1, 16.0, 10.5,0]; % g/min
FCE_eff=[41.69, 43.73, 45.49, 46.81, 47.70, 48.90, 50.57, 51.6, 52.81, 56.19,0];
FCE_eff=0.01*FCE_eff;

P_fc = 0:0.5:60;
P_fce=zeros(size(P_fc));
h2_consumption=zeros(size(P_fc));
fce_eff=zeros(size(P_fc));
n=max(size(P_fc));

for i = 1:n
    P_fce(i)=interp1(P_FC, P_FCE,P_fc(i),'cubic','extrap');
    h2_consumption(i)=interp1(P_FC, H2_consumption,P_fc(i),'cubic','extrap');
    fce_eff(i)=interp1(P_FC, FCE_eff,P_fc(i),'cubic','extrap');
end
% plot(P_fc, P_fce)

% plot(P_fc, h2_consumption)
% plot(P_fc, fce_eff)
%%
% xlabel('Power of FCS (kW)','FontName','Times New Roman','FontSize',16)
% yyaxis left
% plot(P_fc, fce_eff,'r:','LineWidth',2)
% ylim([0,0.6])
% ylabel('Efficiency','FontName','Times New Roman','FontSize',16)
% 
% yyaxis right
% plot(P_fc, h2_consumption/60,'b--','LineWidth',2)
% ylim([0,1.1])
% ylabel('H_{2} Consumption (g/s)','FontName','Times New Roman','FontSize',16)
% 
% legend('Fuel Efficiency', 'Fuel Consumption Rate','FontName','Times New Roman','FontSize',16)

%%
load('P_DCDC.mat')
load('DCDC_eff.mat')
e_dcdc=zeros(size(P_fc));
for i =1:n
    e_dcdc(i)=interp1(P_DCDC, DCDC_eff,P_fc(i),'linear');
end
plot(P_fc,e_dcdc,'LineWidth',2)
xlabel('Power of FCS (kW)','FontName','Times New Roman','FontSize',16)
ylabel('Efficiency of DC/DC Converter','FontName','Times New Roman','FontSize',16)
