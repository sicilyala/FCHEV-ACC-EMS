clear;
load('mot_eff.mat') % [0.7782,0.9803]
load('T_mot.mat')   % [-1800,1800]  
load('W_mot.mat')   % [-2500,2500]
mot_min=importdata('mot_trq_min.mat');
mot_max=importdata('mot_trq_max.mat');

[W,T]=meshgrid(W_mot, T_mot);
E=mot_eff;

xlim([0,2500])
ylim([-1800,1800])
plot(mot_min(:,1),mot_min(:,2),'r',mot_max(:,1),mot_max(:,2),'r','LineWidth',2)
xlabel('Motor Speed (rpm)','FontName','Times New Roman','FontSize',16)
ylabel('Motor Torque (Nm)','FontName','Times New Roman','FontSize',16)
legend('Torque Boundary','FontName','Times New Roman','FontSize',16)
hold on 
[c,h]=contour(W,T,E,30);

% clabel(c,h,'manual');
% set(h,'showtext','on')%显示等高线的值
% set(h,'showtext','on','textlist',[0.915;0.92815;0.96074])%设定等高线的值
% set(h,'string',sprintf('%.3f',get(h,'userdata')))

%% analysis of motor working potins

data_work=importdata("E:\SEU2\CVCI2022\Program CVCI\result2\Standard_ChinaCity_v_w_50_ep1000_v1\episode_data\data_ep999.mat");
W_mot_work = data_work.W_mot;
T_mot_work = data_work.T_mot;
mot_eff_work = data_work.mot_eff;
hold on
plot(W_mot_work, T_mot_work, '*')


