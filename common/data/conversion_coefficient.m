%% 电池功率到等效氢气消耗量的转换系数，动态变化的

p_fc=importdata("E:\SEU2\CVCI2022\Program CVCI\common\data\P_fc.mat");
fce_eff=importdata("E:\SEU2\CVCI2022\Program CVCI\common\data\fce_eff.mat");
h2_cons=importdata("E:\SEU2\CVCI2022\Program CVCI\common\data\h2_consumption.mat");

p_fc_conv=importdata("E:\SEU2\CVCI2022\Program CVCI\common\data\P_fc_conv.mat");
conversion = importdata("E:/SEU2/CVCI2022/Program CVCI/common/data/conversion.mat");

plot(p_fc_conv, conversion)
hold on
plot(p_fc_conv(13), conversion(13),'r*')
ylim([0,0.13])
text(6,0.02,"the value 0.0164 is used in paper", ...
    'FontName','Times New Roman','FontSize',16)
xlabel('Power of FCS (kW)','FontName','Times New Roman','FontSize',16)
ylabel('Conversion Coefficient','FontName','Times New Roman','FontSize',16)

conv_eff=median(conversion);
