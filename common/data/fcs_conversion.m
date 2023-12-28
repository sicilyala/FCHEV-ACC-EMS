clear;
% FCS-efficient and conversion coefficient


load P_fc.mat
load fce_eff.mat
load P_fc_conv.mat
load conversion.mat


xlabel('Power of FCS (kW)','FontName','Times New Roman','FontSize',16)
yyaxis left
plot(P_fc, fce_eff,'r','LineWidth',2)
ylim([0,0.6])
ylabel('Efficiency','FontName','Times New Roman','FontSize',16)

fce_eff_max = max(fce_eff);
fce_eff_max_id = find(fce_eff==max(fce_eff));
hold on
plot(P_fc(fce_eff_max_id), fce_eff(fce_eff_max_id),'r*',LineWidth=2)

%%

yyaxis right
plot(P_fc_conv, conversion,'b','LineWidth',2)
% ylim([0,1.1])
ylabel('Equivalent H_2 conversion coefficient','FontName','Times New Roman','FontSize',16)
hold on
plot(P_fc_conv(14), conversion(14),'b*',LineWidth=2)

legend('fuel efficiency', 'the most efficient point', ...
    'power-varying coefficient', 'the fixed coefficient', ...
    'FontName','Times New Roman','FontSize',16)



%% 
% line([P_fc_conv(13),P_fc_conv(13)],[0,0.6])








