% Import dataset
data = rmmissing(readtable("DatiProject.xlsx"));
head(data);

% Data preparation
data.yt = log(data.GDPC1);
data.log_pce = log(data.PCECTPI);
data.T_spread = data.GS10 - data.TB3MS;

% Computing first-differences of yt and PCE logs
delta_yt1 = diff(data.yt);
delta_log_pce = diff(data.log_pce);
head(data)

% Plot of the series - Section 1
plot(data.Time, data.yt); title("Log of GDP per Year"); grid on;
ylabel("Log of GDP"); 
xlabel("Years"); 

plot(delta_yt1); title("First difference of LogGDP per Year"); grid on;
ylabel("First difference in Log of GDP");

plot(data.Time, data.log_pce); title("Log of PCE per Year"); grid on;
ylabel("Log of PCE"); 
xlabel("Years");

plot(delta_log_pce); title("First difference of LogPCE per Year"); grid on;
ylabel("First difference in Log of PCE");

plot(data.Time, data.T_spread); title("Tspread per Year"); grid on;
ylabel("Tspread"); 
xlabel("Years");

% Training set up to Q1 1985
flt = data.Time <= datetime(2018,09,30);

delta_yt = diff(data(flt,:).yt);
delta_log_pce = diff(data(flt,:).log_pce);
dataTspread = data(flt,:).T_spread(2:end,:);

flt_test_set = data.Time > datetime(1985,03,30) & data.Time <= datetime(2018,09,30);

% ACF of VAR(4) series

% ACF delta_yt
autocorr(delta_yt);
parcorr(delta_yt);

% ACF delta_log_pce
autocorr(delta_log_pce);
parcorr(delta_log_pce);

% ACF T_spread
autocorr(data.T_spread);
parcorr(data.T_spread);

% Random Walk for yt
rw = arima('D', 1, 'Constant', 0);

data_rw = data(flt,:).yt; % considering only yt column
actual_data_rw = data(flt_test_set,:).yt; % extracting real data

fore_rw = computeForecasts(rw, data_rw, data_rw, 'Random Walk Model');
rw_uhat = exp(actual_data_rw) - exp(fore_rw(end-133:end,:));

plot(rw_uhat); 
title('Random Walk residuals');
grid on;

autocorr(rw_uhat);
parcorr(rw_uhat);

rmse_rw = sqrt(mean(rw_uhat.^2))

% BONUS
autocorr(data.yt)
parcorr(data.yt)
% p = 1 looks like the best option

% AIC for p selection da fare

ar_1 = arima(1,1,0);
fore_ar_1 = computeForecasts(ar_1, data_rw, data_rw, 'ARIMA(1,1,0) Model');
ar_1_uhat = exp(actual_data_rw) - exp(fore_ar_1(end-133:end,:));
rmse_ar1d = sqrt(mean(ar_1_uhat.^2))

% AR(2) for deltay
fore_ar_2_d = computeForecasts(arima(2,1,0), data_rw, data_rw, 'ARIMA(2,1,0)');
ar_2d_uhat = exp(actual_data_rw) - exp(fore_ar_2_d(end-133:end,:));
rmse_ar2d = sqrt(mean(ar_2d_uhat.^2))

% AR(4) for delta_yt
ar_4 = arima(4,1,0);
actual_delta_yt = diff(data(flt_test_set,:).yt); % actual data of yt used as the test set
fore_ar_4 = computeForecasts(ar_4, data_rw, data_rw, 'AR(4) Model');
ar_4_uhat = exp(actual_data_rw) - exp(fore_ar_4(end-133:end,:));

plot(ar_4_uhat);
title('AR(4) residuals');
grid on;

autocorr(ar_4_uhat);
parcorr(ar_4_uhat);

rmse_ar_4 = sqrt(mean(ar_4_uhat.^2))

% VAR(4) (using our function)
data_var = [delta_yt delta_log_pce dataTspread]
var_4 = varm(3,4)
%fore_var_4 = computeForecasts(var_4, data_var, data_var, 'VAR(4) Model') --> call to the function 

% VAR(4) with Rolling window on 100 observation with VAR(4)
fore_var_4 = [delta_yt delta_log_pce dataTspread];
k = 1;
j = 100;

for i = 1:134
    fore_var_4(100+i,1:3) = forecast(estimate(varm(3,4), data_var(k:j,:)), 1, data_var(k:j,:));
    
    k = k + 1;
    j = j + 1;
end

% Transform VAR(4) predictions to log of GDP forecasts
var_4_yt = data_rw

for i = length(var_4_yt)-133:length(var_4_yt)
    var_4_yt(i) = data_rw(i-1) + fore_var_4(i-1,1);
end

plot(var_4_yt, Color="red"); hold on;
plot(data_rw, Color="blue"); hold off;
title('VAR(4) for y_t');
legend("Forecast", "Actual", 'Location','northwest');
grid on;

var_4_uhat = exp(actual_data_rw) - exp(var_4_yt(end-133:end,:));
rmse_var_4 = sqrt(mean(var_4_uhat.^2))

% Plotting the 3 series

plot(fore_var_4(:,1), Color="red"); hold on;
plot(data_var(:,1), Color="blue"); hold off;
title('VAR(4) for \Delta_{y_t}');
legend("Forecast", "Actual", 'Location','northwest');
grid on;

plot(fore_var_4(:,2), Color="red"); hold on;
plot(data_var(:,2), Color="blue"); hold off;
title('VAR(4) for \pi_t');
legend("Forecast","Actual", 'Location','southwest');
grid on;

plot(fore_var_4(:,3), Color="red"); hold on;
plot(data_var(:,3), Color="blue"); hold off;
title('VAR(4) for Tspread');
legend( "Forecast","Actual", 'Location','southwest');
grid on;

% Residuals for the 3 series, RMSE for deltaY

var4_uhat(:,1) = exp(data_var(end-133:end,1)) - exp(fore_var_4(end-133:end,1));
var4_uhat(:,2) = exp(data_var(end-133:end,2)) - exp(fore_var_4(end-133:end,2));
var4_uhat(:,3) = exp(data_var(end-133:end,3)) - exp(fore_var_4(end-133:end,3));

rmse_var4_yd = sqrt(mean(var4_uhat(:,1).^2))
rmse_var4_pi = sqrt(mean(var4_uhat(:,2).^2))
rmse_var4_tspread = sqrt(mean(var4_uhat(:,3).^2))

% Plots of residuals and their ACF PCF

plot(var4_uhat(:,1));
title('VAR(4) for \Delta_{y_t} residuals')
grid on;
autocorr(var4_uhat(:,1));
parcorr(var4_uhat(:,1));
plot(var4_uhat(:,2));
title('VAR(4) for \pi_t residuals');
grid on;
autocorr(var4_uhat(:,2));
parcorr(var4_uhat(:,2));
plot(var4_uhat(:,3));
title('VAR(4) for Tspread residuals')
grid on;
autocorr(var4_uhat(:,3))
parcorr(var4_uhat(:,3))

% AIC for p selection
for pp = 1:12
    m1 = estimate(varm(3,pp), data_var(1:100,:));
    tmp = summarize(m1);
    IC(pp,:) = [tmp.AIC];
end
disp(IC)
[minaic,popt] = min(IC)

num = (1:12)'
AIC = [num IC] 
ticks = (1:12)
plot(num, IC, '-o', 'MarkerFaceColor','red');
ylabel("AIC Value"); 
xlabel("Lags");
xticks(ticks);
grid on;

% VAR(1)

data_var = [delta_yt delta_log_pce dataTspread]
var_1 = varm(3,1)
fore_var_1 = computeForecasts(var_1, data_var, data_var, 'VAR(1) Model')

var_1_yt = data_rw

for i = length(var_1_yt)-133:length(var_1_yt)
    var_1_yt(i) = data_rw(i-1) + fore_var_1(i-1,1);
end

plot(var_1_yt, Color="red"); hold on;
plot(data_rw, Color="blue"); hold off;
title('VAR(1) for y_t');
legend("Forecast", "Actual", 'Location','northwest');
grid on;

var_1_uhat = exp(actual_data_rw) - exp(var_1_yt(end-133:end,:));
rmse_var_1 = sqrt(mean(var_1_uhat.^2))


% Plotting the 3 series

plot(fore_var_1(:,1), Color="red"); hold on;
plot(data_var(:,1), Color="blue"); hold off;
title('VAR(1) for \Delta_{y_t}');
legend("Forecast", "Actual", 'Location','northwest');
grid on;

plot(fore_var_1(:,2), Color="red"); hold on;
plot(data_var(:,2), Color="blue"); hold off;
title('VAR(1) for \pi_t');
legend("Forecast","Actual", 'Location','southwest');
grid on;

plot(fore_var_1(:,3), Color="red"); hold on;
plot(data_var(:,3), Color="blue"); hold off;
title('VAR(1) for Tspread');
legend( "Forecast","Actual", 'Location','southwest');
grid on;

% Residuals for the 3 series, RMSE for deltaY

var1_uhat(:,1) = exp(data_var(end-133:end,1)) - exp(fore_var_4(end-133:end,1));
var1_uhat(:,2) = exp(data_var(end-133:end,2)) - exp(fore_var_4(end-133:end,2));
var1_uhat(:,3) = exp(data_var(end-133:end,3)) - exp(fore_var_4(end-133:end,3));

rmse_var1_yd = sqrt(mean(var1_uhat(:,1).^2))

plot(var1_uhat(:,1));
title('VAR(1) for \Deltay_t residuals')
grid on;
autocorr(var1_uhat(:,1));
parcorr(var1_uhat(:,1));
plot(var1_uhat(:,2));
title('VAR(1) for \pi_t residuals');
grid on;
autocorr(var1_uhat(:,2));
parcorr(var1_uhat(:,2));
plot(var1_uhat(:,3));
title('VAR(1) for Tspread residuals')
grid on;
autocorr(var1_uhat(:,3))
parcorr(var1_uhat(:,3))

% AR-X + PC


data_factors = readtable('DatiProject.xlsx', 'Sheet', 'DataForFactors');
Time = (datetime(1961,03,30):calquarters(1):datetime(1985,12,30))';
flt_pc = data_factors.Time <= datetime(2018,09,30);
data_factors = removevars(data_factors,{'Time'});
head(data_factors);

factors = table2array(data_factors);
%(1:100)
factors_s = zscore(factors(1:100,:)); % standardize
[coeff,F,cc, vv, explained, cg] = pca(factors_s);
bar(1:10,explained(1:10));
xlabel('Factor number (PC number)');
ylabel('Fraction of total variance of X explained');
grid on;


factors_fore = table2array(data_factors(flt_pc,:));

factor_all = zscore(factors)
[coeff_all,F_all,cca, vsv, explained_all, cgsda] = pca(factor_all);

plot(Time,movmean(F(:,1),4)); title('(4 quarters avg.) First PC ($\hat{F}_{1})$','Interpreter','latex'); grid on;

K = size(factors_s,2);
for kk=1:K
    res = fitlm(factors_s(:,kk),F(:,1));
    R2(kk,1) = res.Rsquared.Ordinary;
    res = fitlm(factors_s(:,kk),F(:,2));
    R2(kk,2) = res.Rsquared.Ordinary;
    res = fitlm(factors_s(:,kk),F(:,3));
    R2(kk,3) = res.Rsquared.Ordinary;
end
[~,id] = sort(R2(:,1),1,"descend");
bar(1:10, R2(id(1:10),1)'); xticks(1:10); xticklabels(data_factors.Properties.VariableNames(id(1:10))); xtickangle(45); ylabel('R^2'); grid on;

[~,id] = sort(R2(:,2),1,"descend");
bar(1:10, R2(id(1:10),2)'); xticks(1:10); xticklabels(data_factors.Properties.VariableNames(id(1:10))); xtickangle(45); 
ylabel('R^2')
grid on;

[~,id] = sort(R2(:,3),1,"descend");
bar(1:10, R2(id(1:10),3)'); xticks(1:10); xticklabels(data_factors.Properties.VariableNames(id(1:10))); xtickangle(45)
ylabel('R^2')
grid on;

% AR-X estimation giusto
w_start = 1;
w_size = 100;
n_fore = 134;
i = 1
fore_ar_x = delta_yt;
while i < n_fore

    yT = delta_yt(w_start:w_size,:); % y(t)
    factor_100 = F_all(w_start:w_size,1) 
    ar_x = estimate(varm(1,4), yT, 'X', factor_100, 'Display','off')
    fore_ar_x(100+i) = forecast(ar_x, 1, yT, "X", factor_100(end)); % with F at time t-1

    i = i + 1;
    w_start = w_start + 1;
    w_size = w_size + 1;
end

% AR-X estimation - base
w_start = 1;
w_size = 100;
n_fore = 134;
i = 1
fore_ar_x = delta_yt;
while i < n_fore
    factors_st = zscore(factors_fore(w_start:w_size,:)); % standardize and subset
    [coeff,F,cc,sgdf, explained,ccs] = pca(factors_st); % pca
   
    fhat1 = forecast(estimate(varm(1,1),F(:,1)),1,F(:,1));
    yT = delta_yt(w_start:w_size,:); % y(t)
   
    ar_x = estimate(varm(1,4), yT, 'X', F(:,1), 'Display','off')
    fore_ar_x(100+i) = forecast(ar_x, 1, yT, "X", fhat1); % with F at time t-1

    i = i + 1;
    w_start = w_start + 1;
    w_size = w_size + 1;
end

% Plot forecats for delta_yt (AR(4)-X) and actual delta_yt
plot(fore_ar_x, 'Color', 'red'); hold on;
plot(delta_yt, 'Color','blue');
title('AR(4)-X for \Delta_{y_t}');

% Transform AR(4)-X predictionns to log of GDP forecasts
ar_x_yt = data_rw

for i = length(ar_x_yt)-133:length(ar_x_yt)
    ar_x_yt(i) = data_rw(i-1) + fore_ar_x(i-1,1);
end

plot(ar_x_yt, Color="red"); hold on;
plot(data_rw, Color="blue"); hold off;
title('AR(4)-X for y_t');
legend("Forecast", "Actual", 'Location','northwest');
grid on;

ar_x_uhat = exp(actual_data_rw) - exp(ar_x_yt(end-133:end,:));
rmse_ar_x = sqrt(mean(ar_x_uhat.^2))

function fore = computeForecasts(model, data, fore, title_plot)

    k = 1;
    j = 100;

    for i = 1:134
        if contains(title_plot,'VAR')
            
            fore(100+i,1:3) = forecast(estimate(model, data(k:j,:)), 1, data(k:j,:));
            
            k = k + 1;
            j = j + 1;
        else 
            model = estimate(model, data(k:j,:));
            fore(100+i) = forecast(model, 1, data(k:j,:));
    
            k = k + 1;
            j = j + 1;
        end
    end
    if contains(title_plot,'VAR') == 0
        plot(fore, Color="red"); hold on;
        plot(data, Color="blue"); hold off;
        title(title_plot);
        legend("Forecast", "Actual", 'Location','northwest');
        grid on;
    end
end

