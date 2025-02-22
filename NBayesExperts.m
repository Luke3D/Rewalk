%Compute Features for NaiveBayes Model Using Experts
%Features
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Number of steps per minute

clearvars -except IpMulti FigfeatH
% Features = [1 2 3 4];

nbins=[10 10 10 30];

datapath = './MetricsData/Experts/All/';
load(strcat(datapath,'MetricsMeanAll.mat')) %mean features across 6 minutes
load(strcat(datapath,'MetricsAll.mat')) %feature values for each minute

% X = MetricsMeanAll(:,Features);
X = MetricsMeanAll;
X1min = MetricsAll;
Nsubj = size(X,1);

%Compute sum of Z-scores (GNB) for each subject (compared to the others)
%loop through S-1 subjects
for s = 1:Nsubj
    
    ind = 1:Nsubj; ind(s) = []; %exclude s-th subject to compute nanmean and var
    %compute nanmean and variance of each feature across S-1 subjects
    mui = nanmean(X(ind,:));
    sdi = nanstd(X(ind,:));    
    Xs = X(s,:);        %features of subj s
    logPh(s) = -sum( 0.5*log(2*pi*sdi.^2) + ((Xs-mui).^2)./(2*sdi.^2) );  %sum of Z-scores
    
end

%Compute expertise of each healthy control relative to the others
for s = 1:Nsubj
    
    ind = 1:Nsubj; ind(s) = []; %exclude s-th subject to compute nanmean and var
    MuPsih = nanmean(logPh(ind)); SdPsih = nanstd(logPh(ind));
    I(s) = (logPh(s)-MuPsih)/SdPsih;
end


if exist('FigfeatH','var')
    figure(FigfeatH), hold on, subplot(2,3,6);
else
    figure, hold on
end

bp = boxplot(I), ylabel('z-score z_{\Psi_h}','Interpreter','latex');
set(bp(:),'LineWidth',2)
set(gca,'FontSize',16)
set(findall(gcf,'type','text'),'fontSize',16)
set(gca,'XtickLabel',{''})

%Compute Psih Distribution
logPh = logPh';
MuPsih = nanmean(logPh), SdPsih = nanstd(logPh)

%the mean and stddev of features across all healthy subjects
muH = nanmean(X);
sdH = nanstd(X);

%show covariance matrix of selected features
figure
AX = plotmatrix(X);

%plot normal distribution of Z for experts
muZ = mean(I);
sdZ = std(I);
pdH = makedist('Normal',muZ,sdZ);
xpdH = muZ-3*sdZ:4*sdZ/20:muZ+3*sdZ;
ypdH = pdf(pdH,xpdH);
figure, hold on
plot(xpdH,ypdH./max(ypdH))
plot(I,zeros(length(I)),'o')           %use average from 6 mins
xlabel('z-score'), ylabel('p(Z_\{psi})')


%% Analysis for each minute 

%covariance matrix for each minute
figure
FeatureNames = {'Step F [Hz]','Sd \phi [deg]','Energy [counts]','Steps'};
Features = [1 2 3 4];    
Fontsize = 10;
[H,AX,BigAx,P,PAx] = plotmatrix(X1min,'og');
set(H,'Color',[0 0.8 0])
set(P,'FaceColor',[0 0.8 0])
set(H,'MarkerFaceColor',[0 0.8 0])
set(H,'MarkerSize',6)


%INCREASE NUMBER OF BINS OF HISTOGRAMS
for i = 1:length(Features)
    
    hold(PAx(i), 'on')
    cla(PAx(i))
    
    ylabel(AX(i),FeatureNames{Features(i)},'FontSize',Fontsize)
    xlabel(AX(4,i),FeatureNames{Features(i)},'FontSize',Fontsize)
    set(AX(i),'FontSize',Fontsize)
    set(AX(4,i),'FontSize',Fontsize)
    
    h=histogram(PAx(i),X1min(:,i),nbins(i))
    h.FaceColor=[0 0.8 0];
%     histogram(AX(i,i),X1min(:,i),30)
end
%set axis limits as for healthy plot
% if exist('XlimH','var')
%     for a = 1:size(AX,1)*size(AX,2)
%         set(AX(a),'XLim',XlimH{a})
%         set(AX(a),'YLim',YlimH{a})
%     end
% end

%correlation of features
[rho,prho] = corr(X1min);

%plot separate histograms for features with a normal distribution overlaid

%remove steps outliers
% X1min([53 54 63],:) = [];
fig1 = figure;
fig2 = figure;
bins = [11 8 11 11];
for f = 1:4
x1 = X1min(:,f);
pd1 = fitdist(x1,'Normal'); mu1 = pd1.mean; sd1 = pd1.sigma;
x1pdf = mu1-3*sd1:4*sd1/20:mu1+3*sd1;
y1pdf = pdf(pd1,x1pdf);
figure(fig1); hold on
subplot(2,2,f), hold on, title(FeatureNames{f})
h1 = histogram(x1,bins(f)); scale(f) = max(h1.Values)/max(y1pdf);
plot(x1pdf,y1pdf.*scale(f))
hold off
muH2(f) = mu1;  %mean and std deviation of the normal distr. fitted on all the samples
sdH2(f) = sd1;

figure(fig2); hold on
subplot(2,2,f);
z1 = (x1-mu1)./sd1;
h = kstest(z1)
[fcdf,x_values] = ecdf(z1);
F = plot(x_values,fcdf);
set(F,'LineWidth',2);
hold on;
G = plot(x_values,normcdf(x_values,0,1),'r-');
set(G,'LineWidth',2);
legend([F G],...
       'Empirical CDF','Standard Normal CDF',...
       'Location','SE');
end

%plot distribution with muH and sdH as parameters 
bins = [11 8 11 30];
figure
for f = 1:4
pd = makedist('Normal',muH(f),sdH(f));
x1 = X1min(:,f);
x1pdf = muH(f)-3*sdH(f):4*sdH(f)/20:muH(f)+3*sdH(f);
y1pdf = pdf(pd,x1pdf);
subplot(2,2,f), hold on, xlabel(FeatureNames{f}), ylabel('p(x)')
%histogram
% h1 = histogram(x1,bins(f));                   %use data from each min
% scale(f) = max(h1.Values)/max(y1pdf);
plot(x1pdf,y1pdf./max(y1pdf))
%scatter plot
plot(X(:,f),zeros(size(X,1)),'o')           %use average from 6 mins
% plot(X1min(:,f),zeros(size(X1min,1)),'o')   %use data from each min
end
hold off
%% Compute Expertise Index on Patients

%load metrics data (one patient)
clear Ip
symb = {'b-o','r-o','c-o','m-o'}; %symbol used to plot data for that patient
patient = 1;    %code for saving data
Patient = 'R09';
datapath_patients = './MetricsData/NaiveBayes/Patients/';
Metricswmean = load([datapath_patients Patient '_MetricswMean.mat']); %matrix with results from each training session
Xp = Metricswmean.Datawmean;    %features for the patient each training session (row)
% Xp = Xp(:,Features);
Nsessions = size(Xp,1);

%Index for each train sessions
for s = 1:Nsessions
       
    Xps = Xp(s,:);        %features for session s
      
    logPp(s) = -sum( 0.5*log(2*pi*sdH.^2) + ((Xps-muH).^2)./(2*sdH.^2) );  %sum of Z-scores
    Ip(s) = (logPp(s)-MuPsih)./SdPsih;  %Expertiese index for session s
end

figure('name','Expertise index'), hold on
plot(1:Nsessions,Ip,symb{patient},'Linewidth',2,'MarkerSize',6);

%save z-score and std deviation for each session for current patient
IpMulti{patient} = Ip;

%% Compute z-score for each 1 min window across all sessions
clear Ip Ipmean
symb = {'b-o','r-o','c-o','m-o'}; %symbol used to plot data for that patient
% patient = 4;    %code for saving data
% Patient = 'R15';
datapath_patients = './MetricsData/NaiveBayes/Patients/';
Metricsall = load([datapath_patients Patient '_Metricsall.mat']); %matrix with results from each training session
Xp = Metricsall.DataAll;    %features for the patient each training session (row)
Nsessions = size(Xp,2);
Ipmean = {};

%Index for each train sessions
for s = 1:Nsessions
       
    Xps = Xp{s};        %features for session s
    for ss = 1:size(Xps,1)
        logPp = -sum( 0.5*log(2*pi*sdH.^2) + ((Xps(ss,:)-muH).^2)./(2*sdH.^2) );  %sum of Z-scores
        Ip{s}(ss,1) = (logPp-MuPsih)./SdPsih;  %z-score for each minute of session s
    end
%     plot(s,cell2mat(Ipmean),'Linewidth',2,'MarkerSize',6);
%     Ipmean{s} = mean(Ip{s})
end

% plot(1:Nsessions,cell2mat(Ipmean),'Linewidth',2,'MarkerSize',6);

IpMultiAll{patient} = Ip

%% Bootstrap last 2 sessions to compute CI of z-score
XEOT = cell2mat(Xp(end-1:end)');
N = size(XEOT,1)
[bootstat,bootsam] = bootstrp(100,'Zscore',XEOT,repmat(muH,[N 1]),repmat(sdH,[N 1]),MuPsih,SdPsih)

