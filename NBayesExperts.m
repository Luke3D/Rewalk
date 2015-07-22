%Compute Features for NaiveBayes Model Using Experts
%Features
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Number of steps per minute

clearvars -except IpMulti FigfeatH
% Features = [1 2 3 4];
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
    ylabel(AX(i),FeatureNames{Features(i)},'FontSize',Fontsize)
    xlabel(AX(4,i),FeatureNames{Features(i)},'FontSize',Fontsize)
    set(AX(i),'FontSize',Fontsize)
    set(AX(4,i),'FontSize',Fontsize)
%     histogram(AX(i,i),X1min(:,i),30)
end
%set axis limits as for healthy plot
% if exist('XlimH','var')
%     for a = 1:size(AX,1)*size(AX,2)
%         set(AX(a),'XLim',XlimH{a})
%         set(AX(a),'YLim',YlimH{a})
%     end
% end



%% Compute Expertise Index on Patients

%load metrics data (one patient)
clear Ip
symb = {'b-o','r-o','c-o','m-o'}; %symbol used to plot data for that patient
patient = 2;    %code for saving data
Patient = 'R10';
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

