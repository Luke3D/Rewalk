%Compute Features for NaiveBayes Model Using Experts
%Features
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Number of steps per minute

clearvars -except IpMulti FigfeatH
% Features = [1 2 3 4];
datapath = './MetricsData/Experts/All/';
load(strcat(datapath,'MetricsMeanAll.mat'))

% X = MetricsMeanAll(:,Features);
X = MetricsMeanAll;
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

bp = boxplot(I), ylabel('z-score z \psi');
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


%% Compute Expertise Index on Patients

%load metrics data (one patient)
clear Ip
symb = {'b-o','r-o','c-o','m-o'}; %symbol used to plot data for that patient
patient = 4;    %code for saving data
Patient = 'R15';
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
patient = 4;    %code for saving data
Patient = 'R15';
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
    plot(s,cell2mat(Ipmean),'Linewidth',2,'MarkerSize',6);
    Ipmean{s} = mean(Ip{s})
end

plot(1:Nsessions,cell2mat(Ipmean),'Linewidth',2,'MarkerSize',6);

IpMultiAll{patient} = Ip;

%% Bootstrap z-scores over 1 min windows
%Combine sessions into blocks
Xpnew = Xp{p};
Sxb = 2; %# of sessions per block
Nb = floor(length(Im)/Sxb);  %# of blocks
Nbb = mod(length(Im),Sxb);
for b = 1:Nb
    Imnew{b} = cell2mat( Im((b*Sxb)-1:(b*Sxb))' );
end
if Nbb > 0
    Imnew{end} = [Imnew{end};cell2mat(Im(end))];
end

IponeAll{p} = Imnew;    %restructured in blocks
    