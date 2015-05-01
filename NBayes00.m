%Compute Features for NaiveBayes Model using Expert Patient (R00)

%Features
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Length of walking session (in seconds)
% 5. Ratio of time spent walking to total window time (deprecated)
% 6. Number of steps in walking section

clearvars -except IpMulti FigfeatH

%Features to use
Features = [1 2 3 6];
datapath = './MetricsData/R00/';
load(strcat(datapath,'MetricsPost.mat'))

X = MetricsPost(:,Features);

Ntrials = size(X,1);

%Compute sum of Z-scores (GNB) for each trial (compared to the others)
%loop through S-1 trials
%1 trial correspond to a 1 min walk clip during a 6mWT
for s = 1:Ntrials
    
    ind = 1:Ntrials; ind(s) = []; %exclude s-th subject to compute nanmean and var
    %compute nanmean and variance of each feature across S-1 subjects
    mui = nanmean(X(ind,:));
    sdi = nanstd(X(ind,:));
    
    Xs = X(s,:);        %features of subj s
    logPh(s) = -sum( 0.5*log(2*pi*sdi.^2) + ((Xs-mui).^2)./(2*sdi.^2) );  %sum of Z-scores
    
end

%Compute expertise of each healthy control relative to the others
for s = 1:Ntrials
    
    ind = 1:Ntrials; ind(s) = []; %exclude s-th subject to compute nanmean and var
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


%% Compute Expertise Index on Patients (TO UPDATE)

%load metrics data (one patient)
symb = {'b-o','r-s','c-*','m-x'}; %symbol used to plot data for that patient
patient = 3;    %code for saving data
Patient = 'R09';
datapath_patients = './MetricsData/NaiveBayes/Patients/';
Metricswmean = load([datapath_patients Patient '_Metricswmean.mat']); %matrix with results from each training session
Xp = Metricswmean.Datawmean;    %features for the patient each training session (row)

% %Normalize StepF to max device StepF [% of max]
% datapath_patients_Set = './MetricsData/Patients/';
% Fmax = load([datapath_patients_Set Patient '/StepSettings.mat']); Fmax = Fmax.StepSettings;
% Xp(:,1) = Xp(:,1)./Fmax;

Xp = Xp(:,Features);
Nsessions = size(Xp,1);

%Index for each train sessions
for s = 1:Nsessions
    
    %     %StepF Normalization - correct Healthy step Freq for R10 from session 4
    %     if strcmp(Patient,'R10') && s==4
    %         f0max = (1+0.25)^(-1);  %based on Healthy Settings
    %         f1max = (0.78+0.225)^(-1); %based on mean R10 settings
    %         muH(1) = muH(1)*(f1max/f0max);
    %         sdH(1) = sdH(1)*(f1max/f0max);
    %     end
    
    Xps = Xp(s,:);        %features for session s
    
    %Threshold features larger than healthy (StepF,Wratio,Nsteps)
    Xps(1) = min(Xps(1),muH(1));
    Xps(4) = min(Xps(4),muH(4));
    Xps(5) = min(Xps(5),muH(5));
    
    
    logPp(s) = -sum( 0.5*log(2*pi*sdH.^2) + ((Xps-muH).^2)./(2*sdH.^2) );  %sum of Z-scores
    Ip(s) = (logPp(s)-MuPsih)./SdPsih;  %Expertiese index for session s
end

figure('name','Expertise index'), hold on
plot(1:Nsessions,Ip,symb{patient},'Linewidth',2,'MarkerSize',6);

%save z-score Index for current patient
IpMulti{patient} = Ip;