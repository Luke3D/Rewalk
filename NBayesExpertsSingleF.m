%Compute Features for NaiveBayes Model
clearvars -except IpOne
%Features
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Number of steps in walking section

%Features to use
Features = [1 2 3 4];
FeatureNames = {'Step F','Sd phi','Energy/Step','Nsteps'};
symb = {'b-o','r-s','c-*','m-x'}; %symbol used to plot data for that patient

%healthy data
datapath = './MetricsData/Experts/All/';
load(strcat(datapath,'MetricsMeanAll.mat'))
MetricsMeanAll(3,:) = [];
Nsubj = size(MetricsMeanAll,1);

%Patient Data
Patient = 'R11';
patient = 2;    %code for saving data
datapath_patients = './MetricsData/NaiveBayes/Patients/';
%load metrics data (one patient)
Metricswmean = load([datapath_patients Patient '_Metricswmean.mat']); %matrix with results from each training session
Metricswmean = Metricswmean.Datawmean;    %features for the patient each training session (row)


for f=1:length(Features)
    
    X = MetricsMeanAll(:,Features(f));
    
    %Compute sum of Z-scores for each subject (compared to the others)
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
    
    figure(1), hold on
    subplot(2,2,f), hold on
    boxplot(I), title(FeatureNames{Features(f)})
    
    %Compute Psih Distribution
    logPh = logPh';
    MuPsih = nanmean(logPh), SdPsih = nanstd(logPh)
    %save Psih Distribution for single features
    MuPsihOne(f) = MuPsih; 
    SdPsihOne(f) = SdPsih;
    
    %the mean and stddev of features across all healthy subjects
    muH = nanmean(X);
    sdH = nanstd(X);
    


%% Compute Expertise Index on Patients
% figure('name','Expertise index'), hold on

    
    Xp = Metricswmean(:,Features(f));
    Nsessions = size(Xp,1);
    
    %Index for each train sessions
    for s = 1:Nsessions
        
%         %%correct Healthy step Freq for R10 from session 4
%         if strcmp(Patient,'R10') && s==4 && Features(f)==1
%             f0max = (1+0.25)^(-1);  %based on Healthy Settings
%             f1max = (0.78+0.225)^(-1); %based on mean R10 settings
%             muH = muH*(f1max/f0max);
%             sdH = sdH*(f1max/f0max);
%         end
    
        Xps = Xp(s,:);        %features for session s
        
        %Threshold features larger than healthy (StepF,Wratio,Nsteps)
        if Features(f) == 1 || Features(f) == 5 || Features(f) == 6
            Xps = min(Xps,muH);
        end
        
        logPp(s) = -sum( 0.5*log(2*pi*sdH.^2) + ((Xps-muH).^2)./(2*sdH.^2) );  %sum of Z-scores
        Ip(s,f) = (logPp(s)-MuPsih)./SdPsih;  %Expertiese index for session s when using feature f
    end
    
    figure(2), hold on
    subplot(2,2,f), hold on,  title(FeatureNames{Features(f)})    
    plot(1:Nsessions,Ip(:,f),symb{patient},'Linewidth',2,'MarkerSize',6);
    
end
IpOne{patient} = Ip;
save('PsiHOne.mat','MuPsihOne','SdPsihOne');