%% Ratio Index
clear all

%Features to use
%[STEP FREQ, ENERGY. WALKING TIME RATIO, NUMBER OF STEPS]
Features = [1 3 5 6];
datapath = './MetricsData/NaiveBayes/';
load(strcat(datapath,'MetricsPost.mat'))
X = MetricsPost(:,Features);
Nsubj = size(MetricsPost,1);

%the mean and stddev of features across all healthy subjects
muH = nanmean(X);
sdH = nanstd(X);
X0 = min(X)*0.99;    %baseline


%compute ratio R for each subject
%loop through S-1 subjects
for s = 1:Nsubj
    
    ind = 1:Nsubj; ind(s) = []; %exclude s-th subject to compute nanmean and var
    %compute nanmean and variance of each feature across S-1 subjects
    Xopt = nanmean(X(ind,:));
    sdi = nanstd(X(ind,:));
    wi = 1./sdi;        %weights
    Xs = X(s,:);        %features of subj s            
    
    R = wi(1)*(Xs(1)-X0(1))./(Xopt(1)-X0(1)) + ...
        + wi(2)*(Xopt(2)*(X0(2)-Xs(2)))/((Xs(2)*(X0-Xopt(2)))) + ...
        + wi(3)*(Xs(3)-X0(3))./(Xopt(3)-X0(3)) + ...
        + wi(4)*(Xs(4)-X0(4))./(Xopt(4)-X0(4));
    
    IR(s) = R/sum(wi);
    
end

figure, boxplot(IR), title('Distribution of Ratio Expertise index across Healthy controls')


