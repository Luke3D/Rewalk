clear all, close all
%Load Data Pre and Post
datapath = './MetricsData/NewMetrics/';
filenames = dir(strcat(datapath,'*.mat'));
rnan = 0;           %# of removed rows containing Nans
rstd = 0;           %# of removed rows with values outside 2stddev
Data = cell(1,4);    %cell with metrics from all sessions and subjects
Datanorm = cell(1,4);   %cell with normalized metrics from all sessions and subjects

for subj = 5:6%length(filenames)
        
    MetricsAll = load(strcat(datapath,filenames(subj).name));
    disp(['File: ' strcat(datapath,filenames(subj).name)]); 
    MetricsAll = MetricsAll.MetricsAll;
    
    %Remove Nan
    in = isnan(MetricsAll);
    [r,c] = find(in==1);
    r = unique(r);
    MetricsAll(r,:) = [];   %remove rows with nans
    rnan = rnan+sum(length(r));          %keeps track of # of rows removed
    
    %remove feature vals outside 3 stddev from each col
    if size(MetricsAll,1) > 1
        mu = mean(MetricsAll,1); sdev = std(MetricsAll,[],1);
        for k = 1:size(MetricsAll,2)
            ind = find(MetricsAll(:,k) > mu(k)+3*sdev(k) | MetricsAll(:,k) < mu(k)-3*sdev(k));
            if ~isempty(ind)
                MetricsAll(ind,:) = [];
                rstd = rstd+1;  %keep track of removed rows
            end
        end
    end
    
    %Split Pre into 3 equal parts (Pre1,Pre2,Pre3) and
    %Aggregate data from each session and all subjects 
    if strcmp(filenames(subj).name(7:9),'Pre')
        
        num = floor(size(MetricsAll,1)/3);    %size of each session
        for session=1:3
            Data{session} = [Data{session};MetricsAll((session-1)*num+1 : session*num,:)];
        end
    else
        Data{4} = [Data{4};MetricsAll];
    end
    
    RemovedNan(subj) = rnan; %keep track of removed data
    RemovedStd(subj) = rstd; %keep track of removed data
    rnan = 0; rstd = 0;
end
   
%Take Max and Min from each col and Normalize features
for k = 1:4
    mins(k,:) = min(Data{k});
    maxs(k,:) = max(Data{k});
end

mins = min(mins,[],1);
maxs = max(maxs,[],1);

for k = 1:4
    repmins = repmat(mins,[size(Data{k},1) 1]);
    repmaxs = repmat(maxs,[size(Data{k},1) 1]);
    Datanorm{k} = (Data{k} - repmins)./(repmaxs - repmins);
  
end

%Remove Walk ratio and Expertise Index (last col) from Data matrices (redundant)
for k = 1:4
    Data{k}(:,[5 end]) = [];
    Datanorm{k}(:,[5 end]) = [];
end

%Histogram and Boxplot of feature for each session (all subjects)  
Labels = {'Step F','Sd phi','E','Walkdur','Nsteps','muTSw','sdTSw','muTSt','sdTSt'};
bplotnorm = figure(1);
for k = 1:4
    figure(bplotnorm), hold on
    subplot(2,2,k), boxplot(Datanorm{k}(:,1:end),'labels',Labels), ylim([-0.1 1.1])
    figure(k+1), hold on
    for f =1:size(Data{k},2)
        subplot(2,5,f), hist(Data{k}(:,f),10), title(Labels{f});
    end
end

RemovedNan
RemovedStd


%  1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Length of walking session (in seconds)
% 5. Ratio of time spent walking to total window time
% 6. Number of steps in walking section
% 7. Average of twa_swing
% 8. Stddev of twa_swing
% 9. Average of twa_stance
% 10. Stddev of twa_stance
% 11. Calculated index for walking section
    


%Compute mean and std dev for each feature and session


