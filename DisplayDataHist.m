clear all,% close all
%Load Data Pre and Post
datapath = './MetricsData/Detailed/';
filenames = dir(strcat(datapath,'*.mat'));
% rnan = 0;           %# of removed rows containing Nans
% rstd = 0;           %# of removed rows with values outside 2stddev
Data = [];    %cell with metrics from post session and subjects
%[STEP FREQ, STD PHI, ENERGY, WALK TIME, WALK TIME RATIO, NUMBER OF STEPS]

Features = [1 2 3 5 6];
titles = {'Step F','Sd phi','Energy/Step','Walkdur [s]', 'Walk/Ttot','Nsteps','muTSw','sdTSw','muTSt','sdTSt'};


for subj = 1:length(filenames)
    if ~strcmp(filenames(subj).name(7:10),'Post')
        Metrics = load(strcat(datapath,filenames(subj).name));
        disp(['File: ' strcat(datapath,filenames(subj).name)]);
        Metrics = Metrics.Metrics;
        
        Data = [Data;Metrics];
    end
end

figure('name','Histogram of Features')
for i = 1:length(Features)
    subplot(2,3,i), hold on
    hist(Data(:,Features(i)),50); title(titles{Features(i)})
    set(gca,'FontSize',16)
end
