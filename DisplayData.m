% clear all, close all

%Load Data Pre and Post
datapath = './MetricsData/Healthy30Hz/';
filenames = dir(strcat(datapath,'*.mat'));
% rnan = 0;           %# of removed rows containing Nans
% rstd = 0;           %# of removed rows with values outside 2stddev
Data = cell(1,4);    %cell with metrics from all sessions and subjects
DataAll = [];
%Datanorm = cell(1,4);   %cell with normalized metrics from all sessions and subjects

for subj = 1:length(filenames)
        
    MetricsAll = load(strcat(datapath,filenames(subj).name));
    disp(['File: ' strcat(datapath,filenames(subj).name)]); 
    MetricsAll = MetricsAll.MetricsAll;

    %Split Pre into 3 parts (Pre1,Pre2,Pre3) and
    %Aggregate data from each session and all subjects 
    if strcmp(filenames(subj).name(7:9),'Pre')
        
        num = floor(size(MetricsAll,1)/3);    %size of each session
        for session=1:3
            Data{session} = [Data{session};MetricsAll((session-1)*num+1 : session*num,:)];
        end
    else
        Data{4} = [Data{4};MetricsAll];
    end
    
end

   
%Boxplot of each feature for 4 sessions (all subjects)  
Features = [1 2 3 5 6];  %Features to Display  - check that matches with var in NBayes.m
titles = {'Step F [Hz]','Std \phi [deg]','Energy [counts]','Walkdur [s]', 'Twalk/Ttest','Steps','muTSw','sdTSw','muTSt','sdTSt'};
FigfeatH = figure; hold on
for i = 1:length(Features)%size(MetricsAll,2)-1
    datai = [];
    for session = 1:4
        datai = [datai Data{session}(:,Features(i))];   %data for i-th feature across 4 sessions
    end
%     figure(i), hold on, title(titles{i})
    subplot(2,3,i), hold on, ylabel(titles{Features(i)})
    xlabel('Test #'), %ylabel(titles{Features(i)})
    bp = boxplot(datai,{'1','2','3','4'});
    set(bp(:),'LineWidth',2)
    set(gca,'FontSize',16)
    set(findall(gcf,'type','text'),'fontSize',16)

end
    
%plot cov matrix of data
for i = 1:4
   DataAll = [DataAll;Data{i}];
end
figure('name','Healthy - Cov Matrix');
hold on
[H,AX,BigAx,P,PAx] = plotmatrix(DataAll(:,Features));
set(H,'Color','g')
set(P,'FaceColor','g')   
RHO_H = corr(DataAll(:,Features));  %Linear (Pearson) correlation

for i = 1:length(Features)
    ylabel(AX(i),titles{Features(i)},'FontSize',16)
    xlabel(AX(5,i),titles{Features(i)},'FontSize',16)
    set(AX(i),'FontSize',15)
    set(AX(5,i),'FontSize',15)
    set(AX(:,3),'xlim',[0 25])
%     set(AX(:,5),'xlim',[-5 70])
end

%save xlims
XlimH = get(AX(:,:),'XLim');
YlimH = get(AX(:,:),'YLim');


%% plot Individual Feature values (no weighted mean) for last session
titlesy = {'Step F','Std \phi','Energy','Walkdur', 'Twalk/Ttest','Steps','muTSw','sdTSw','muTSt','sdTSt'};

FeaturesPostAll = [];
datapathdet = './MetricsData/Healthy30Hz/Detailed/';
filenames = dir(strcat(datapathdet,'*.mat'));
Nsubj = 0;
for subj = 1:length(filenames)
        if strcmp(filenames(subj).name(7:10),'Post')
            Nsubj = Nsubj+1;
                FeaturesPost = load(strcat(datapathdet,filenames(subj).name));
                FeaturesPost = FeaturesPost.Metrics;
                FeaturesPost(:,end) = Nsubj;    %add col with subj #
                FeaturesPostAll = [FeaturesPostAll;FeaturesPost(1:end-1,:)];
        end
end

figure('name','Healthy - Cov Matrix POST');
hold on
[H2,AX2,BigAx2,P2,PAx2] = plotmatrix(FeaturesPostAll(:,Features));
set(H2,'Color','g')
set(H2,'MarkerSize',20)
set(P2,'FaceColor','g')   
[RHO_H,pRHO_H] = corr(FeaturesPostAll(:,Features))  %Linear (Pearson) correlation

for i = 1:length(Features)
    ylabel(AX2(i),titlesy{Features(i)},'FontSize',16)
    xlabel(AX2(5,i),titles{Features(i)},'FontSize',16)
    set(AX2(i),'FontSize',16)
    set(AX2(5,i),'FontSize',16)
    %     set(AX2(:,2),'xlim',[3 9])
%     set(AX2(:,1),'xlim',[0.55 0.75])
%     set(AX(:,5),'xlim',[-5 70])
end
%axis lims 
set(AX2(2:5,1),'xlim',[0.55 0.75])
set(AX2(1,2:5),'ylim',[0.55 0.75])
set(PAx2(1),'XLim',[0.55 0.75])

set(AX2([1 3:5],2),'xlim',[2.5 8.5])
set(AX2(2,[1 3:5]),'ylim',[2.5 8.5])
set(PAx2(2),'XLim',[2.5 8.5])
set(AX2([1 3:5],2),'XTick',[4 7])
set(AX2(2,[1 3:5]),'YTick',[4 7])

set(AX2([1 2 4 5],3),'xlim',[4.5 10])
set(AX2(3,[1 2 4 5]),'ylim',[4.5 10])
set(PAx2(3),'XLim',[4.5 10])
set(AX2([1 2 4 5],3),'XTick',[6 8])
set(AX2(3,[1 2 4 5]),'YTick',[6 8])

set(AX2([1:3 5],4),'xlim',[-0.05 0.3])
set(AX2(4,[1:3 5]),'ylim',[-0.05 0.3])
set(PAx2(4),'XLim',[-0.05 0.3])

set(AX2(1:4,5),'xlim',[0 85])
set(AX2(5,1:4),'ylim',[0 85])
set(PAx2(5),'XLim',[0 85])


    

    %Features
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
    



