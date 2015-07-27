%display metrics for one subject and avg across walk sections
%Patients(1 = R09; 2 = R10; 3 = R11; 4 = R15)

%EXPERT: Based on expert subjects data 
%EXPERT: No longer use weighted mean. Each row of Metrics contain weighted
%average of features within a 1 minute window


% close all
clearvars -except muH sdH MuPsih SdPsih Sall ccoeff pccoeff CIfig ICIfig Featplot IpMulti IpOne
load PsiHOne.mat    %load muPsiH and sdPsiH for One feature Z-scores

%Load Data from all training sessions
patient = 4;    %Xpatient # for plot
datapath = './MetricsData/Patients/R15/Detailed/';
filenames = dir(strcat(datapath,'*.mat'));
Nsessions = length(filenames);  %total # of training sessions
Datamean = [];                 %Avg Metrics (Feature values) across walk sections
DataAllMatrix = [];            %Matrix with feature values for each minute for all sessions
DataAll = cell(1,Nsessions);   %metrics (feature values) for each minute splitted by session
Features = [1 2 3 4];         %check that matches with var in NBayes.m
FeatureNames = {'Step F [Hz]','Sd \phi [deg]','Energy [counts]','Steps'};
numFeat=length(Features); %Number of features (columns) in Metrics matrix
symb = {'bo','ro','co','mo'}; %symbol used to plot data for that patient
colorsymb = { [    0    0.4470    0.7410];
    [0.8500    0.3250    0.0980];  [0.9290    0.6940    0.1250];
    [0.4940    0.1840    0.5560]};  %colors for each patient

%plot weighted average for each feature
for s = 1:Nsessions
    Data = load(strcat(datapath,filenames(s).name)); %load Metrics (each row is a 1 minute window)
    Data = Data.Metrics;
    Data(:,end) = [];    %Remove last column (number of samples)
    disp(['File: ' strcat(datapath,filenames(s).name)]);
    
    rn0 = find(Data(:,1) ~= 0);      %zero rows have no walk data in the minute window
    Datamean = [Datamean;mean(Data(rn0,1:numFeat-1)) mean(Data(:,numFeat))];   %Features mean across 6 minutes
    DataAllMatrix = [DataAllMatrix; Data(rn0,:)];   %concatenate all data in a single matrix
    DataAll{s} = Data(rn0,:);
    
end

%show covariance matrix of selected features
if ~exist('Covplot','var')
    Covplot = figure;
    hold on
else
    figure(Covplot); hold on
end
[H,AX,BigAx,P,PAx] = plotmatrix(DataAllMatrix(:,Features),'or');
set(H,'Color','r')
set(P,'FaceColor','r')

for i = 1:length(Features)
    ylabel(AX(i),FeatureNames{Features(i)},'FontSize',16)
    xlabel(AX(4,i),FeatureNames{Features(i)},'FontSize',16)
    set(AX(i),'FontSize',15)
    set(AX(4,i),'FontSize',15)
end
%set axis limits as for healthy plot
if exist('XlimH','var')
    for a = 1:size(AX,1)*size(AX,2)
        set(AX(a),'XLim',XlimH{a})
        set(AX(a),'YLim',YlimH{a})
    end
end



%plot metrics for all walk sections
figure
for f = 1:numFeat
    subplot(2,2,f), hold on
    ylabel(FeatureNames{Features(f)})
%     title(FeatureNames{Features(f)})
    for s = 1:Nsessions
        np = size(DataAll{s},1);
        plot(repmat(s,[1 np]),DataAll{s}(:,Features(f)),symb{patient},'MarkerSize',6)
        
    end
    plot(1:Nsessions,Datamean(:,Features(f)),[symb{patient}(1) '-s'],'Linewidth',2,'MarkerSize',6);
end


%% Features with CI with bootstrap - Aggregate sessions in blocks
%Aggregate sessions in blocks

Session = 1:Nsessions;

% Nb = 2;     %# of sessions per block
% for b = 1:floor(Nsessions/Nb)
%     Datab{b} = [DataAll{(b*2)-1};DataAll{b*2}];
% end
% if mod(Nsessions,Nb) > 0
%     Datab{end} = [Datab{end};DataAll{end}];
% end

%Combine sessions into blocks (the 1st block contains the remainder sessions)
Sxb = 2; %# of sessions per block
Nb = floor(Nsessions/Sxb);  %# of blocks
Nbb = mod(Nsessions,Sxb);
Datab = cell(1,Nb+Nbb);
bb = 0;
if Nbb > 0
    Datab{1} = cell2mat(DataAll(1:Nbb));
    DataAll = DataAll(Nbb+1:end);
    bb = 1;
end
for b = 1:Nb
    Datab{b+bb} = cell2mat( DataAll((b*Sxb)-1:(b*Sxb))' );
end


%compute weighted mean and CI for each block by bootstrap
for b = 1:length(Datab)
    mean1000 = bootstrp(1000,@mean,Datab{b});  %resample the feature values and recompute the mean 1000 times
    fm(b,:) = mean(mean1000,1);                %the mean from the 1000 resampled values
    ci(:,:,b) = bootci(1000,@mean,Datab{b});   %95% confidence interval of the mean for each block
end

%plot each feature with its CI
if exist('CIfig','var')
    figure(CIfig), hold on
else
    CIfig = figure('name','Features with CI'); hold on
end

% ylimFeat = {[0.25 0.95],[4 9],[5 20],[0 90]};
for f = 1:numFeat

    subplot(2,2,f), hold on
    %Healthy mean
    if exist('muH','var')
        plot(0:length(Datab)+1,repmat(muH(f),[length(Datab)+2, 1]),'-.','Color',[0.4660 0.6740 0.1880],'Linewidth',2)
    end
  
    cif = reshape(ci(:,Features(f),:),[2 length(Datab)]); %confidence intervals for feature f
       
    ylabel(FeatureNames{Features(f)}) % ylim(ylimFeat{f})
    plot(1:length(Datab),fm(:,Features(f)),'o','Color',colorsymb{patient},'Linewidth',2,'MarkerSize',6);

    %fit linear model to the data
    linfit(:,f)=glmfit(1:length(Datab),fm(:,Features(f))); %fit on blocks of 2 sessions
    S(f)=linfit(2,f); %slope of the line for feature f z
    %plot linear fit
    plot(1:length(Datab),glmval(linfit(:,f),1:length(Datab),'identity'),'Color',colorsymb{patient},'LineWidth',2)
    
    %CI of linfit slope (WARNING There are not enough Datapoints, consider running the regression on all sessions)
%     DATAlinfit(:,1) = 1:length(Datab); DATAlinfit(:,2) = fm(:,Features(f));
%     Slope_CI(:,f) = bootci(999,{@Slope,DATAlinfit});
%     
    
    %compute Pearson correlation coefficient of time vs data and CI of corr
    [rho,prho] = corr((1:length(Datab))',fm(:,Features(f)));
    ccoeff(patient,f) = rho; pccoeff(patient,f) = prho;
    CIccoeff(:,f) = bootci(999,@corr,(1:length(Datab))',fm(:,Features(f)));
    
    %plot CI on feature values with error bars
    Lci = fm(:,Features(f))-cif(1,:)';
    Uci = cif(2,:)-fm(:,Features(f))';
    errorbar(1:length(Datab),fm(:,Features(f)),Lci,Uci,'o','MarkerFaceColor',colorsymb{patient});
    
    xlabel('Block')
    set(gca,'FontSize',16)
    set(findall(gcf,'type','text'),'fontSize',16)

end




%% add plot with slope values for each feature
%save slopes for each patient
% Sall(patient,:) = S;    %slopes for all features
% % Slope_CIL = Sall(patient,:)-Slope_CI(1,:);
% % Slope_CIU = Slope_CI(2,:)-Sall(patient,:);
% %log of slope
% Slope_CIL = sign(Sall(patient,:)-Slope_CI(1,:)).*abs(log10(abs(Sall(patient,:)-Slope_CI(1,:)))); %log of CI
% Slope_CIU = sign(Slope_CI(2,:)-Sall(patient,:)).*abs(log10(abs(Slope_CI(2,:)-Sall(patient,:))));
% Sall(patient,:) = sign(S).*abs(log10(abs(S)));  %log of slopes for each feature
% 
% figure(CIfig), subplot(3,2,6), hold on
% errorbar(1:numFeat,Sall(patient,:),Slope_CIL,Slope_CIU,symb{patient},'MarkerSize',8,'LineWidth',2)
% line([1,5],[0,0], 'LineWidth',2,'Color',[.8 .8 .8])
% xlabel('Feature'), %set(gca,'XTickLabel',FeatureNames(Features))
% title('log of Slopes')
% set(gca,'FontSize',16)
% set(findall(gcf,'type','text'),'fontSize',16)

%% plot Pearson corr coefficient
% figure(CIfig), subplot(3,2,6), hold on
% sp = [-0.2 -0.1 0.1 0.2]; %to space points
% plot((1:numFeat)+sp(patient),ccoeff(patient,:),symb{patient},'MarkerSize',8,'LineWidth',2)
% % errorbar((1:numFeat)+sp(patient),ccoeff(patient,:),ccoeff(patient,:)-CIccoeff(1,:),CIccoeff(2,:)-ccoeff(patient,:),symb{patient},'MarkerSize',8,'LineWidth',2)
% % errorbar((1:numFeat)+sp(patient),ccoeff(patient,:),seCC(patient,:),symb{patient},'MarkerSize',8,'LineWidth',2)
% ylim([-1 1])
% line([1,5],[0,0], 'LineWidth',2,'Color',[.8 .8 .8])
% xlabel('Feature'), %set(gca,'XTickLabel',FeatureNames(Features))
% ylabel('Pearson r')
% set(gca,'FontSize',16,'XTick',1:5,'XTickLabel',{'x1','x2','x3','x4','x5'})
% set(findall(gcf,'type','text'),'fontSize',16)


%% Expertise index (GNB) with CI
%Index for each train sessions

%TO UPDATE