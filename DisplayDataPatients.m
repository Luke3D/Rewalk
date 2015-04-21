%display metrics for one subject and weighted avg across walk sections
%Patients(1 = R02; 2 = R08; 3 = R09; 4 = R10)

% close all
clearvars -except muH sdH MuPsih SdPsih Sall ccoeff pccoeff CIfig ICIfig Featplot IpMulti IpOne
load PsiHOne.mat    %load muPsiH and sdPsiH for One feature Z-scores

%Load Data from all training sessions
patient = 4;    %Xpatient # for plot
datapath = './MetricsData/Patients/R02/Detailed/';
filenames = dir(strcat(datapath,'*.mat'));
Nsessions = length(filenames);  %total # of training sessions
Datawmean = [];                 %weighted avg metrics across walk sections
DataAll = cell(1,Nsessions);    %metrics for each walk section across training
DataAllMatrix = [];
Features = [1 2 3 5 6];         %check that matches with var in NBayes.m
FeatureNames = {'Step F [Hz]','Sd \phi [deg]','Energy [counts]','Walkdur [s]','Twalk/Ttot','Steps'};
numFeat=length(Features); %Number of features (columns) in Metrics matrix
% symb = {'bo','rs','c*','mx'}; %symbol used to plot data for that patient
symb = {'bo','ro','co','mo'}; %symbol used to plot data for that patient


%plot weighted average for each feature
for s = 1:Nsessions
    Data = load(strcat(datapath,filenames(s).name)); %load Metrics
    Data = Data.Metrics;
    disp(['File: ' strcat(datapath,filenames(s).name)]);
    
    DataAllMatrix = [DataAllMatrix; Data(1:end-1,:)];   %concatenate all data in a single matrix
    DataAll{s} = Data(1:end-1,:);          %features from each walk section broken into sessions
    Datawmean = [Datawmean;Data(end,:)];   %weighted features mean across walk sections
    
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
    xlabel(AX(5,i),FeatureNames{Features(i)},'FontSize',16)
    set(AX(i),'FontSize',15)
    set(AX(5,i),'FontSize',15)
end
%set axis limits as for healthy plot
if exist('XlimH','var')
    for a = 1:size(AX,1)*size(AX,2)
        set(AX(a),'XLim',XlimH{a})
        set(AX(a),'YLim',YlimH{a})
    end
end

%Plot Covariance for Init and End training
% figure('name','Start Training Patients')
% [H,AX,BigAx,P,PAx] = plotmatrix(DataAllMatrixInit(:,Features),'+r');
% set(H,'Color','r')
% set(P,'FaceColor','r')
% for a = 1:size(AX,1)*size(AX,2)
%     set(AX(a),'XLim',XlimH{a})
%     set(AX(a),'YLim',YlimH{a})
% end
%
% figure('name','End Training Patients')
% [H,AX,BigAx,P,PAx] = plotmatrix(DataAllMatrixEnd(:,Features),'or');
% set(H,'Color','r')
% set(P,'FaceColor','r')
% for a = 1:size(AX,1)*size(AX,2)
%     set(AX(a),'XLim',XlimH{a})
%     set(AX(a),'YLim',YlimH{a})
% end

%plot of weighted mean across training sessions (show mean healthy controls
%if variable exists - run NBayes.m to generate muH)
if ~exist('Featplot','var')
    Featplot = figure;
    hold on
else
    figure(Featplot);
end

%CI on each session
% for s = 1:length(DataAll)
%     if size(DataAll{s},1) > 1  %at least 2 clips in the session
% %         wmean1000 = bootstrp(1000,@wmean,DataAll{s});
% %         wm(s,:) = mean(wmean1000,1);
%         ci(:,:,s) = bootci(1000,@wmean,DataAll{s});    %confidence interval for block b
%         ci(:,5,s) = bootci(1000,@sum,DataAll{s}(:,5)); %confidence interval for walk ratio
%         
%         %limit on Upper bound of Wratio CI
%         if(ci(2,5,s) > 1)
%             ci(2,5,s) = 1;
%         end
%     else
%         %wm(s,:) = Datawmean(s,:);
%     end
% end


% figure(Featplot), hold on
% for f = 1:numFeat
%     
%     cif = reshape(ci(:,Features(f),:),[2 size(ci,3)]); %confidence intervals for feature f
%     Lci = Datawmean(:,Features(f))-cif(1,:)';
%     Uci = cif(2,:)'-Datawmean(:,Features(f));
%     
%     subplot(3,2,f), hold on
%     title(FeatureNames{Features(f)})
%     %plot Healthy mean
%     if exist('muH','var')
%         plot(1:Nsessions,repmat(muH(f),[Nsessions 1]),'.-g','Linewidth',2)
%     end
%     
%     %plot Patients Weighted mean + CI 
%     Datawmean(:,Features(f)) = smooth(Datawmean(:,Features(f)),3); %smooth data 
%     errorbar(1:Nsessions,Datawmean(:,Features(f)),Lci,Uci,symb{patient});
%     %     plot(1:Nsessions,Datawmean(:,Features(f)),'s-b','Linewidth',2,'MarkerSize',6);
%     xlabel('Session')
%         
%     
%     %fit linear model to the data
%     linfit(:,f)=glmfit(1:Nsessions,Datawmean(:,Features(f))); %linear fit
%     S(f)=linfit(2,f); %slope of the line for feature f
%     
%     %plot linear fit
%     plot(1:Nsessions,glmval(linfit(:,f),1:Nsessions,'identity'),['-' symb{patient}(1)],'LineWidth',2)
%     
%     %compute Pearson correlation coefficient of time vs data
%     ccoeff(patient,f) = corr((1:Nsessions)',Datawmean(:,Features(f)));
%     
% %     %fit linear model to the data
% %     lsline
% %     linfit(:,f)=glmfit(1:Nsessions,Datawmean(:,Features(f))); %fit on blocks of 2 sessions
% %     S(f)=linfit(2,f); %slope of the line for feature f
% %    
% %     %CI of linfit slope 
% %     DATAlinfit(:,1) = 1:Nsessions; DATAlinfit(:,2) = Datawmean(:,Features(f));
% %     Slope_CI(:,f) = bootci(999,{@Slope,DATAlinfit});
%     
%     set(gca,'FontSize',16)
%     set(findall(gcf,'type','text'),'fontSize',16)
% 
% 
% end

%plot metrics for all walk sections
figure
for f = 1:numFeat
    subplot(3,2,f), hold on
    ylabel(FeatureNames{Features(f)})
%     title(FeatureNames{Features(f)})
    for s = 1:Nsessions
        np = size(DataAll{s},1);
        plot(repmat(s,[1 np]),DataAll{s}(:,Features(f)),'ro','MarkerSize',6)
        
    end
    plot(1:Nsessions,Datawmean(:,Features(f)),'r-s','Linewidth',2,'MarkerSize',6);
end

%% Features with CI with bootstrap - Aggregate sessions in blocks

clear Datab DATAlinfit

ylimFeat = {[0.25 0.95],[4 9],[5 20],[0 1],[0 90]};
% For aggregating save tot walk time of each session in last col of Features matrix
% this allows to do the weighted mean of the clips when 2 or more sessions are
% aggregated
for s = 1:Nsessions
    DataAll{s}(:,end) = repmat(sum(DataAll{s}(:,4)),[size(DataAll{s},1) 1]);
end

%Aggregate sessions in blocks
Session = 1:Nsessions;
Nb = 2;     %# of sessions per block
for b = 1:floor(Nsessions/Nb)
    Datab{b} = [DataAll{(b*2)-1};DataAll{b*2}];
end
if mod(Nsessions,Nb) > 0
    Datab{end} = [Datab{end};DataAll{end}];
end


%compute weighted mean and CI for each block by bootstrap
for b = 1:length(Datab)
    wmean1000 = bootstrp(1000,@wmean,Datab{b});
    wm(b,:) = mean(wmean1000,1);
    ci(:,:,b) = bootci(1000,@wmean,Datab{b});    %confidence interval for block b
end

%walk ratio is sum (Feature 5)
for b = 1:floor(Nsessions/Nb)
    %wm(b,5) = mean(Datawmean((b*2)-1:b*2,5));   
    wm(b,5) = mean(min(bootstrp(1000,@sum,Datab{b}(:,5)/2),1));
    ci(:,5,b) = bootci(1000,@sum,Datab{b}(:,5)/2);
    if(ci(2,5,b) > 1) 
        ci(2,5,b) = 1;
    end
end
if mod(Nsessions,Nb) > 0
     wm(end,5) = mean([wm(end,5),Datawmean(end,5)]);    %include last session
end
   
%CI for Wratio (all sessions)
% for t = 1:length(Datawmean)
%     if size(DataAll{t}(:,5),1) > 1  %if there are more than 1 datapoint
%             cit(:,t) = bootci(1000,@sum,DataAll{t}(:,5));
%     end
% end

%plot each wmean feature with its CI
if exist('CIfig','var')
    figure(CIfig), hold on
else
    CIfig = figure('name','Features with CI'); hold on
end

for f = 1:numFeat
    
    cif = reshape(ci(:,Features(f),:),[2 length(Datab)]); %confidence intervals for feature f
    subplot(3,2,f), hold on
    if exist('muH','var')
        plot(0:length(Datab)+1,repmat(muH(f),[length(Datab)+2, 1]),'-.g','Linewidth',2)
    end
%     title(FeatureNames{Features(f)})
    ylabel(FeatureNames{Features(f)}), ylim(ylimFeat{f})
    plot(1:length(Datab),wm(:,Features(f)),symb{patient},'Linewidth',2,'MarkerSize',6);
    %     lsline %add regression line
    
    %fit linear model to the data
    linfit(:,f)=glmfit(1:length(Datab),wm(:,Features(f))); %fit on blocks of 2 sessions
    S(f)=linfit(2,f); %slope of the line for feature f z
    
    %plot linear fit
    plot(1:length(Datab),glmval(linfit(:,f),1:length(Datab),'identity'),['-' symb{patient}(1)],'LineWidth',2)
    
    %compute Pearson correlation coefficient of time vs data and CI of corr
    [rho,prho] = corr((1:length(Datab))',wm(:,Features(f)));
    ccoeff(patient,f) = rho; pccoeff(patient,f) = prho;
    CIccoeff(:,f) = bootci(999,@corr,(1:length(Datab))',wm(:,Features(f)));
%     bootstatCC = bootstrp(999,@corr,(1:length(Datab))',wm(:,Features(f)));

%     ccoeff(patient,f) = corr((1:Nsessions)',Datawmean(:,Features(f)));  %use all sessions
%     CIccoeff(:,f,patient) = bootci(999,@corr,(1:Nsessions)',Datawmean(:,Features(f)));
%     bootstatCC = bootstrp(999,@corr,(1:Nsessions)',Datawmean(:,Features(f)));

%     seCC(patient,f) = std(bootstatCC); %Std error of the mean 

    
    %CI of linfit slope (WARNING There are not enough Datapoints, consider running the regression on all sessions)
    DATAlinfit(:,1) = 1:length(Datab); DATAlinfit(:,2) = wm(:,Features(f));
    Slope_CI(:,f) = bootci(999,{@Slope,DATAlinfit});
    
    %plot CI on feature values with dotted lines
    %     plot(1:length(Datab),cif(2,:),'r--')    %ci upper bound
    %     plot(1:length(Datab),cif(1,:),'r--')    %ci lower bound
    %
    %plot CI on feature values with error bars
    Lci = wm(:,Features(f))-cif(1,:)';
    Uci = cif(2,:)-wm(:,Features(f))';
    errorbar(1:length(Datab),wm(:,Features(f)),Lci,Uci,symb{patient});
    
    %limit Uci for Wratio (Feature 5)
    if Features(f) == 5
        Uci(Uci > 1) = 1;
    end
    
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

%plot Pearson corr coefficient
figure(CIfig), subplot(3,2,6), hold on
sp = [-0.2 -0.1 0.1 0.2]; %to space points
plot((1:numFeat)+sp(patient),ccoeff(patient,:),symb{patient},'MarkerSize',8,'LineWidth',2)
% errorbar((1:numFeat)+sp(patient),ccoeff(patient,:),ccoeff(patient,:)-CIccoeff(1,:),CIccoeff(2,:)-ccoeff(patient,:),symb{patient},'MarkerSize',8,'LineWidth',2)
% errorbar((1:numFeat)+sp(patient),ccoeff(patient,:),seCC(patient,:),symb{patient},'MarkerSize',8,'LineWidth',2)
ylim([-1 1])
line([1,5],[0,0], 'LineWidth',2,'Color',[.8 .8 .8])
xlabel('Feature'), %set(gca,'XTickLabel',FeatureNames(Features))
ylabel('Pearson r')
set(gca,'FontSize',16,'XTick',1:5,'XTickLabel',{'x1','x2','x3','x4','x5'})
set(findall(gcf,'type','text'),'fontSize',16)


%% Expertise index (GNB) with CI
%Index for each train sessions

for s = 1:Nsessions
    
    Xps = DataAll{s}(:,1:end-1);        %features for session s
   
    %Threshold features larger than healthy (StepF,Wratio,Nsteps)
    Xps(:,1) = min(Xps(:,1),muH(1));
    Xps(:,5) = min(Xps(:,5),muH(4));
    Xps(:,6) = min(Xps(:,6),muH(5));
    
    %at least 2 rows of data to compute CI
    if size(Xps,1) > 1
        
        %reformat data for bootstrp fcn
        muHb = repmat(muH,[size(Xps,1) 1]);
        sdHb = repmat(sdH,[size(Xps,1) 1]);
        MuPsihb = repmat(MuPsih,[size(Xps,1) 1]);
        SdPsihb = repmat(SdPsih,[size(Xps,1) 1]);
        MuPsihOneb = repmat(MuPsihOne,[size(Xps,1) 1]);
        SdPsihOneb = repmat(SdPsihOne,[size(Xps,1) 1]);
        Wratio = sum(Xps(:,5));    %Walk ratio is sum of individual ratios
        Wratio = repmat(Wratio,[size(Xps,1) 1]);
        
        I1000 = bootstrp(1000,@GNB,Xps,muHb,sdHb,MuPsihb,SdPsihb,Wratio);
        I(s) = mean(I1000);  %mean z-score of GNB
        varI(s) = var(I1000); %variance of Z-score for session S
        ciI(:,s) = bootci(1000,@GNB,Xps,muHb,sdHb,MuPsihb,SdPsihb,Wratio);    %confidence interval for session s
        UciI(s) =  ciI(1,s)-I(s);
        LciI(s) =  I(s)-ciI(2,s);
        
        %One feature z-score (Walk ratio)
        IOne1000 = bootstrp(1000,@GNBOne,Xps,muHb,sdHb,MuPsihOneb,SdPsihOneb);
        IOne(s) = mean(IOne1000);  %mean z-score of GNB
        varIOne(s) = var(IOne1000); %variance of Z-score for session S
        ciIOne(:,s) = bootci(1000,@GNBOne,Xps,muHb,sdHb,MuPsihOneb,SdPsihOneb);    %confidence interval for session s
        UciIOne(s) =  ciIOne(1,s)-IOne(s);
        LciIOne(s) =  IOne(s)-ciIOne(2,s);
        
    else
        Wratio = sum(Xps(:,5));    %Walk ratio is sum of individual ratios
        I(s) = GNB(Datawmean(s,1:end-1),muH,sdH,MuPsih,SdPsih,Wratio);
        IOne(s) = GNBOne(Datawmean(s,1:end-1),muH,sdH,MuPsihOne,SdPsihOne);

    end
        
end

if ~exist('ICIfig','var')
    ICIfig = figure('name','Expertise index with CI'); hold on
    hold on
else
    figure(ICIfig), hold on
end
subplot(121), hold on, title('Z-score - Multi Features')
errorbar(1:Nsessions,I,LciI,UciI,[symb{patient} '-'],'LineWidth',2);
subplot(122), hold on, title('Z-score - One Feature')
errorbar(1:Nsessions,IOne,LciIOne,UciIOne,[symb{patient} '-'],'LineWidth',2);


%save Z-score with CI and Var for current patient
%[z-score; Variance; UCI; LCI]
IpMulti{patient} = [I;varI;UciI;LciI];
IpOne{patient} = [IOne;varIOne;UciIOne;LciIOne];

