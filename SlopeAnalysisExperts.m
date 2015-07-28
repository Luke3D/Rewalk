%plot Multi and OneFeat indices
close all
clear all
symb = {'b-o','r-o','c-o','m-o'}; %symbol used to plot data for that patient
colorsymb = { [    0    0.4470    0.7410];
    [0.8500    0.3250    0.0980];  [0.9290    0.6940    0.1250];
    [0.4940    0.1840    0.5560]};  %colors for each patient
%% Use each minute window
load ./matFiles/IpMultiExpertsAll.mat
load ./matFiles/IpOneExpertsAll.mat
OneFeat = 4; %what single feature is used
Sxb = 2; %# of sessions per block
nboot = 999;
%ONE FEATURE
Zfig = figure;
Esfig = figure;
figure(Zfig), subplot(121),  hold on,  title('Z_{\psi} - Steps only')
for p = 1:length(IponeAll)
    
    clear muz sdz sem Imnew
    
    %Combine sessions into blocks (the 1st block contains the remainder sessions)
    Im = IponeAll{p};
    Nb = floor(length(Im)/Sxb);  %# of blocks
    Nbb = mod(length(Im),Sxb); 
    Imnew = cell(1,Nb+Nbb);
    bb = 0; 
    if Nbb > 0
        Imnew{1} = cell2mat(Im(1:Nbb));
        Im = Im(Nbb+1:end);
        bb = 1;
    end
    for b = 1:Nb
        Imnew{b+bb} = cell2mat( Im((b*Sxb)-1:(b*Sxb))' );
    end
    
    %combine sessions into blocks (last block contains remainder sessions)
    %     for b = 1:Nb
    %         Imnew{b} = cell2mat( Im((b*Sxb)-1:(b*Sxb))' );
    %     end
    %     if Nbb > 0
    %         Imnew{end} = [Imnew{end};cell2mat(Im(end))];
    %     end
    
    IponeAll{p} = Imnew;    %restructured in blocks
    
    Ns = length(IponeAll{p});
    for s = 1:Ns
        muz(s) = mean(IponeAll{p}{s}(:,OneFeat));
        sdz(s) = std(IponeAll{p}{s}(:,OneFeat));
        sem(s) = sdz(s)/sqrt(size(IponeAll{p}{s},1));
    end
    
    %plot mean and std dev of z-score for each session
    figure(Zfig), subplot(121),  hold on,  title('Z_{\psi} - Steps only')
    errorbar(1:Ns,muz,sdz,'o-','Color',colorsymb{p},'Linewidth',2),   %error bars = std dev
%     errorbar(1:Ns,muz,1.96*sdz/sqrt(size(IponeAll{p}{s},1)),symb{p},'Linewidth',2), %error bars = sem
    ylim([-600 40])
    
    %compute effect size and sem of effect size (for measure 1)
    es1one = diff(muz)./sem(2:end);
    es1one_{p} = diff(muz)./sem(2:end);
    snr1(1,p) = mean(diff(muz)./sem(2:end));
    semsnr1(1,p) = std( diff(muz)./sem(2:end) ) / sqrt(length(sem(2:end)));
    %     snr1(1,p) = mean( (diff(muz)./muz(1:end-1))./sem(2:end) );
    %     semsnr1(1,p) = std( (diff(muz)./muz(1:end-1))./sem(2:end) ) / sqrt(length(sem(2:end)));
    snr2(1,p) = mean(diff(muz))/mean(sem(2:end));
    
    figure(Esfig), subplot(121), hold on
    plot(p*ones(1,length(es1one)),es1one,symb{p})
    %     plot(es1one,symb{p})
    
end
figure(Zfig)
set(gca,'FontSize',14)
set(findall(gcf,'type','text'),'fontSize',14), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',14), %ylim([-1000 20])
xlabel('Block #'), ylabel('Z-score')
legend('R09','R10','R11','R15')
line([0 Ns + 2],[2 2],'LineWidth',1,'Color',[0 0.7 0])
line([0 Ns + 2],[-2 -2],'LineWidth',1,'Color',[0 0.7 0])

%ALL FEATURES
figure(Zfig)
subplot(122),  hold on,  title('Z_{\psi} - All features')
for p = 1:length(IpMultiAll)
    
    clear muz sdz sem Imnew
    
    %     %Combine sessions into blocks
    Im = IpMultiAll{p};
    Nb = floor(length(Im)/Sxb);  %# of blocks
    Nbb = mod(length(Im),Sxb);
    Imnew = cell(1,Nb+Nbb);
    bb = 0;
    if Nbb > 0
        Imnew{1} = cell2mat(Im(1:Nbb));
        Im = Im(Nbb+1:end);
        bb = 1;
    end
    for b = 1:Nb
        Imnew{b+bb} = cell2mat( Im((b*Sxb)-1:(b*Sxb))' );
    end
    
    %     for b = 1:Nb
    %         Imnew{b} = cell2mat( Im((b*Sxb)-1:(b*Sxb))' );
    %     end
    %     if Nbb > 0
    %         Imnew{end} = [Imnew{end};cell2mat(Im(end))];
    %     end
    IpMultiAll{p} = Imnew;    %restructured in blocks
    
    
    Ns = length(IpMultiAll{p});
    for s = 1:Ns
        muz(s) = mean(IpMultiAll{p}{s});
        sdz(s) = std(IpMultiAll{p}{s});
        sem(s) = sdz(s)/sqrt(size(IpMultiAll{p}{s},1));
    end
    
    %plot mean and std dev of z-score for each session
    figure(Zfig),subplot(122),  hold on,  title('Z_{\psi} - All features')
    errorbar(1:Ns,muz,sdz,'o-','Color',colorsymb{p},'Linewidth',2),
    %     errorbar(1:Ns,muz,1.96*sdz/sqrt(size(IpMultiAll{p}{s},1)),symb{p},'Linewidth',2),
    ylim([-600 40])
    
    %compute effect size and sem of effect size (for measure 1)
    es1multi = diff(muz)./sem(2:end);
    es1multi_{p} = diff(muz)./sem(2:end);
    snr1(2,p) = mean( diff(muz)./sem(2:end) );
    semsnr1(2,p) = std( diff(muz)./sem(2:end) ) / sqrt(length(sem(2:end)));
    %     snr1(2,p) = mean( (diff(muz)./muz(1:end-1))./sem(2:end) );
    %     semsnr1(2,p) = std( (diff(muz)./muz(1:end-1))./sem(2:end) ) / sqrt(length(sem(2:end)));
    snr2(2,p) = mean(diff(muz))/mean(sem(2:end));
    
    figure(Esfig), subplot(122), hold on
    plot(p*ones(1,length(es1multi)),es1multi,symb{p})
    %     plot(es1multi,symb{p})
    
    %t-test
    x = es1one_{p}'; y = es1multi_{p}';
    [h,pv] = ttest2(x,y,'Alpha',0.05/4)
    
end
figure(Zfig)
set(gca,'FontSize',14)
set(findall(gcf,'type','text'),'fontSize',14), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',14), %ylim([-1000 20])
xlabel('Block #'), ylabel('Z-score')
% legend('P01','P02','P03','P04')
line([0 Ns + 2],[2 2],'LineWidth',1,'Color',[0 0.7 0])
line([0 Ns + 2],[-2 -2],'LineWidth',1,'Color',[0 0.7 0])

%plot effect size
figure, hold on
bp = bar(snr1');
set(bp(1),'FaceColor',[1 0 0]);
set(bp(2),'FaceColor',[0 0.8 0]);
errorbar((1:4)-0.1,snr1(1,:),semsnr1(1,:),'ok','LineWidth',1)
errorbar((1:4)+0.1,snr1(2,:),semsnr1(2,:),'ok','LineWidth',1)
set(gca,'Xtick',1:4), %ylim([-0.2 1])

figure, hold on
bp = bar(mean(snr2,2));
errorbar(1:2,mean(snr2,2),std(snr2,[],2)/2)
set(bp(1),'FaceColor',[1 0 0]);
% set(bp(2),'FaceColor',[0 0.8 0]);
set(gca,'Xtick',1:4), %ylim([-0.2 1])
[h,pv] = ttest2(snr2(1,:)',snr2(2,:)','Alpha',0.05)



%are z-scores significantly different at EOT?
%bootstrap
bootf=figure;
for p =1:4
%     Z1 = [Z1 ; IponeAll{p}{end}(:,OneFeat)];
%     Zm = [Zm ; IpMultiAll{p}{end}];
% g = [g;p*ones(length(IpMultiAll{p}{end}),1)];
    Z1 = IponeAll{p}{end}(:,OneFeat);
    ZM = IpMultiAll{p}{end};
    [bootstat1,bootsam1] = bootstrp(nboot,@(x)mean(x),Z1);
    [bootstatM,bootsamM] = bootstrp(nboot,@(x)mean(x),ZM);
    ci1(:,p) = bootci(nboot,{@(x)mean(x),Z1})
    ciM(:,p) = bootci(nboot,{@(x)mean(x),ZM})
    subplot(121), hold on
    errorbar(p,mean(bootstat1),mean(bootstat1)-ci1(1,p),ci1(2,p)-mean(bootstat1),symb{p})
    subplot(122), hold on
    errorbar(p,mean(bootstatM),mean(bootstatM)-ciM(1,p),ciM(2,p)-mean(bootstat1),symb{p})
end

%Wilcoxon signed rank test
for p =1:4
    Z1 = IponeAll{p}{end}(:,OneFeat);
    ZM = IpMultiAll{p}{end};
    [p1,h1] = signrank(Z1+2,[],'tail','left','alpha',0.05) 
    [pM,hM] = signrank(ZM+2,[],'tail','left','alpha',0.05) 
end



Z1 = []; ZM = []; g = [];
for p =1:4
    Z1 = [Z1 ; IponeAll{p}{end}(:,OneFeat)];
    ZM = [ZM ; IpMultiAll{p}{end}];
    g = [g;p*ones(length(IpMultiAll{p}{end}),1)];
end
figure, subplot(121)
boxplot(Z1,g)
subplot(122), boxplot(ZM,g)


%% Use mean across 6 1-minute windows
load ./matFiles/IpMultiExperts.mat
load ./matFiles/IpOneExperts.mat
OneFeat = 4; %what single feature is used
nboot = 999;
%MULTI FEATURES
figure,subplot(121),  hold on,  title('Z_{\psi} - All Features')

for p = 1:length(IpMulti)
    clear Im Isem
    
    Im = IpMulti{p};
    %     errorbar(1:Nb,Imm,Isem,symb{p},'Linewidth',2), ylim([-1000 20])
    %plot all sessions
    plot(1:length(Im),Im,symb{p},'Linewidth',2);
    
    %mean S-S impro
    %     I{p}(:,1) = Im';       %save patient z-scores
    SSI = diff(Im);   %S-S improvement
    SSIm(1,p)= mean(SSI); SSIsd(1,p) = std(SSI);
    SSIm2(1,p) =SSIm(1,p)./SSIsd(1,p)./sqrt(length(SSI));
    
    %CI
    bootstat1 = bootstrp(nboot,@SSimpro,SSI'); %bootstat1=bootstat1(~isinf(bootstat1));
    SSIm2(1,p) = mean(bootstat1); SSIm2SE(1,p) = std(bootstat1);
    SSIm2CI(:,p,1) = bootci(nboot,{@SSimpro,SSI'},'alpha',0.1);
    
    
end
set(gca,'FontSize',24)
set(findall(gcf,'type','text'),'fontSize',24), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',24), %ylim([-1000 20])
xlabel('Block #'), ylabel('Z-score')
legend('P01','P02','P03','P04')
line([0 30],[2 2],'LineWidth',1,'Color',[0 0.7 0])
line([0 30],[-2 -2],'LineWidth',1,'Color',[0 0.7 0])


%%  ONE FEATURE
% IpOfig = figure('name','z-scores One Feature'); hold on
subplot(122), hold on, title('Z_{\psi} - Steps')
for p = 1:length(IpOne)
    clear Im Isem
    
    Im= IpOne{p}(:,OneFeat);
    plot(1:length(Im),Im,symb{p},'Linewidth',2);
    
    
    %     errorbar(1:Nb,Imm,Isem,symb{p},'Linewidth',2), ylim([-140 5])
    % %     line([0 Nb+1],[0 0],'LineWidth',1,'Color',[0 0.7 0])
    %
    %     %plot all sessions
    %     %     plot(1:length(Im),Im,[symb{p}([1 3]) '-.'],'Linewidth',3);
    %
    %S-S impro
    %     I{p}(:,2) = Imm';
    SSI = diff(Im);   %S-S improvement
    SSIm(2,p)= mean(SSI); SSIsd(2,p) = std(SSI);
    SSIm2(2,p) =SSIm(2,p)./SSIsd(2,p)./sqrt(length(SSI));
    
    %CI
    bootstat1 = bootstrp(nboot,@SSimpro,SSI'); %bootstat1=bootstat1(~isinf(bootstat1));
    SSIm2(2,p) = mean(bootstat1); SSIm2SE(2,p) = std(bootstat1);
    SSIm2CI(:,p,2) = bootci(nboot,{@SSimpro,SSI'},'alpha',0.1);
    
end
set(gca,'FontSize',24)
set(findall(gca,'type','text'),'fontSize',24), %ylim([-1000 20])
set(findall(gcf,'type','text'),'fontSize',24), %ylim([-1000 20])
xlabel('Block #'), ylabel('Z-score')
line([0 30],[2 2],'LineWidth',1,'Color',[0 0.7 0])
line([0 30],[-2 -2],'LineWidth',1,'Color',[0 0.7 0])

%plot SS impro
figure, hold on
% plot(1:4,SSIm2(1,:),'o-g','LineWidth',2)
% plot(1:4,SSIm2(2,:),'o-r','LineWidth',2)
%CI S
errorbar((1:4)-0.1,SSIm2(1,:),SSIm2(1,:)-SSIm2CI(1,:,1),SSIm2CI(2,:,1)-SSIm2(1,:),'o-g','LineWidth',2)
errorbar((1:4)+0.1,SSIm2(2,:),SSIm2(2,:)-SSIm2CI(1,:,2),SSIm2CI(2,:,2)-SSIm2(2,:),'o-r','LineWidth',2)
%% SEM
bp = bar(SSIm2');
set(bp(1),'FaceColor',[0 0.8 0]);
set(bp(2),'FaceColor',[1 0 0]);
errorbar((1:4)-0.1,SSIm2(1,:),SSIm2(1,:)-SSIm2CI(1,:,1),SSIm2CI(2,:,1)-SSIm2(1,:),'ok','LineWidth',1)
errorbar((1:4)+0.1,SSIm2(2,:),SSIm2(2,:)-SSIm2CI(1,:,2),SSIm2CI(2,:,2)-SSIm2(2,:),'ok','LineWidth',1)
set(gca,'Xtick',1:4), ylim([-0.2 1])

% errorbar((1:4)-0.1,SSIm2(1,:),SSIm2SE(1,:),'o-g','LineWidth',2)
% errorbar((1:4)+0.1,SSIm2(2,:),SSIm2SE(2,:),'o-r','LineWidth',2)

set(gca,'FontSize',24)
set(findall(gcf,'type','text'),'fontSize',24), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',24), %ylim([-1000 20])
xlabel('Patient #'), ylabel('Z-score sensitivity \gamma'),
legend('All features','Steps only')

%Relative
% figure, hold on
% errorbar(1:4,SSIrelm(1,:),SSIrelSE(1,:),'o-g','LineWidth',2)
% errorbar(1:4,SSIrelm(2,:),SSIrelSE(2,:),'o-r','LineWidth',2)

%bootstrap the difference of the effect size
for p = 1:4
    SSId = diff(I{p});
    bootstatDiff = bootstrp(nboot,@SSimproDiff,SSId);
    meandiff(p) = mean(bootstatDiff);
    CIdiff(:,p) = bootci(nboot,{@SSimproDiff,SSId},'alpha',0.1);
end

figure, hold on
bp = bar(meandiff);
errorbar((1:4),meandiff,meandiff-CIdiff(1,:),CIdiff(2,:)-meandiff,'ok','LineWidth',1)
set(gca,'Xtick',1:4), %ylim([-0.2 1])
set(gca,'FontSize',24)
set(findall(gcf,'type','text'),'fontSize',24), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',24), %ylim([-1000 20])
xlabel('Patient #'), ylabel('Z-score sensitivity \gamma Difference'),

