%plot Multi and OneFeat indices
close all
clear all
symb = {'b-o','r-o','c-o','m-o'}; %symbol used to plot data for that patient

%% Use each minute window
load ./matFiles/IpMultiExpertsAll.mat
load ./matFiles/IpOneExpertsAll.mat
OneFeat = 4; %what single feature is used
Sxb = 2; %# of sessions per block
Nboot = 999;
%ONE FEATURE
figure,subplot(121),  hold on,  title('Z_{\psi} - Steps only')
for p = 1:length(IponeAll)
    
    clear muz sdz sem Imnew
    
    %Combine sessions into blocks
    Im = IponeAll{p};
    Nb = floor(length(Im)/Sxb);  %# of blocks  
    Nbb = mod(length(Im),Sxb);
    for b = 1:Nb
        Imnew{b} = cell2mat( Im((b*Sxb)-1:(b*Sxb))' );
    end
    if Nbb > 0
        Imnew{end} = [Imnew{end};cell2mat(Im(end))];
    end   
    IponeAll{p} = Imnew;    %restructured in blocks
    
    Ns = length(IponeAll{p});
    for s = 1:Ns
       muz(s) = mean(IponeAll{p}{s}(:,OneFeat));
       sdz(s) = std(IponeAll{p}{s}(:,OneFeat));
       sem(s) = sdz(s)/sqrt(size(IponeAll{p}{s},1));
    end
        
    %plot mean and std dev of z-score for each session
    errorbar(1:Ns,muz,sdz,symb{p},'Linewidth',2), 
    ylim([-600 40])   

    %compute effect size and sem of effect size (for measure 1)
    snr1(1,p) = mean(diff(muz)./sem(2:end));
    semsnr1(1,p) = std( diff(muz)./sem(2:end) ) / sqrt(length(sem(2:end)));
    snr2(1,p) = mean(diff(muz))/mean(sem(2:end));
    %bootstrap z-score to obtain 90% CI
    %CI
%     bootstat1 = bootstrp(Nboot,@SSimpro,SSI'); %bootstat1=bootstat1(~isinf(bootstat1));
%     SSIm2(1,p) = mean(bootstat1); SSIm2SE(1,p) = std(bootstat1);
%     SSIm2CI(:,p,1) = bootci(Nboot,{@SSimpro,SSI'},'alpha',0.1);

end
set(gca,'FontSize',14)
set(findall(gcf,'type','text'),'fontSize',14), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',14), %ylim([-1000 20])
xlabel('Block #'), ylabel('Z-score')
legend('P01','P02','P03','P04')
line([0 Ns + 2],[2 2],'LineWidth',1,'Color',[0 0.7 0])
line([0 Ns + 2],[-2 -2],'LineWidth',1,'Color',[0 0.7 0])

%ALL FEATURES
subplot(122),  hold on,  title('Z_{\psi} - All features')
for p = 1:length(IpMultiAll)
    
    clear muz sdz sem Imnew
    
%     %Combine sessions into blocks
    Im = IpMultiAll{p};
    Nb = floor(length(Im)/Sxb);  %# of blocks  
    Nbb = mod(length(Im),Sxb);
    for b = 1:Nb
        Imnew{b} = cell2mat( Im((b*Sxb)-1:(b*Sxb))' );
    end
    if Nbb > 0
        Imnew{end} = [Imnew{end};cell2mat(Im(end))];
    end
    IpMultiAll{p} = Imnew;    %restructured in blocks
    
    
    Ns = length(IpMultiAll{p});
    for s = 1:Ns
       muz(s) = mean(IpMultiAll{p}{s});
       sdz(s) = std(IpMultiAll{p}{s});
       sem(s) = sdz(s)/sqrt(size(IpMultiAll{p}{s},1));
    end
    
    %plot mean and std dev of z-score for each session
    errorbar(1:Ns,muz,sdz,symb{p},'Linewidth',2),
    ylim([-600 40])   
    
    %compute effect size and sem of effect size (for measure 1)
    snr1(2,p) = mean( diff(muz)./sem(2:end) );
    semsnr1(2,p) = std( diff(muz)./sem(2:end) ) / sqrt(length(sem(2:end)));
    snr2(2,p) = mean(diff(muz))/mean(sem(2:end));
    
    
    
end
set(gca,'FontSize',14)
set(findall(gcf,'type','text'),'fontSize',14), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',14), %ylim([-1000 20])
xlabel('Block #'), ylabel('Z-score')
legend('P01','P02','P03','P04')
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
bp = bar(snr2');
set(bp(1),'FaceColor',[1 0 0]);
set(bp(2),'FaceColor',[0 0.8 0]);
set(gca,'Xtick',1:4), %ylim([-0.2 1])


%% Use mean across 6 1-minute windows
load ./matFiles/IpMultiExperts.mat
load ./matFiles/IpOneExperts.mat
OneFeat = 4; %what single feature is used
Nboot = 999;
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
    bootstat1 = bootstrp(Nboot,@SSimpro,SSI'); %bootstat1=bootstat1(~isinf(bootstat1));
    SSIm2(1,p) = mean(bootstat1); SSIm2SE(1,p) = std(bootstat1);
    SSIm2CI(:,p,1) = bootci(Nboot,{@SSimpro,SSI'},'alpha',0.1);

    
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
    bootstat1 = bootstrp(Nboot,@SSimpro,SSI'); %bootstat1=bootstat1(~isinf(bootstat1));
    SSIm2(2,p) = mean(bootstat1); SSIm2SE(2,p) = std(bootstat1);
    SSIm2CI(:,p,2) = bootci(Nboot,{@SSimpro,SSI'},'alpha',0.1);

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
    bootstatDiff = bootstrp(Nboot,@SSimproDiff,SSId); 
    meandiff(p) = mean(bootstatDiff);
    CIdiff(:,p) = bootci(Nboot,{@SSimproDiff,SSId},'alpha',0.1);
end

figure, hold on
bp = bar(meandiff);
errorbar((1:4),meandiff,meandiff-CIdiff(1,:),CIdiff(2,:)-meandiff,'ok','LineWidth',1)
set(gca,'Xtick',1:4), %ylim([-0.2 1])
set(gca,'FontSize',24)
set(findall(gcf,'type','text'),'fontSize',24), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',24), %ylim([-1000 20])
xlabel('Patient #'), ylabel('Z-score sensitivity \gamma Difference'), 

