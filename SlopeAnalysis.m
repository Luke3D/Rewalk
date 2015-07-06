%plot Multi and OneFeat indices
close all
clear all
% symb = {'b-o','r-s','c-*','m-x'}; %symbol used to plot data for that patient
symb = {'b-o','r-o','c-o','m-o'}; %symbol used to plot data for that patient
load ./matFiles/IpMultiTHR.mat
load ./matFiles/IpOneTHR.mat
OneFeat = 4; %what single feature is used
Nboot = 999;
%MULTI FEATURES 
figure, subplot(121), hold on,  title('Z_{\psi} - All Features')

for p = 1:length(IpMulti)
    clear Imm Isem
    
    Im = medfilt1(IpMulti{p}(1,:));      
 
    %bin data in blocks to plot with error bars (SEM over sessions)
    Sxb = 2; %# of sessions per block
    Nb = floor(length(Im)/Sxb);  %# of blocks  
    Nbb = mod(length(Im),Sxb);
    for k = 1:Nb
        Imm(k) = mean(Im((k*Sxb)-(Sxb-1):k*Sxb));
        Isem(k) = std(Im((k*Sxb)-(Sxb-1):k*Sxb))/sqrt(Sxb);
        if Nbb > 0 && k == Nb
            Imm(k) = mean(Im((k*Sxb)-(Sxb-1):(k*Sxb)+Nbb));
            Isem(k) = std(Im((k*Sxb)-(Sxb-1):(k*Sxb)+Nbb))/sqrt(Sxb+Nbb);
        end
    end
    
    %corr coeff
    [rhoI(1,p),prhoI(1,p)] = corr((1:Nb)',Imm');
    
    errorbar(1:Nb,Imm,Isem,symb{p},'Linewidth',2), ylim([-1000 20])
    %plot all sessions
%     plot(1:length(Im),Im,symb{p},'Linewidth',3);    

    %mean S-S impro
    I{p}(:,1) = Imm';       %save patient z-scores
    SSI = diff(Imm(1,:));   %S-S improvement
    SSIm(1,p)= mean(SSI); SSIsd(1,p) = std(SSI);
    SSIm2(1,p) =SSIm(1,p)./SSIsd(1,p)./sqrt(length(SSI));
    
    %CI
    bootstat1 = bootstrp(Nboot,@SSimpro,SSI'); %bootstat1=bootstat1(~isinf(bootstat1));
    SSIm2(1,p) = median(bootstat1); SSIm2SE(1,p) = std(bootstat1);
    SSIm2CI(:,p,1) = bootci(Nboot,{@SSimpro,SSI'},'alpha',0.1);

    %mean relative SSimpro
    SSIrel = diff(Imm(1,:))./abs(Imm(1:end-1));
    SSIrelm(1,p) = mean(SSIrel); SSIrelSE(1,p) = std(SSIrel)/length(SSIrel);
    
end
set(gca,'FontSize',24)
set(findall(gcf,'type','text'),'fontSize',24), %ylim([-1000 20])
set(findall(gca,'type','text'),'fontSize',24), %ylim([-1000 20])
xlabel('Block #'), ylabel('Z-score')
legend('P01','P02','P03','P04')
line([0 10],[2 2],'LineWidth',1,'Color',[0 0.7 0])
line([0 10],[-2 -2],'LineWidth',1,'Color',[0 0.7 0])


%ONE FEATURE
% IpOfig = figure('name','z-scores One Feature'); hold on
subplot(122), hold on, title('Z_{\psi} - Walk Time only')
for p = 1:length(IpOne)
    clear Imm Isem

    Im = medfilt1((IpOne{p}(:,OneFeat)));
    
    %bin data in blocks
    Nb = floor(length(Im)/Sxb);  %# of blocks  
    Nbb = mod(length(Im),Sxb);
    for k = 1:Nb
        Imm(k) = mean(Im((k*Sxb)-(Sxb-1):k*Sxb));
        Isem(k) = std(Im((k*Sxb)-(Sxb-1):k*Sxb))/sqrt(Sxb);
        if Nbb > 0 && k == Nb
            Imm(k) = mean(Im((k*Sxb)-(Sxb-1):(k*Sxb)+Nbb));
            Isem(k) = std(Im((k*Sxb)-(Sxb-1):(k*Sxb)+Nbb))/sqrt(Sxb+Nbb);
        end
    end
    
    %corr coeff
    [rhoI(2,p),prhoI(2,p)] = corr((1:Nb)',Imm');
    
    errorbar(1:Nb,Imm,Isem,symb{p},'Linewidth',2), ylim([-140 5])
%     line([0 Nb+1],[0 0],'LineWidth',1,'Color',[0 0.7 0])

    %plot all sessions
    %     plot(1:length(Im),Im,[symb{p}([1 3]) '-.'],'Linewidth',3);
    
    %S-S impro
    I{p}(:,2) = Imm';
    SSI = diff(Imm(1,:));   %S-S improvement
    SSIm(2,p)= mean(SSI); SSIsd(2,p) = std(SSI);
    SSIm2(2,p) =SSIm(2,p)./SSIsd(2,p)./sqrt(length(SSI));

    %CI
    bootstat1 = bootstrp(Nboot,@SSimpro,SSI'); %bootstat1=bootstat1(~isinf(bootstat1));
    SSIm2(2,p) = median(bootstat1); SSIm2SE(2,p) = std(bootstat1);
    SSIm2CI(:,p,2) = bootci(Nboot,{@SSimpro,SSI'},'alpha',0.1);

    %mean relative SSimpro
    SSIrel = diff(Imm(1,:))./abs(Imm(1:end-1));
    SSIrelm(2,p) = mean(SSIrel); SSIrelSE(2,p) = std(SSIrel)/length(SSIrel);

end
set(gca,'FontSize',24)
set(findall(gca,'type','text'),'fontSize',24), %ylim([-1000 20])
set(findall(gcf,'type','text'),'fontSize',24), %ylim([-1000 20])
xlabel('Block #'), ylabel('Z-score')
line([0 10],[2 2],'LineWidth',1,'Color',[0 0.7 0])
line([0 10],[-2 -2],'LineWidth',1,'Color',[0 0.7 0])

%plot SS impro
figure, hold on
% plot(1:4,SSIm2(1,:),'o-g','LineWidth',2)
% plot(1:4,SSIm2(2,:),'o-r','LineWidth',2)
%CI S
% errorbar((1:4)-0.1,SSIm2(1,:),SSIm2(1,:)-SSIm2CI(1,:,1),SSIm2CI(2,:,1)-SSIm2(1,:),'o-g','LineWidth',2)
% errorbar((1:4)+0.1,SSIm2(2,:),SSIm2(2,:)-SSIm2CI(1,:,2),SSIm2CI(2,:,2)-SSIm2(2,:),'o-r','LineWidth',2)
%SEM
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
legend('All features','Walk Time only')

%Relative
% figure, hold on
% errorbar(1:4,SSIrelm(1,:),SSIrelSE(1,:),'o-g','LineWidth',2)
% errorbar(1:4,SSIrelm(2,:),SSIrelSE(2,:),'o-r','LineWidth',2)

%bootstrap the difference of the effect size
for p = 1:4
    SSId = diff(I{p});
    bootstatDiff = bootstrp(Nboot,@SSimproDiff,SSId); 
    meandiff(p) = median(bootstatDiff);
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





% %% Plot Index with CI
% symb = {'b-o','r-s','c-*','m-x'}; %symbol used to plot data for that patient
% load IpMultiTHR_CI.mat
% load IpOneTHR_CI.mat
% figure
% for p = 1:length(IpMulti)
%     I = IpMulti{p}(1,:);
%     UciI = IpMulti{p}(3,:); LciI = IpMulti{p}(4,:);
%     subplot(121), hold on, title('Z-score - Multi Features')
%     errorbar(1:length(I),I,LciI,UciI,[symb{p}],'LineWidth',2);
% end
% 
% for p = 1:length(IpOne)
%     I = IpOne{p}(1,:);
%     UciI = IpOne{p}(3,:); LciI = IpOne{p}(4,:);
%     subplot(122), hold on, title('Z-score - One Feature')
%     errorbar(1:length(I),I,LciI,UciI,symb{p},'LineWidth',2);
% end
% %% Weighted Session by session impro (weighted mean)
% for p = 1:length(IpMulti)
%     clear SSI varSSI
%     Im = (IpMulti{p});  
%     VarI = Im(2,:);    %variance of z-scores I
%     SSI = diff(Im(1,:));   %S-S improvement
%     for t = 1:length(VarI)-1
%         varSSI(t) = VarI(t+1)-VarI(t);
%     end
%     wSSI = sum(SSI.*varSSI.^(-2))/sum(varSSI.^(-2));    %weighed mean of S-S impro
%     VarwSSI = 1/sum(varSSI.^(-2));
%     
%     wSSI2(1,p) = wSSI/sqrt(VarwSSI);    
%    
% end
% 
% for p = 1:length(IpOne)
%     clear SSI varSSI
%     Im = (IpOne{p});  
%     VarI = Im(2,:);    %variance of z-scores I
%     SSI = diff(Im(1,:));   %S-S improvement
%     for t = 1:length(VarI)-1
%         varSSI(t) = VarI(t+1)-VarI(t);
%     end
%     wSSI = sum(SSI.*varSSI.^(-2))/sum(varSSI.^(-2));    %weighed mean of S-S impro
%     VarwSSI = 1/sum(varSSI.^(-2));
%     
%     wSSI2(2,p) = wSSI/sqrt(VarwSSI);    
%    
% end
% 
% figure('name','Multi Feature'), hold on
% plot(1:4,wSSI2(1,:),'o-g')
% plot(1:4,wSSI2(2,:),'o-r')
% xlabel('Patient'), ylabel('S-S Impro in Z-score')
% 
% 
% 
% %% Session by session impro
% 
% for p = 1:length(IpMulti)
%     Im = (IpMulti{p});  %filter with median filter
%     SSI = diff(Im);   %mean S-S improvement
%     SSIm(1,p)= mean(SSI); SSIsd(1,p) = std(SSI);
%     SSIm2(1,p) =SSIm(1,p)./SSIsd(1,p)./sqrt(length(SSI))
% end
% figure('name','Multi Feature'), hold on
% % errorbar(1:4,SSIm(1,:),SSIsd(1,:),'g','LineWidth',2)
% plot(1:4,SSIm2(1,:),'o-g')
% xlabel('Patient'), ylabel('S-S Impro in Z-score')
% 
% for p = 1:length(IpOne)
%     Io = (IpOne{p}(:,OneFeat));
%     SSI = diff(Io);   %mean S-S improvement
%     SSIm(2,p)= mean(SSI); SSIsd(2,p) = std(SSI);
%     SSIm2(2,p) =SSIm(2,p)./SSIsd(2,p)./sqrt(length(SSI))
% 
% end
% 
% % figure('name','One Feature')
% % errorbar(1:OneFeat,SSIm(2,:),SSIsd(2,:),'r','LineWidth',2)
% plot(1:4,SSIm2(2,:),'o-r')
% set(gca,'FontSize',16)
% set(findall(gcf,'type','text'),'fontSize',16), %ylim([-1000 20])
% legend('Multi','One Feature')
% 
% %% Fit Exponential to each index
% 
% %exp1
% myfittype = fittype('P0*exp(a*x)',...
%     'dependent',{'y'},'independent',{'x'},...
%     'coefficients',{'a','P0'})
% %exp2
% % myfittype = fittype('(P0-Pinf)*exp(a*x)+Pinf',...
% %     'dependent',{'y'},'independent',{'x'},...
% %     'coefficients',{'a','P0','Pinf'})
% % fitopt = fitoptions(myfittype); 
% % fitopt.Upper = [0 0 0];
% 
% % p = 1;  %patient #
% fits = {};
% for p = 1:length(IpMulti{p})
%     clear x y
% %     y = medfilt1(IpMulti{p},3);
%     y = medfilt1(IpOne{p}(:,end-1),3); y = y';
%     x = 1:length(y);
%     
%     fits{p} = fit(x',y',myfittype)
% %     figure(IpMfig), hold on
%     figure(IpOfig), hold on
%     plot(fits{p},x,y)
%     
% end
% 
% %% Fit Exponential model and compute CI of slope for different sessions
% p = 1;  %patient #
% figure, hold on
% % for p = 1:length(IpMulti)
% 
%     clear a CCI pV
% 
% %      variab=medfilt1(IpMulti{p},3);
% variab=medfilt1(IpOne{p}(:,end-1),3);
% 
%     for i=5:length(variab)
%         i
%         clear DATA
%         DATA(:,1)=1:i;%length(variab);
%         DATA(:,2)=variab(1:i);
%         
%         % a(i) = efit(DATA);  %Fit exp of the form f(x) = a*exp(b*x)
%         f = fit(DATA(:,1),DATA(:,2),'exp1');
%         a(i) = f.a;
%         %confidence interval (100*(1-alpha)% confidence interval)
% %         CI = confint(f,0.95);
% %         aCI(i,:) = CI(:,1);
%         
%         a_CI = bootci(999,{@efit,DATA},'alpha',0.1);
%         [bootstat,bootsam] = bootstrp(999,@efit,DATA);
%         pV(i)=sum(bootstat<0)/999;
%         CCI(i,:)=a_CI;
%         
%         %plot the fit and the data
%         plot(f,DATA(:,1),DATA(:,2),symb{p})
%         
%     end
% % end
% 
% figure(8),
% hold on
% errorbar(1:length(variab),a,a'-aCI(:,1),aCI(:,2)-a','Color',[0,1,0])
% line([0,20],[0,0])
% % title('08')
% figure(80)
% hold on
% plot(1:length(variab),pV,'Color',[0,1,0])
% 
% 
% %% Fit linear model and compute CI of slope for different sessions
% % p = 1;  %patient #
% % clear SS CCI pV
% % 
% % variab=medfilt1(IpMulti{p},3);
% % % variab=medfilt1(Ip02_OneFeat,3);
% % 
% % for i=5:length(variab)
% %     i
% %     clear DATA
% % DATA(:,1)=1:i;%length(variab);
% % DATA(:,2)=variab(1:i);
% % %R2
% % SSlope=@(x)(Slope(x));
% % Slope_result = SSlope(DATA)
% % %confidence interval (100*(1-alpha)% confidence interval)
% % Slope_CI = bootci(999,{SSlope,DATA},'alpha',0.05)
% % [bootstat,bootsam] = bootstrp(999,@Slope,DATA);
% % pV(i)=sum(bootstat<0)/999;
% % CCI(i,:)=Slope_CI;
% % SS(i)= Slope_result;
% % end
% % figure(8), 
% % hold on
% % errorbar(1:length(variab),SS,SS'-CCI(:,1),CCI(:,2)-SS','Color',[1,0,0])
% % line([0,20],[0,0])
% % title('08')
% % figure(80)
% % hold on
% plot(1:length(variab),pV,'Color',[1,0,0])