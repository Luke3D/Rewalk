% % ACTIGRAPH DATA ANALYSIS
% Loads the .mat file containing raw acceleration data for further analysis
% Make sure to indicate the correct sampling frequency for the data set

Fs = 100;   %Sampling freq
Resample = 1;   %Flag to indicate Resampling at 30 Hz

acc = cell2mat(accraw(:,2:end));

%remove zeros
% [r,c] = find(acc(:,1) == 0 & acc(:,2) == 0 & acc(:,3) == 0);
% acc(r,:) = [];

%plot raw data
t = 0:1/Fs:(length(acc)/Fs-1/Fs);
figure
set(0,'defaultlinelinewidth',2)
plot(t,acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Time [s]'), ylabel('acc [g]')

%% Extract Walk accelerations
% Takes the start and end time of the 6mw test from the TestTimes cell
% array.  The saved TestTimes arrays for previous tests are found in the
% TestTimes folder under the appropriate patient folder.

% close all
acc = accraw;

% num=2;% number of section of trial you wish to evaluate (See TestTimes cell array from the appropriate Cutoff Time file)
% 
% Starttime = [TestTimes{num,1} ':00.000']; 
% Endtime = [TestTimes{num,2} ':00.000'];

Starttime = '11:33:00.000';
Endtime = '11:40:00.000';

i = 0; t = 0;
while i == 0
    t = t+1;
    i = strcmp(acc{t}(length(acc{t})-11:end),Starttime);
    if t+1>length(acc)
       i=1;
    end
end
indstart = t;

i = 0; t = 0;
while i == 0
    t = t+1;
    i = strcmp(acc{t}(length(acc{t})-11:end),Endtime);
    if t+1>length(acc)
       i=1;
    end
end
indend = t; 

acc = acc(indstart:indend,:);
acc = cell2mat(acc(:,2:end));

%RESAMPLE DATA TO 30 HZ
if Resample
    x=0:1/Fs:(length(acc)/Fs-1/Fs);
    xq=0:1/30:(length(acc)/Fs-1/Fs);
    acc=interp1(x, acc, xq);
    Fs = 30;
end

acctrim=acc;
% plot with time in secs
figure
t = 0:1/Fs:(length(acc)/Fs-1/Fs);
plot(t,acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Time [s]'), ylabel('acc [g]')

%plot with samples shown
figure
plot(acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
ylabel('acc [g]')

%% Extract Steps
% searches for possible walking sections in the previously selected data
% range and outputs the resulting start and end indices in the matrix "times"
% Copy the values from "times" into the TestTimes array to plot individual
% walking sections at the next step

avg=mean(acc(:,1));
x=0;
step=1;
prevstep=1;
count=0;
times=[0, 0];

for n=1:length(acc)
    x=0;
    if step==1
        if abs(acc(n)-avg)>.1
            for m=1:10
                x=x+abs(acc(n+m)-avg);
            end
            if x>.1
                step=0;
            end
        end
    end
    if step==0
        for m=1:10
            x=x+abs(acc(n+m)-avg);
        end
        if x<.06
            step=1;
        end
    end
    if prevstep~=step
        times(count+1,step+1)=n;
        if step
            count=count+1;
        end
    end
    prevstep=step;
end
done=0;
times2=[0, 0];
while done==0
for n=1:count
    if ~(200<(times(n,2)-times(n,1)) & (times(n,2)-times(n,1))<18000)
        if n==1
            times2=times([2:end],:);
            times=times2;
        else
            times2=times([1:n-1,n+1:end],:);
            times=times2;
        end
        break
    end
    if n==count
        done=1;
    end
end
[count,~]=size(times);
end


%% select a specific walk sequence
% Plots the walking section with the start and end indices indicated by the
% row of TestTimes corresponding to the value of "num"

% close all
num=1;
t1 = TestTimes{num,3}; t2 = TestTimes{num,4};
accwalk = acctrim(t1:t2,:);
figure
plot(accwalk,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
ylabel('acc [g]')

% accCheck=sum(accwalk.^2,2);
% figure
% plot(accCheck)
% acc = accwalk;
%% Compute lateral and frontal tilt from accelerometer data 

%tilt (frontal) inclination 
% phi = (180/pi)*atan2(accwalk(:,2),accwalk(:,1));    %ATAN2 does not suffer from sensitivity issues - OLD Actigraph (1 = g, 2 = x)
phi = (180/pi)*atan2(accwalk(:,1),accwalk(:,2));    %ATAN2 does not suffer from sensitivity issues - NEW Actigraph (2 = g, 1 = x)

%roll (lateral) inclination
% alpha = (180/pi)*atan2(accwalk(:,3),accwalk(:,1)); %OLD Actigraph (1 = g)
alpha = (180/pi)*atan2(accwalk(:,3),accwalk(:,2));  %NEW Actigraph (2 = g)


figure; 
subplot(121)            
plot(phi), title('frontal angle')
subplot(122)
plot(alpha), title('lateral angle')
%% simple spectral analysis of the signal
fmax = 5;           %fmax to plot
% fsmooth = 0.5;        %kernel size in freq to smooth
L = length(phi);    %signal length
df = Fs/2/(L/2+1);  %min freq step
Y = fft(phi,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
Pyy = Y.*conj(Y)/L; %power spectrum
Pyy = Pyy(3:L/2+1);
% Pyy = Pyy(3:round(fmax/df));
f = Fs/2*linspace(0,1,L/2+1);   %frequency axis
% f = f(3:round(fmax/df));                   %skip DC (0 Hz)
f = f(3:end);                   %skip DC (0 Hz)
% Pyy = smooth(Pyy,fsmooth/df);
% Pyy = Pyy/trapz(f,Pyy); %normalize power spectrum
figure; 
subplot(121), hold all
plot(f,Pyy,'LineWidth',2)
% plot(f,2*abs(Y(1:L/2+1)))     %Fourier magnitude
% plot(f(3:L/2+1),Pyy(3:L/2+1),'LineWidth',2)
title('PSD - \phi')
xlabel('Frequency (Hz)');  xlim([0 fmax])
ylabel('Magnitude'); 


[~,is] = max(Pyy(3:end)); %approx count of steps per sec 
Stepf = f(is+2);

Y = fft(alpha,L);
Pyy2 = Y.*conj(Y)/L;
Pyy2 = Pyy2(3:L/2+1); 
% Pyy2 = Pyy2(3:round(fmax/df)); 
% Pyy2 = smooth(Pyy2,fsmooth/df);
Pyy2 = Pyy2/trapz(f,Pyy2);  %normalize power spectrum
subplot(122)
plot(f,Pyy2,'LineWidth',2)
title('PSD - \alpha')
xlabel('Frequency (Hz)');  xlim([0 fmax])
ylabel('Magnitude'); 

Pyy=[Pyy, Pyy2];

%Compute Kurtosis of freq spectrum
Pyy = Pyy(:,1);
clear P
fc = [];
Nbins = 100; 
xm = 1; xM = length(f);
x = round(linspace(1,length(f),Nbins));
%Prob for each bin
for k = 1:length(x)-1
    P(k) = trapz(f(x(k):x(k+1)),Pyy(x(k):x(k+1)));  %CDF 
    fc = [fc; (f(x(k))+f(x(k+1)))/2];   %bin centers
end
%check
sum(P);

%sample from the distribution
samples = datasample(fc,5000,'Weights',P);
figure, hist(samples,Nbins);
kurt = kurtosis(samples)
sdev = std(samples)

%% Low pass filter 
%phi
t = 0:1/Fs:(length(accwalk)/Fs-1/Fs);

ft = 1.5; %cut-off freq
[B,A] = butter(2, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
phif = filtfilt(B,A,phi);   %the filtered version of the signal
figure('name','Filtered angles - Actigraph');
subplot(121)
plot(t,phif); hold on
xlabel('Time [s]')
ylabel('\phi Trunk [deg]')
xlim([0 30])
% xlim([0 t(end)+1]);  
% ylim([-12 20]);

% %alpha
ft = 0.5;
[B,A] = butter(2, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
alphaf = filtfilt(B,A,alpha);   %the filtered version of the signal
subplot(122)
plot(t,alphaf);  
xlim([0 30])
% xlim([0 t(end)+1]); 
% ylim([-12 20]);
xlabel('Time [s]')
ylabel('\alpha Trunk [deg]')

%% Compute measure of smoothness
% dphif = diff(phif);
% ddphif = diff(dphif);
% T = length(ddphif)/Fs;
% smoothness = 1/(trapz(ddphif.^2)/T)
% 
% [rrphi,lagsphi] = xcorr(phif,'coeff');
% [rralpha,lagsalpha] = xcorr(alphaf,'coeff');
% 
% rrphi = rrphi(lagsphi>0);
% lagsphi = lagsphi(lagsphi>0);
% 
% figure
% plot(lagsphi/Fs,rrphi)
% xlabel('Lag (s)')



%% Plot ellipse whose axes are variance of tilt angles

%remove first and last 3 seconds of data
phif(1:3*Fs) = [];
phif(end-3*Fs:end) = [];
alphaf(1:3*Fs) = [];
alphaf(end-3*Fs:end) = [];

alpha0 = alphaf - mean(alphaf);

phi_lp = smooth(phif,10);
alpha_lp = smooth(alpha0,10); 

figure, hold on
subplot(121), plot(phi_lp), hold on
subplot(122), plot(alpha_lp), hold on

dPhi = diff(phi_lp);
dAlpha = diff(alpha_lp);
subplot(121), plot(dPhi,'.r')
subplot(122), plot(dAlpha,'.-r')

%detect zero crossing for dPhi
signdPhi = [];
for k=1:length(dPhi)-1
    signdPhi(k) = dPhi(k)*dPhi(k+1);
end
   
ind0 = find(signdPhi < 0);
ind0opt = [];
corr=0;
for k=1:length(ind0)
    [~,ik] = min(dPhi(ind0(k)-1:ind0(k)+1));    %search in the neighbor of sign inversion
    ind0opt(k-corr)=ind0(k)+ik-2;
    if k>1
        if ind0opt(k-corr)-ind0opt(k-1-corr)<10
            corr=corr+2;
        end
    end
end
ind0 = ind0opt; %indices of min and max values of phi

%show min and max 
subplot(121),plot(ind0,dPhi(ind0),'mx','MarkerSize',16)

%plot alpha and phi when phi=phimax 
figure, hold on, grid on, xlim([-15 15]), ylim([-8 20.5])
plot(alpha0(ind0),phif(ind0),'or'), xlabel('alpha'), ylabel('phi')

%% extract datapoints between 2 consecutive minima 
d2Phi = diff(dPhi);
d2Phi0 = d2Phi(ind0);
im = find(d2Phi0 > 0);
ind0m = ind0(im);   %indices of local minimas in phif

x = {}; t= {}; x100 = [];
t100 = linspace(0,1,101);
figure, hold on
for k =1:length(ind0m)-1
    
    x{k} = phif(ind0m(k):ind0m(k+1));
    x{k} = x{k}-x{k}(1);    %remove amplitude offset
    t{k} = linspace(0,1,length(x{k}));
    %     plot(t{k},x{k})
    %resample to uniform # of points and normalize amplitude
    xnew = interp1(t{k},x{k},t100);
    xnew = xnew./max(abs(xnew));         %normalize amplitude
    x100 = [x100;xnew];
    plot(t100,x100(k,:))
    
    %compute RMSE between 2 consecutive trials
    if k > 1
        RMSE(k-1) = sum( (x100(k-1,:)-x100(k,:)).^2 ) / length(t100);
        
    end

    %compute max Amplitude and Time where max occurs
    [Amax(k),Tmax1(k)] = max(x100(k,:))
    Tmax(k) = t100(Tmax1(k));

    
    % input('')
end

ylim([-1.2 1.2])



%min and max phi values separated
% PhimM = phi_lp(ind0);
% Phim = PhimM(PhimM < 0); 
% PhiM = PhimM(PhimM > 0);


%detect zero crossing for Alpha
% signAlpha = [];
% for k=1:length(dAlpha)-1
%     signAlpha(k) = dAlpha(k)*dAlpha(k+1);
% end
%    
% ind0 = find(signAlpha < 0);
% ind0opt = [];
% for k=1:length(ind0)
%     [~,ik] = min(dAlpha(ind0(k)-1:ind0(k)+1));
%     ind0opt(k)=ind0(k)+ik-2;
%     
% end
% %show min and max 
% % subplot(122),plot(ind0,dAlpha(ind0opt),'mx','MarkerSize',16)
% 
% %min and max alpha 
% AlphamM = alpha0(ind0);
% Alpham = AlphamM(AlphamM < 0); 
% AlphaM = AlphamM(AlphamM > 0);
%% ENERGY EXPENDITURE 
        clear E
        clear K
        clear V

        Epoch = 0.5; %epoch length [s]
        T = Fs*Epoch;

        grav=mean(accwalk);
        grav=grav/(sqrt(sum(grav.^2)));
        
        for k = 1:floor((size(accwalk,1)-1)/T)
            int=[];
            grav=mean(accwalk((k-1)*T+1:k*T,:));
            grav=grav/(sqrt(sum(grav.^2)));
            for n=T*(k-1)+1:k*T
                int(n-T*(k-1))=sum(trapz(abs(accwalk(n,:)-dot(accwalk(n,:),grav)*grav)));
            end
            E(k) = sum(int);
        end


        % Energy per step
        E
        Etot = sum(E)      %total energy over walk session
        Nsteps = round(Stepf*t(end))
        Estep = Etot/Nsteps
        


%% simple spectral analysis of the filtered signal
L = length(phif);    %signal length
Y = fft(phif,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
Pyy = Y.*conj(Y)/L; %power spectrum
f = Fs/2*linspace(0,1,L/2+1);   %frequency axis
figure; 
subplot(121), hold all
% plot(f,2*abs(Y(1:L/2+1)))     %Fourier magnitude
plot(f,Pyy(1:L/2+1),'LineWidth',2)
% plot(f,smooth(Pyy(1:L/2+1),200),'LineWidth',2)    %smooth values
title('Power spectral density - Phi')
xlabel('Frequency (Hz)');  xlim([0 10])

Y = fft(alphaf,L);
Pyy2 = Y.*conj(Y)/L;
f2 = Fs/2*linspace(0,1,L/2+1);
subplot(122)
plot(f2,Pyy2(1:L/2+1),'LineWidth',2)
title('Power spectral density - Alpha')
xlabel('Frequency (Hz)');  xlim([0 10])

%% Scatter plot of frontal and lateral tilt
%(TO BE CONTINUED)
% figure('name','Filtered angles'); subplot(121)
% plot(phif); hold on
% title('Phi filtered');
% subplot(122)
% plot(alphaf);
% title('Alpha filtered')
% 
% dPhi = diff(phi);
% d2Phi= diff(diff(phi))
% Phimm = d2Phi(ind_dPhi)

% subplot(121), plot(dPhi,'r.-')
% ind_dPhi = find(diff(phi)==0)
%% ACTIGRAPH ANKLE DATA 
load ./MatlabData/Rewalk_p02_Ankle' (2014-05-06)RAW.mat'

acc = cell2mat(accraw(:,2:end));
Fs = 30;   %Sampling freq

%remove zeros
% [r,c] = find(acc(:,1) == 0 & acc(:,2) == 0 & acc(:,3) == 0);
% acc(r,:) = [];

%plot raw data
% t = 0:1/Fs:(length(acc)/Fs-1/Fs);
% figure
% set(0,'defaultlinelinewidth',2)
% plot(t,acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
% xlabel('Time [s]'), ylabel('acc [g]')

%% Extract Walk accelerations
% close all
acc = accraw;

Starttime = '13:41:00.000'; 
Endtime = '13:46:00.000';

i = 0; t = 0;
while i == 0
    t = t+1;
    i = strcmp(acc{t}(12:end),Starttime);
end
indstart = t;

i = 0; t = 0;
while i == 0
    t = t+1;
    i = strcmp(acc{t}(12:end),Endtime);
end
indend = t; 

acc = acc(indstart:indend,:);
acc = cell2mat(acc(:,2:end));

%plot with time in secs
% figure
% t = 0:1/Fs:(length(acc)/Fs-1/Fs);
% plot(t,acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
% xlabel('Time [s]'), ylabel('acc [g]')

%plot with samples shown
figure('name','Ankle acceleration Actigraph')
plot(acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
ylabel('acc [g]')

%% select a specific walk sequence
t1 = 7760; t2 = 9300;
accwalk = acc(t1:t2,:);
figure('name','Ankle acceleration Actigraph')
plot(accwalk,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
ylabel('acc [g]')

%% PHONE DATA ANALYSIS (Purple Robot)
close all;

FsPh = 48;  %phone sample rate (from Sohrab script)

% load ./MatlabData/Kate_7_14_t.mat
accPh = Probedata{1};
grvPh = Probedata{2};
rotPh = Probedata{3};

%use same convention for axis as actigraph
accPh = [accPh(:,3) accPh(:,[2 4])];
grvPh = [grvPh(:,3) grvPh(:,[2 4])];
rotPh = [rotPh(:,3) rotPh(:,[2 4])];

figure('name','Phone gravitometer')
plot(grvPh./9.8), legend('x (g)','y (frontal)','z (perp)')
% 
% figure('name','Phone accelerometer')
% plot(accPh./9.8), legend('x (g)','y (frontal)','z (perp)')

figure('name','Phone Data'), hold on
subplot(311)
plot(accPh./9.8), legend('x (g)','y (frontal)','z (perp)')
subplot(312)
plot(grvPh./9.8), legend('x (g)','y (frontal)','z (perp)')
subplot(313)
plot(rotPh), legend('x (g)','y (frontal)','z (perp)')


%sanity check - norm of the gravity (OK this is accurate) 
grvnorm = sqrt(grvPh(:,1).^2+grvPh(:,2).^2+grvPh(:,3).^2);  
figure, plot(grvnorm);

%% manually isolate one part of the walk data (need to Sync with Actigraph data)
num=1;
t1 = PhoneTestIntervals{num,1}; t2 = PhoneTestIntervals{num,2};
grvPhW = grvPh(t1:t2,:);
figure
plot(grvPhW./9.8)

%Phi and Alpha from phone grv probe
phiPh = (180/pi)*atan2(grvPhW(:,2),grvPhW(:,1));    %ATAN2 does not suffer from sensitivity issues
%roll (lateral) inclination
alphaPh = (180/pi)*atan2(-grvPhW(:,3),grvPhW(:,1));

figure('name','Tilt angles - Phone'); subplot(121)
plot(phiPh), title('frontal angle')
subplot(122)
plot(alphaPh), title('lateral angle')

x=0:1/FsPh:(length(phiPh)/FsPh-1/FsPh);
xq=0:1/30:(length(phiPh)/FsPh-1/FsPh);
grv30=interp1(x, grvPhW, xq);
figure('name','Interpolated to 30 Hz');
plot(grv30./9.8)

phiPh30=(180/pi)*atan2(grv30(:,2),grv30(:,1));  
figure
plot(phiPh30)
%% simple spectral analysis of the signal
L = length(phiPh);    %signal length
Y = fft(phiPh,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
Pyy = Y.*conj(Y)/L; %power spectrum
f = FsPh/2*linspace(0,1,floor(L/2+1));   %frequency axis
figure; 
subplot(121), hold all
% plot(f,2*abs(Y(1:L/2+1)))     %Fourier magnitude
plot(f,Pyy(1:floor(L/2+1)),'LineWidth',2)
% plot(f,smooth(Pyy(1:L/2+1),200),'LineWidth',2)    %smooth values
title('Power spectral density - Phi Phone')
xlabel('Frequency (Hz)');  xlim([0 10])


Y = fft(alphaPh,L);
Pyy = Y.*conj(Y)/L;
f = FsPh/2*linspace(0,1,floor(L/2+1));
subplot(122)
plot(f,Pyy(1:floor(L/2+1)),'LineWidth',2)
title('Power spectral density - Alpha Phone')
xlabel('Frequency (Hz)');  xlim([0 10])


%% low pass filter phi and alpha
t = 0:1/FsPh:(length(phiPh)/FsPh-1/FsPh);

ft = 1.0; %cut-off freq
[B,A] = butter(1, 2*ft/FsPh);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
phiPhf = filtfilt(B,A,phiPh);   %the filtered version of the signal
figure('name','Filtered angles - Phone'); subplot(121)
plot(t,phiPhf); hold on
title('Phi filtered'); xlim([0 t(end)+1]); ylim([-15 15]);

%alpha
ft = 0.5;
[B,A] = butter(1, 2*ft/FsPh);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
alphaPhf = filtfilt(B,A,alphaPh);   %the filtered version of the signal
subplot(122)
plot(t,alphaPhf); 
title('Alpha filtered'); xlim([0 t(end)+1]); ylim([-4 12]);


