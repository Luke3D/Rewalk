%% ACTIGRAPH DATA ANALYSIS

% load ./MatlabData/Rewalk_p02_Waist' (2014-05-06)RAW.mat'
% load ./MatlabData/Rewalk_Test_Waist' (2014-06-16)RAW.mat'

acc = cell2mat(accraw(:,2:end));
Fs = 100;   %Sampling freq

%remove zeros
% [r,c] = find(acc(:,1) == 0 & acc(:,2) == 0 & acc(:,3) == 0);
% acc(r,:) = [];

%plot raw data
t = 0:1/Fs:(length(acc)/Fs-1/Fs);
% t = t/60; %[min]
figure
set(0,'defaultlinelinewidth',2)
plot(t,acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Time [s]'), ylabel('acc [g]')

%% Extract Walk accelerations
% close all
acc = accraw;

Starttime = '12:43:05.000'; 
Endtime = '12:44:00.000';

%end of date and beginning of time character
len = find(acc{1} == ' ');

i = 0; t = 0;
while i == 0
    t = t+1;
    i = strcmp(acc{t}(len+1:end),Starttime);
end
indstart = t;

i = 0; t = 0;
while i == 0
    t = t+1;
    i = strcmp(acc{t}(len+1:end),Endtime);
end
indend = t; 

acc = acc(indstart:indend,:);
acc = cell2mat(acc(:,2:end));

%Resample at 30 Hz
% t=0:1/Fs:(length(acc)/Fs-1/Fs);
% tnew=0:1/30:(length(acc)/Fs-1/Fs);
% acc=interp1(t, acc, tnew); Fs = 30;

%plot with time in secs (raw acc)
figure('name','Raw acc')
t = 0:1/Fs:(length(acc)/Fs-1/Fs);
plot(t,acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Time [s]'), ylabel('acc [g]')

%plot with samples shown
figure
plot(acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Samples #'), ylabel('acc [g]')


%band pass filter 
% ft = [0.5 1.2]; %cut-off freq [Hz]
% [B1,A1] = butter(4, 2*ft/Fs,'bandpass');   %4th order,low pass, cutoff freq Normalized by 2*pi*Sf [rad/s]
% fvtool(B1,A1)
% accf = filtfilt(B1,A1,acc);   %the filtered version of the signal

%plot with time in secs (filtered)
% figure('name','Filtered')
% t = 0:1/Fs:(length(acc)/Fs-1/Fs);
% plot(t,accf,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
% xlabel('Time [s]'), ylabel('acc [g]')


%% Ankle Sagittal Angle

phiankle = atan2(acc(:,2),acc(:,1));    %ATAN2 does not suffer from sensitivity issues

%low pass filter
ft = 1; %cut-off freq
[B,A] = butter(1, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
phiankle = filtfilt(B,A,phiankle);   %the filtered version of the signal
%Ankle velocity 
dphiankle = diff(phiankle);

%filter velocity
ft = 6; %cut-off freq
[Bv,Av] = butter(2, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
dphiankle = filtfilt(B,A,dphiankle);   %the filtered version of the signal

%phi < 0 flex; phi > 0 extension

figure('name','Sagittal Orientation and angular speed - Ankle') 
subplot(211), plot(t,-phiankle,'m','LineWidth',2); 
xlabel('Time [s]'), ylabel('phi [rad]')
subplot(212), plot(t(1:end-1),-dphiankle,'m','LineWidth',2); title('Angular speed')
xlabel('Time [s]'), ylabel('dphi [rad/s]')



%% select a specific walk sequence
t1 = 7760; t2 = 9300;
t1 = 7760; t2 = 9300;
% t1 = 802; t2 = length(acc);
accwalk = acc(t1:t2,:);
figure
plot(accwalk,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
ylabel('acc [g]')

acc = accwalk;
%% Compute lateral and frontal tilt from accelerometer data 

%tilt (frontal) inclination 
phi = (180/pi)*atan2(acc(:,2),acc(:,1));    %ATAN2 does not suffer from sensitivity issues
%roll (lateral) inclination
alpha = (180/pi)*atan2(acc(:,3),acc(:,1));

% figure; subplot(121)
% plot(t,phi), title('frontal angle')
% subplot(122)
% plot(t,alpha), title('lateral angle')

%% simple spectral analysis of the signal
L = length(phi);    %signal length
Y = fft(phi,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
Pyy = Y.*conj(Y)/L; %power spectrum
f = Fs/2*linspace(0,1,L/2+1);   %frequency axis
figure; 
subplot(121), hold all
% plot(f,2*abs(Y(1:L/2+1)))     %Fourier magnitude
plot(f,Pyy(1:L/2+1),'LineWidth',2)
% plot(f,smooth(Pyy(1:L/2+1),200),'LineWidth',2)    %smooth values
title('Power spectral density - Phi')
xlabel('Frequency (Hz)');  xlim([0 10])

[~,is] = max(Pyy); %approx count of steps per sec 
Nsteps = f(is);

Y = fft(alpha,L);
Pyy = Y.*conj(Y)/L;
f = Fs/2*linspace(0,1,L/2+1);
subplot(122)
plot(f,Pyy(1:L/2+1),'LineWidth',2)
title('Power spectral density - Alpha')
xlabel('Frequency (Hz)');  xlim([0 10])


%% Low pass filter 
%phi
t = 0:1/Fs:(length(acc)/Fs-1/Fs);

ft = 0.5; %cut-off freq
[B,A] = butter(1, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
phif = filtfilt(B,A,phi);   %the filtered version of the signal
figure('name','Filtered angles - Actigraph'); subplot(121)
phif = phif*pi/180; %[rad]
plot(t,phif); hold on
title('Phi filtered'); xlim([0 t(end)+1]);  

%alpha
ft = 0.25;
[B,A] = butter(1, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
alphaf = filtfilt(B,A,alpha);   %the filtered version of the signal
alphaf = alphaf*pi/180
subplot(122)
plot(t,alphaf); xlim([0 t(end)+1])
title('Alpha filtered')


%% ENERGY EXPENDITURE 

Epoch = 0.5; %epoch length [s]
T = Fs*Epoch;

for k = 1:floor(size(acc,1)/T)
    E(k) = sum(trapz(abs(acc(T*(k-1)+1:k*T,:))));
end


%Energy per step
Etot = sum(E);      %total energy over walk session
% Emean = mean(E);    %mean over walk session
Nsteps = round(Nsteps*t(end));
Estep = Etot/Nsteps;

figure
% plot(E)
bar(Estep)

clear E

%% Plot ellipse whose axes are variance of tilt angles
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
signPhi = [];
for k=1:length(dPhi)-1
    signPhi(k) = dPhi(k)*dPhi(k+1);
end
   
ind0 = find(signPhi < 0);
ind0opt = [];
for k=1:length(ind0)
    [~,ik] = min(dPhi(ind0(k)-1:ind0(k)+1));
    ind0opt(k)=ind0(k)+ik-2;
    
end
ind0 = ind0opt; %indices of min and max values of phi

%show min and max 
subplot(121),plot(ind0,dPhi(ind0),'mx','MarkerSize',16)

%plot alpha and phi when phi=phimax 
figure, hold on, grid on, xlim([-15 15]), ylim([-8 20.5])
plot(alpha0(ind0),phif(ind0),'or'), xlabel('alpha'), ylabel('phi')

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

%% Phase plot of Phi and Alpha
dPhi = diff(phif);
dAlpha = diff(alpha0);

figure
subplot(211), plot(phif(1:end-1),dPhi), xlabel('Phi'), ylabel('dPhi')
subplot(212), plot(alpha0(1:end-1),dAlpha), xlabel('Alpha'), ylabel('dAlpha')



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
FsPh = 48;  %phone sample rate (from Sohrab script)

load ./MatlabData/Probedata_p02_(2014-05-06).mat
accPh = Probedata{1};
grvPh = Probedata{2};
rotPh = Probedata{3};

%use same convention for axis as actigraph
accPh = [accPh(:,3) accPh(:,[2 4])];
grvPh = [grvPh(:,3) grvPh(:,[2 4])];
rotPh = [rotPh(:,3) rotPh(:,[2 4])];

figure('name','Phone acceleration')
plot(accPh./9.8), legend('x (g)','y (frontal)','z (perp)')

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
t1 = 27130; t2 = 28740;
grvPhW = grvPh(t1:t2,:);
figure
plot(grvPhW)

%Phi and Alpha from phone grv probe
phiPh = (180/pi)*atan2(grvPhW(:,2),grvPhW(:,1));    %ATAN2 does not suffer from sensitivity issues
%roll (lateral) inclination
alphaPh = (180/pi)*atan2(grvPhW(:,3),grvPhW(:,1));

figure('name','Tilt angles - Phone'); subplot(121)
plot(phiPh), title('frontal angle')
subplot(122)
plot(alphaPh), title('lateral angle')

%% simple spectral analysis of the signal
L = length(phiPh);    %signal length
Y = fft(phiPh,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
Pyy = Y.*conj(Y)/L; %power spectrum
f = FsPh/2*linspace(0,1,L/2+1);   %frequency axis
figure; 
subplot(121), hold all
% plot(f,2*abs(Y(1:L/2+1)))     %Fourier magnitude
plot(f,Pyy(1:L/2+1),'LineWidth',2)
% plot(f,smooth(Pyy(1:L/2+1),200),'LineWidth',2)    %smooth values
title('Power spectral density - Phi Phone')
xlabel('Frequency (Hz)');  xlim([0 10])

Y = fft(alpha,L);
Pyy = Y.*conj(Y)/L;
f = FsPh/2*linspace(0,1,L/2+1);
subplot(122)
plot(f,Pyy(1:L/2+1),'LineWidth',2)
title('Power spectral density - Alpha Phone')
xlabel('Frequency (Hz)');  xlim([0 10])


%% low pass filter phi and alpha
ft = 0.5; %cut-off freq
[B,A] = butter(1, 2*ft/FsPh);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
phiPhf = filtfilt(B,A,phiPh);   %the filtered version of the signal
figure('name','Filtered angles - Phone'); subplot(121)
plot(phiPhf); hold on
title('Phi filtered');

%alpha
ft = 0.25;
[B,A] = butter(1, 2*ft/FsPh);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
alphaf = filtfilt(B,A,alphaPh);   %the filtered version of the signal
subplot(122)
plot(alphaf); 
title('Alpha filtered')


