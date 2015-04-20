%Extract Waist-Ankle times

accraw = accrawAnkle;

acc = cell2mat(accraw(:,2:end));
Fs = 100;   %Sampling freq

%plot raw data
t = 0:1/Fs:(length(acc)/Fs-1/Fs);
figure
set(0,'defaultlinelinewidth',2)
plot(t,acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Time [s]'), ylabel('acc [g]')

%% Extract Walk times
% close all
acc = accraw;

Starttime = '17:02:41.000'; 
Endtime = '17:03:25.000';

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

accAnkle = cell2mat(accrawAnkle(indstart:indend,2:end));
accWaist = cell2mat(accrawWaist(indstart:indend,2:end));

%Resample at 30 Hz
% t=0:1/Fs:(length(acc)/Fs-1/Fs);
% tnew=0:1/30:(length(acc)/Fs-1/Fs);
% acc=interp1(t, acc, tnew); Fs = 30;

%plot with time in secs (raw acc)
t = 0:1/Fs:(length(accAnkle)/Fs-1/Fs);
figure('name','Ankle-Waist acc'), subplot(211)
plot(t,accWaist,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Time [s]'), ylabel('acc [g]'), title('Waist')
subplot(212)
plot(t,accAnkle,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Time [s]'), ylabel('acc [g]'), title('Ankle')

%plot with samples shown
% figure
% plot(acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
% xlabel('Samples #'), ylabel('acc [g]')


%% extract precise walk sequence
% t1 = 15.70; t2 = 40; %[s]
% x1 = t1*Fs; x2 = t2*Fs; %[samples]

accwalkA = accAnkle(x1:x2,:);
accwalkW = accWaist(x1:x2,:);
figure, subplot(211)
plot(accwalkA,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
ylabel('acc [g]')
subplot(212)
plot(accwalkW,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
ylabel('acc [g]')

accWaist = accwalkW;
accAnkle = accwalkA;


%% Ankle Sagittal Angle
acc = accAnkle;
t = 0:1/Fs:(length(acc)/Fs-1/Fs);

phiankle = atan2(acc(:,2),acc(:,1));    %ATAN2 does not suffer from sensitivity issues
%phi < 0 flex; phi > 0 extension convention
phiankle = -phiankle;
%zero mean
phiankle0 = phiankle - mean(phiankle);
phiankle = phiankle0;

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

%compare Butterworth with moving avg filter
anklefig = figure('name','Sagittal Orientation and angular speed - Ankle'); 
subplot(211), plot(t,phiankle,'m','LineWidth',2); title('Sagittal angle')
xlabel('Time [s]'), ylabel('phi [rad]')
subplot(212), hold on, plot(t(1:end-1),dphiankle,'m','LineWidth',2); title('Angular speed')

% plot(t(1:end-1),dphiankleB,'m','LineWidth',2);
% xlabel('Time [s]'), ylabel('dphi [rad/s]')
% plot(t(1:end-1),dphiankle_MA,'r','LineWidth',2); legend('Orig','Butter','Mov Avg')



%% Extract Ankle peaks (max Extension)
%Zero-mean

subplot(211), plot(t,phiankle,'m','LineWidth',2); title('Sagittal angle')

%detect zero crossing for dPhi
signPhi = [];
for k=1:length(dphiankle)-1
    signPhi(k) = dphiankle(k)*dphiankle(k+1);
end
   
ind0 = find(signPhi < 0);
ind0opt = [];
for k=1:length(ind0)
    [~,ik] = min(dphiankle(ind0(k)-1:ind0(k)+1));
    ind0opt(k)=ind0(k)+ik-2;
    
end
ind0 = ind0opt; %indices of min and max values of phi

%show min and max 
% subplot(211),hold on, plot(ind0/Fs,dphiankle(ind0),'kx','MarkerSize',12)

%extract max (values > 0)
iM = find(phiankle(ind0) > 0)
ind0M = ind0(iM);   %max
% figure(anklefig), subplot(211); hold on, plot(ind0M/Fs,dphiankle(ind0M),'rx','MarkerSize',12)

%Optimization alg: find local maxima
Wsize = 30;
for k = 1:length(ind0M)
    [~,ik] = max(phiankle(ind0M(k)-Wsize:ind0M(k)+Wsize));
    ind0M(k) = ind0M(k)+ik-(Wsize+1);
end

figure(anklefig), hold on
subplot(211); hold on, plot(ind0M/Fs,dphiankle(ind0M),'rx','MarkerSize',6)

tM_Ankle = ind0M/Fs;    %times of max peaks for ankle


%% Lateral and frontal tilt from waist accelerometer data 
acc = accWaist;
%phi - tilt (frontal) inclination 
phiwaist = (180/pi)*atan2(acc(:,    2),acc(:,1));    %ATAN2 does not suffer from sensitivity issues
%alpha - roll (lateral) inclination
alphawaist = (180/pi)*atan2(acc(:,3),acc(:,1));

t = 0:1/Fs:(length(acc)/Fs-1/Fs);

ft = 0.5; %cut-off freq
[B,A] = butter(1, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
phiwaistf = filtfilt(B,A,phiwaist);   %the filtered version of the signal
figure('name','Waist Phi Filtered angles'); subplot(121)
phiwaistf = phiwaistf*pi/180; %[rad]
plot(t,phiwaistf); hold on
title('Phi filtered'); xlim([0 t(end)+1]);  

figure(anklefig), subplot(211), hold on
plot(t,phiwaistf)

phiwaist = phiwaistf;

%% Extract waist peaks 

phiwaist0 = phiwaist - mean(phiwaist);
phiwaist = phiwaist0;           %zero-mean waist frontal angle
dphiwaist = diff(phiwaist);     %waist velocity

%filter velocity
ft = 6; %cut-off freq
[Bv,Av] = butter(2, 2*ft/Fs);   %2nd order, cutoff frequency 6Hz (Normalized by 2*pi*Sf) [rad/s]
dphiwaistf = filtfilt(B,A,dphiwaist);   %the filtered version of the signal
dphiwaist = dphiwaistf;

waistfig = figure; subplot(211), hold on, plot(t,phiwaist,'b','LineWidth',2); title('Waist angle')
subplot(212), hold on, plot(t(1:end-1), dphiwaistf,'b','LineWidth',2); title('Filtered Waist velocity')

%detect zero crossing for dPhi
signPhi = [];
for k=1:length(dphiwaist)-1
    signPhi(k) = dphiwaist(k)*dphiwaist(k+1);
end
   
ind0 = find(signPhi < 0);
ind0opt = [];
for k=1:length(ind0)
    [~,ik] = min(dphiwaist(ind0(k)-1:ind0(k)+1));
    ind0opt(k)=ind0(k)+ik-2;
end
ind0 = ind0opt; %indices of min and max values of phi

%extract max (d2phi < 0)
d2phiwaist = diff(dphiwaist);
iM = find(d2phiwaist(ind0) < 0);
ind0M = ind0(iM);

%extract max (values > 0)
% iM = find(phiwaist(ind0) > 0)
% ind0M = ind0(iM);   %max

%show min and max
% figure(anklefig)
% subplot(211),hold on, plot(ind0/Fs,dphiwaist(ind0),'b+','MarkerSize',6)

%extract max (values > 0)
% iM = find(phiwaist(ind0) > 0);
% ind0M = ind0(iM);   %max

% figure(anklefig), hold on
% subplot(211); hold on, plot(ind0M/Fs,dphiwaist(ind0M),'r+','MarkerSize',6)

%optimize peaks with iterative algorithm
for k = 1:length(ind0M)
    while(phiwaist(ind0M(k)) < phiwaist(ind0M(k)+1))
       ind0M(k) = ind0M(k)+1;
    end
       
    while(phiwaist(ind0M(k)) < phiwaist(ind0M(k)-1))
       ind0M(k) = ind0M(k)-1;
    end
end

% figure(anklefig), hold on
% subplot(211); hold on, plot(ind0M/Fs,dphiwaist(ind0M),'g*','MarkerSize',6)


%Optimization 2: find local maximum
Wsize = 30;
for k = 1:length(ind0M)
    [~,ik] = max(phiwaist(ind0M(k)-Wsize:ind0M(k)+Wsize))
    ind0M(k) = ind0M(k)+ik-(Wsize+1);
end

figure(anklefig), hold on
subplot(211); hold on, plot(ind0M/Fs,dphiwaist(ind0M),'g*','MarkerSize',6)

tM_Waist = ind0M/Fs;    %times of max peaks for waist

%% Extract time differences
twa_swing = []; twa_stance = [];
  

for i = 1:length(tM_Ankle)
    d = tM_Waist - tM_Ankle(i); %time differences between peaks
    dneg = d((d<0));   dpos = d((d>0)); 
    if ~isempty(dneg) && ~isempty(dpos)
        twa_swing(i) = max(dneg);     %max trunk tilt to max ankle extension (swing ipsilateral)
        twa_stance(i) = min(dpos);    %max ankle extension to max trunk tilt (stance ipsilateral)
    end 
end 
    




