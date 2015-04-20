%Generate data for training/testing the classifier
%Provide cell array of raw accelerations (with timestamps), Testtimes
%matrix with start and end times of the test and Sampling Frequency Fs

function X = ClassDataExtr(acc,TestTimes,Fs)

%Extract classifier Data

%Load Data: Test Times file + csv 

% Fs = 30;   %Sampling freq

%Extract Walking Sessions
% close all
% acc = accraw;

% Starttime = '10:55:00.000'; 
% Endtime = '11:02:00.000';

Starttime = TestTimes{1,1};
Endtime = TestTimes{1,2};
Starttime = strcat(Starttime,':00.000');
Endtime = strcat(Endtime,':00.000');

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

%plot with samples shown
figure
plot(acc,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
xlabel('Samples #'), ylabel('acc [g]')

%% Labeling (0 = No walk, 1 = Walk)

%Indexed Labels
% L = 1:length(acc);  %sample length
% a1 = [];    %walk acc section
Label = zeros(length(acc),1);
Twalk = cell2mat(TestTimes(:,3:4));
for k = 1:size(TestTimes,1)
    Label(Twalk(k,1):Twalk(k,2)) = 1;
%     acc1 = [acc1;acc(Twalk(k,1):Twalk(k,2),:)];
end

%extract windows and label datapoints
Wsize = 5*Fs;   %[secs*SamplingRate = samples]
N = round(length(acc)/Wsize);   %# of data points
X = cell(N,3);                  %data points
clipthres = 0.9;                %ratio of data points in one clip to be considered walking

%Build Dataset
for w = 1:N
    
    X{w} = acc( (w-1)*Wsize+1:w*Wsize,: );    %acc data
    
    l1 = Label(Label((w-1)*Wsize+1:w*Wsize) == 1);
    l0 = Label(Label((w-1)*Wsize+1:w*Wsize) == 0);
    if length(l1) > length(l0) && length(l1) > clipthres*Wsize 
        X{w,2} = 1;
    elseif length(l0) > length(l1) && length(l0) > clipthres*Wsize
        X{w,2} = 0;
    else
        X{w,2} = 2; %discard
    end
end
%remove data to discard
i2 = find(cell2mat(X(:,2)) == 2);
discarded = length(i2)/N;
disp([num2str(discarded*100) ' % of data points discarded'])
X(i2,:) = [];

%visualize data
X1 = X((cell2mat(X(:,2))) == 1);
X0 = X((cell2mat(X(:,2))) == 0);
X1 = cell2mat(X1);
X0 = cell2mat(X0);
figure, hold on, subplot(211),plot(X1)
subplot(212), plot(X0)



%% Compute lateral and frontal tilt from accelerometer data
i1 = find(cell2mat(X(:,2)) == 0);
acc1 = X{i1(4),1};
%tilt (frontal) inclination
phi = (180/pi)*atan2(acc1(:,2),acc1(:,1));    %ATAN2 does not suffer from sensitivity issues
%roll (lateral) inclination
alpha = (180/pi)*atan2(acc1(:,3),acc1(:,1));
%% simple spectral analysis of the signal
L = length(phi);    %signal length
Y = fft(phi,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
Pyy = Y.*conj(Y)/L; %power spectrum
Pyy = Pyy./sum(Pyy);    %normalized Power spectrum
f = Fs/2*linspace(0,1,L/2+1);   %frequency axis

Y2 = fft(alpha,L);
Pyy2 = Y2.*conj(Y2)/L;    %power spectrum
Pyy2 = Pyy2./sum(Pyy2);   %normalized Power spectrum
f2 = Fs/2*linspace(0,1,L/2+1);

figure; 
subplot(121), hold all
% plot(f,2*abs(Y(1:L/2+1)))     %Fourier magnitude
plot(f(2:end),Pyy(2:L/2+1),'LineWidth',2)
% plot(f,smooth(Pyy(1:L/2+1),200),'LineWidth',2)    %smooth values
title('Power spectral density - Phi')
xlabel('Frequency (Hz)');  xlim([0 10])
subplot(122)
plot(f2(2:end),Pyy2(2:L/2+1),'LineWidth',2)
title('Power spectral density - Alpha')
xlabel('Frequency (Hz)');  xlim([0 10])


%% Extract Features

for w = 1:length(X)
    X{w,3} = featureextr(X{w,1},Fs);
end

disp('Features Extracted')
% X{:,3} = f; %last col of X contains the features





