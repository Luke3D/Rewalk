% Extracts relevant features from the acceleration data for the patient and
% dates listed below.  The results for each training session are saved as
% files labeled "Patient_(Date)_Metrics.mat"

% Columns in the output matrix Metrics:
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Length of walking session (in actigraph samples: 30 Hz)
% 5. Ratio of time spent walking to total test time
% 6. Number of steps in walking section
% 7. Calculated index for walking section

% Final row has average results for training session
% Be sure to indicate the appropriate sampliong frequency below


% Dates={'05-27','06-02','06-03','06-09','06-11','06-18','06-19','06-20','06-23','06-24','06-27','07-03','07-08','07-09','07-10', '07-15', '07-16', '07-17'};
% Dates={'03-25', '03-26', '03-31', '04-01', '04-02', '04-07', '04-08', '04-09', '04-15', '04-16', '05-01', '05-05', '05-06', '05-07', '05-21', '06-09'};
Patient='RC02';
Dates={'10-23'};
for indDates=1:length(Dates)
%     load(['./MatlabData/Patients/' Patient '/' Patient '_Waist(2014-' Dates{indDates} ')RAW.mat'])
%     load(['./TestTimes/' Patient '/' Patient '_2014-' Dates{indDates} '_Times.mat'])
    
    load ../../MatlabData/Healthy' Controls'/RC02_23_10_2014.mat
    
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
    %% One Date
    [count,~]=size(TestTimes);
    Metrics=zeros(count+1, 7);
    for num=1:count
        %% Extract Walk accelerations
        close all
        acc = accraw;

        % num=2;% number of section of trial you wish to evaluate (See TestTimes cell array from the appropriate Cutoff Time file)

        Starttime = [TestTimes{num} ':00.000']; 
        Endtime = [TestTimes{num,2} ':00.000'];

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
        
        %RESAMPLING TO 30 HZ
%         x=0:1/Fs:(length(acc)/Fs-1/Fs);
%         xq=0:1/30:(length(acc)/Fs-1/Fs);
%         acc=interp1(x, acc, xq);
        
        acctrim=acc;
        %% select a specific walk sequence
        close all
        t1 = TestTimes{num,3}; t2 = TestTimes{num,4};
        accwalk = acctrim(t1:t2,:);
        acc = accwalk;
        %% Compute lateral and frontal tilt from accelerometer data 

        %tilt (frontal) inclination 
        phi = (180/pi)*atan2(acc(:,2),acc(:,1));    %ATAN2 does not suffer from sensitivity issues
        %roll (lateral) inclination
        alpha = (180/pi)*atan2(acc(:,3),acc(:,1));
        %% simple spectral analysis of the signal
        L = length(phi);    %signal length
        Y = fft(phi,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
        Pyy = Y.*conj(Y)/L; %power spectrum
        f = Fs/2*linspace(0,1,L/2+1);   %frequency axis

        fstepsMax = 1.5;       %Upper bound on Step Freq (> gives error)
        
        [~,is] = max(Pyy(3:fstepsMax*L/Fs)); %approx count of steps per sec 
        Nsteps = f(is+2);
  
        Y = fft(alpha,L);
        Pyy2 = Y.*conj(Y)/L;
        f2 = Fs/2*linspace(0,1,L/2+1);


        %% Low pass filter 
        %phi
        t = 0:1/Fs:(length(acc)/Fs-1/Fs);

        ft = 1.0; %cut-off freq
        [B,A] = butter(1, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
        phif = filtfilt(B,A,phi);   %the filtered version of the signal

        %alpha
        ft = 0.5;
        [B,A] = butter(1, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
        alphaf = filtfilt(B,A,alpha);   %the filtered version of the signal

        pm=mean(phif);
        ps=std(phif);
        am=mean(alphaf);
        as=std(alphaf);

        Metrics(num,1:2)=[Nsteps, ps];
        Metrics(num, 4:5)=[length(acc), length(acc)/length(acctrim)];

        %% ENERGY EXPENDITURE 
        Epoch = 0.5; %epoch length [s]
        T = Fs*Epoch;

        grav=mean(acc);
        grav=grav/(sqrt(sum(grav.^2)));

        for k = 1:floor((size(acc,1)-1)/T)
        %     E(k) = sum(trapz(acc(T*(k-1)+1:k*T,:))-15*grav);
            int=[];
            grav=mean(acc((k-1)*T+1:k*T,:));
            grav=grav/(sqrt(sum(grav.^2)));
            for n=T*(k-1)+1:k*T
                int(n-T*(k-1))=sum(abs(acc(n,:)-dot(acc(n,:),grav)*grav));
            end
            E(k) = sum(trapz(int));
        %     K(k) = 900*.5*L^2*sum(trapz((dPhi(T*(k-1)+1:k*T,:)).^2));
        %     V(k) = -9.8*sum(trapz(phif(T*(k-1)+1:k*T).*sin(phif(T*(k-1)+1:k*T,:))));
        end


        % Energy per step
        Etot = sum(E);      %total energy over walk session
        % Emean = mean(E);    %mean over walk session
        Nsteps = round(Nsteps*t(end));
        Estep = Etot/Nsteps;
        Metrics(num, 3)=Estep;
        Metrics(num, 6)=Nsteps;
        % 
        % Ktot = sum(K);
        % Kstep=Ktot/Nsteps;
        % Metrics(num, 8)=Kstep;
        % 
        % Vtot=sum(V);
        % Vstep=Vtot/Nsteps;
        % Metrics(num, 9)=Vstep;
        % Metrics(num, 10)=Kstep-Vstep;

        % figure
        % plot(E)
        % bar(Estep)

        clear E
    end
    %% Calculate Index
    % calculates the expertise index based on extracted features and the
    % optimal/baseline values indicated below.  Update these values as
    % needed to reflect the current values
    
    f_opt=0.983853048000000;
    f_0=0.617061381000000;
    sd_opt=5.75789138400000;
    sd_0=0.699106734000000;
    E_opt=8.765627103;
    E_0=0.693985086;
    WT_opt=0.8;
    WT_0=0.0330900000000000;
    Steps_opt=30;
    Steps_0=4;
    
    
    Metrics(count+1,1)=mean(Metrics(1:count,1));
    Metrics(count+1,2)=mean(Metrics(1:count,2));
    Metrics(count+1,3)=mean(Metrics(1:count,3));
    Metrics(count+1,4)=mean(Metrics(1:count,4));
    Metrics(count+1,5)=sum(Metrics(1:count,5));
    Metrics(count+1,6)=mean(Metrics(1:count,6));
    
%     Metrics(count+1,1)=mean(Metrics(1:9,1));
%     Metrics(count+1,2)=mean(Metrics(1:9,2));
%     Metrics(count+1,3)=mean(Metrics(1:9,3));
%     Metrics(count+1,4)=mean(Metrics(1:9,4));
%     Metrics(count+1,5)=mean(Metrics(1:9,5));
%     Metrics(count+1,6)=mean(Metrics(1:9,6));
% 
%     Metrics(count+2,1)=mean(Metrics(10:16,1));
%     Metrics(count+2,2)=mean(Metrics(10:16,2));
%     Metrics(count+2,3)=mean(Metrics(10:16,3));
%     Metrics(count+2,4)=mean(Metrics(10:16,4));
%     Metrics(count+2,5)=mean(Metrics(10:16,5));
%     Metrics(count+2,6)=mean(Metrics(10:16,6));
%     
%     Metrics(count+3,1)=mean(Metrics(17:count,1));
%     Metrics(count+3,2)=mean(Metrics(17:count,2));
%     Metrics(count+3,3)=mean(Metrics(17:count,3));
%     Metrics(count+3,4)=mean(Metrics(17:count,4));
%     Metrics(count+3,5)=mean(Metrics(17:count,5));
%     Metrics(count+3,6)=mean(Metrics(17:count,6));
    
    for i=1:count+1
        index=zeros(1,5);
        
        index(1)=(Metrics(i,1)+f_0-f_opt)/(f_0);
        if index(1)>1
            index(1)=1;
        elseif index(1)<0
            index(1)=0;
        end
        index(1)=index(1)*.3;
        
        index(2)=(sd_opt/Metrics(i,2)-sd_0)/(1-sd_0);
        if index(2)>1
            index(2)=1;
        elseif index(2)<0
            index(2)=0;
        end
        index(2)=index(2)*.1;
        
        index(3)=(E_opt/Metrics(i,3)-E_0)/(1-E_0);
        if index(3)>1
            index(3)=1;
        elseif index(3)<0
            index(3)=0;
        end
        index(3)=index(3)*.1;
        
        index(4)=(Metrics(i,5)-WT_0)/(WT_opt-WT_0);
        if index(4)>1
            index(4)=1;
        elseif index(4)<0
            index(4)=0;
        end
        index(4)=index(4)*.25;
        
        index(5)=(Metrics(i,6)-Steps_0)/(Steps_opt-Steps_0);
        if index(5)>1
            index(5)=1;
        elseif index(5)<0
            index(5)=0;
        end
        index(5)=index(5)*.25;
        
        Metrics(i,7)=sum(index); 
    end
%%     
save([Patient '_' Dates{indDates} '_Metrics.mat'], 'Metrics')
end
