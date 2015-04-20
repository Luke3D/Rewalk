% Extracts relevant features from the acceleration data for the patient and
% dates listed below.  The results for each training session are saved as
% files labeled "Patient_(Date)_Metrics.mat"

% Columns in the output matrix Metrics:
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Length of walking session (in seconds)
% 5. Ratio of time spent walking to total window time
% 6. Number of steps in walking section
% 7. Average of twa_swing
% 8. Stddev of twa_swing
% 9. Average of twa_stance
% 10. Stddev of twa_stance
% 11. Calculated index for walking section

% Final row has average results for training session
% Be sure to indicate the appropriate sampliong frequency below
close all

minSectionLength=400; %Minimum length of walking section that must be within the window (in samples)
numFeat=11; %Number of features (columns) in Metrics matrix
addedSamples=100;  %Samples to add to beginning to window in timing data
MetricsAll = [];    %Matrix with results from each minute (over the rows)
MetricsPath = './MetricsData/NewMetrics/'; %folder where Results are saved
MetricsPathDet = './MetricsData/Detailed/NewMetrics/'; %folder for minute-by-min results

%RUN OVER MULTIPLE SUBJECTS
datapath = './TestTimes/HealthyControls/';
filenames = dir(strcat(datapath,'*.mat'));

for subj = 1:length(filenames)
    
    %RUN OVER MULTIPLE SUBJECTS
    disp('Loading file')
    load(strcat(datapath,filenames(subj).name))
    disp(['Subject ' strcat(datapath,filenames(subj).name)]); 
    Patient = filenames(subj).name(1:4);
    Dates={''}; %for now leave blank
    MetricsAll = [];    %Matrix with results from each minute (over the rows)

    %Manual Run - Insert Subject Code and Date
    % Patient='RC01';
    % Dates={'10-17'};
    
    for indDates=1:length(Dates)
        %     load(['./MatlabData/Patients/' Patient '/' Patient '_Waist(2014-' Dates{indDates} ')RAW.mat'])
        %     load(['./TestTimes/' Patient '/' Patient '_2014-' Dates{indDates} '_Times.mat'])
        
        accWaist = cell2mat(accraw(:,2:end));
        accAnkle = cell2mat(accrawAnkle(:,2:end));
        accWrist = cell2mat(accrawWrist(:,2:end));
        Fs = 100;   %Sampling freq
        
        %remove zeros
        % [r,c] = find(acc(:,1) == 0 & acc(:,2) == 0 & acc(:,3) == 0);
        % acc(r,:) = [];
        
        %plot raw data
        t = 0:1/Fs:(length(accWaist)/Fs-1/Fs);
        % t = t/60; %[min]
        figure
        set(0,'defaultlinelinewidth',2)
        plot(t,accWaist,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
        xlabel('Time [s]'), ylabel('acc [g]')
        %%
        [count,~]=size(TestTimes);
        numOfTests=0;
        start='';
        for i=1:count
            if ~strcmp(TestTimes{i,1},start)
                start=TestTimes{i,1};
                numOfTests=numOfTests+1;
                testStart(numOfTests)=i;
            end
        end
        if numOfTests==1
            Test={'Post'};
        else
            Test={'Test1', 'Test2', 'Test3'};
        end
        
        for numTest=1:numOfTests
            if numTest<numOfTests
                testCount=testStart(numTest+1)-testStart(numTest);
            else
                testCount=count-testStart(numTest)+1;
            end
            
            accWaist = accraw;
            
            % num=2;% number of section of trial you wish to evaluate (See TestTimes cell array from the appropriate Cutoff Time file)
            
            Starttime = [TestTimes{testStart(numTest)} ':00.000'];
            Endtime = [TestTimes{testStart(numTest),2} ':00.000'];
            
            i = 0; t = 0;
            while i == 0
                t = t+1;
                i = strcmp(accWaist{t}(length(accWaist{t})-11:end),Starttime);
                if t+1>length(accWaist)
                    i=1;
                end
            end
            indstart = t;
            
            i = 0; t = 0;
            while i == 0
                t = t+1;
                i = strcmp(accWaist{t}(length(accWaist{t})-11:end),Endtime);
                if t+1>length(accWaist)
                    i=1;
                end
            end
            indend = t;
            
            accWaist = accWaist(indstart:indend,:);
            accWaist = cell2mat(accWaist(:,2:end));
            
            accAnkle = accrawAnkle(indstart:indend,:);
            accAnkle = cell2mat(accAnkle(:,2:end));
            
            %RESAMPLING TO 30 HZ
            %     x=0:1/Fs:(length(acc)/Fs-1/Fs);
            %     xq=0:1/30:(length(acc)/Fs-1/Fs);
            %     acc=interp1(x, acc, xq);
            
            acctrim=accWaist;
            acctrimAnkle=accAnkle;
            numMinutes=(indend-indstart)/(Fs*60);
            %%
            for minute=1:numMinutes
                
                Metrics=zeros(count+1, numFeat);
                for num=testStart(numTest):(testCount+testStart(numTest)-1)
                    if or(and(TestTimes{num, 3}<(Fs*60*minute+1-minSectionLength),TestTimes{num, 3}>(Fs*60*(minute-1))), and(TestTimes{num, 4}<(Fs*60*minute+1),TestTimes{num, 4}>(Fs*60*(minute-1)+minSectionLength)))
                        %% Run from here for a single walk section
                        % set num and minute variables for the section you want
                        
                        close all
                        t1 = TestTimes{num,3}; t2 = TestTimes{num,4};
                        if TestTimes{num, 3}<(Fs*60*(minute-1)+1)
                            t1=Fs*60*(minute-1)+1;
                        end
                        if TestTimes{num,4}>Fs*60*minute
                            t2=Fs*60*minute;
                        end
                        accwalk = acctrim(t1:t2,:);
                        accWaist = accwalk;
                        
                        accwalkTiming = acctrim(max(t1-addedSamples,1):t2,:);
                        accWaistTiming = accwalkTiming;
                        
                        accwalkAnkle = acctrimAnkle(max(t1-addedSamples,1):t2,:);
                        accAnkle = accwalkAnkle;
                        
                        %% Compute lateral and frontal tilt from accelerometer data
                        
                        %tilt (frontal) inclination
                        phi = (180/pi)*atan2(accWaist(:,2),accWaist(:,1));    %ATAN2 does not suffer from sensitivity issues
                        %roll (lateral) inclination
                        alpha = (180/pi)*atan2(accWaist(:,3),accWaist(:,1));
                        %% simple spectral analysis of the signal
                        %                     Lmin = 3;        %minimum signal length [s] to achieve min resolution in freq spectrum (df = 1/Lmin)
                        %                     Lmin = Lmin*Fs;
                        L = length(phi);    %signal length
                        Y = fft(phi,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
                        Pyy = Y.*conj(Y)/L; %power spectrum
                        f = Fs/2*linspace(0,1,L/2+1);   %frequency axis
                        fstepsMax = 1.5;       %Upper bound on Step Freq
                        
                        %# of steps can be reliably computed only when freq spectrum has enough resolution!
                        %                     if L >= Lmin
                        [~,is] = max(Pyy(3:round(fstepsMax*L/Fs))); %approx count of steps per sec
                        Nsteps = f(is+2);
                        if isempty(Nsteps)  %if # of steps can not be reliably computed
                            Nsteps = 0;
                        end
                        %                     else
                        %                         Nsteps = nan;
                        %                     end
                        
                        Y = fft(alpha,L);
                        Pyy2 = Y.*conj(Y)/L;
                        f2 = Fs/2*linspace(0,1,L/2+1);
                        
                        %% Low pass filter
                        %phi
                        t = 0:1/Fs:(length(accWaist)/Fs-1/Fs);
                        
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
                        Metrics(num, 4:5)=[length(accWaist)/Fs, length(accWaist)/(Fs*60)];
                        
                        %% ENERGY EXPENDITURE (EE)
                        
                        %compute EE/step only if Nsteps is defined
                        if Nsteps > 0
                            
                            %WAIST
                            Epoch = 0.5; %epoch length [s]
                            T = Fs*Epoch;
                            
                            grav=mean(accWaist);
                            grav=grav/(sqrt(sum(grav.^2)));
                            
                            for k = 1:floor((size(accWaist,1)-1)/T)
                                %     E(k) = sum(trapz(acc(T*(k-1)+1:k*T,:))-15*grav);
                                int=[];
                                grav=mean(accWaist((k-1)*T+1:k*T,:));
                                grav=grav/(sqrt(sum(grav.^2)));
                                for n=T*(k-1)+1:k*T
                                    int(n-T*(k-1))=sum(abs(accWaist(n,:)-dot(accWaist(n,:),grav)*grav));
                                end
                                E(k) = sum(trapz(int));
                                
                            end
                            
                            
                            % Energy per step
                            Etot = sum(E);      %total energy over walk session
                            Nsteps = round(Nsteps*t(end));
                            Estep = Etot/Nsteps;
                            Metrics(num, 3)=Estep;
                            Metrics(num, 6)=Nsteps;
                            
                            clear E
                            
                        else
                            
                            Estep = 0;
                            Metrics(num, 3)=Estep;
                            Metrics(num, 6)=Nsteps;
                            
                        end
                        
                        %WRIST
%                         grav=mean(accWrist);
%                         grav=grav/(sqrt(sum(grav.^2)));
%                         
%                         for k = 1:floor((size(accWrist,1)-1)/T)
%                             %     E(k) = sum(trapz(acc(T*(k-1)+1:k*T,:))-15*grav);
%                             int=[];
%                             grav=mean(accWrist((k-1)*T+1:k*T,:));
%                             grav=grav/(sqrt(sum(grav.^2)));
%                             for n=T*(k-1)+1:k*T
%                                 int(n-T*(k-1))=sum(abs(accWrist(n,:)-dot(accWrist(n,:),grav)*grav));
%                             end
%                             Ew(k) = sum(trapz(int));
%                             
%                         end
%                         
%                         % Energy per step
%                         Ewtot = sum(Ew);      %total energy over walk session
%                         Ewstep = Ewtot/Nsteps;
%                         Metrics(num, 11)=Ewstep;
%                         clear Ew
                        
                        %high pass filter
                        %                     ft = [0.25 10]; %cut-off freq
                        %                     [B,A] = butter(2, 2*ft/Fs,'bandpass');   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
                        %                     accf = filtfilt(B,A,acc);
                        %                     figure, plot(accf)
                        %
                        %                     %Counts
                        %                     clear C
                        %                     accWrist = accf;
                        %                     Epoch = 1; %epoch length [s]
                        %                     T = Fs*Epoch;
                        %                     normaccW = sum(accWrist.^2,2);
                        %                     for k = 1:floor(size(accWrist,1)/T)
                        %                         C(k) = sum(accWrist((k-1)*T+1:k*T));
                        %                     end
                        %                     plot(C)
                        
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
                        threshold=mean(phiankle)+.1;
                        
                        
                        subplot(211), plot(t,phiankle,'m','LineWidth',2); title('Sagittal angle')
                        
                        %detect zero crossing for dPhi
                        signPhi = [];
                        for k=1:length(dphiankle)-1
                            signPhi(k) = dphiankle(k)*dphiankle(k+1);
                        end
                        
                        ind0 = find(signPhi < 0);

                        %Original Optimization 
%                         ind0opt = [];
%                         for k=1:length(ind0)
%                             [~,ik] = min(dphiankle(max(1,ind0(k)-1):min(ind0(k)+1,length(dphiankle)))); 
%                             ind0opt(k)=ind0(k)+ik-2;
%                             
%                         end
%                         ind0 = ind0opt; %indices of min and max values of phi
                        
                        %show min and max
                        % subplot(211),hold on, plot(ind0/Fs,dphiankle(ind0),'kx','MarkerSize',12)
                        
                        %extract max (values > 0)
                        iM = find(phiankle(ind0) > threshold);
                        ind0M = ind0(iM);   %max
                        % figure(anklefig), subplot(211); hold on, plot(ind0M/Fs,dphiankle(ind0M),'rx','MarkerSize',12)
                        
                        %Optimization alg: find local maxima
                        Wsize = 30;
                        for k = 1:length(ind0M)
                            [~,ik] = max(phiankle(max(1,ind0M(k)-Wsize):min(ind0M(k)+Wsize,length(phiankle)))); %changed here
                            ind0M(k) = max(1,ind0M(k)+ik-(Wsize+1));     %to prevent indices out of bound 
                            ind0M(k) = min(ind0M(k),length(dphiankle));
                        end
                        
                        figure(anklefig), hold on
                        subplot(211); hold on, plot(ind0M/Fs,dphiankle(ind0M),'mx','MarkerSize',6)
                        
                        tM_Ankle = ind0M/Fs;    %times of max peaks for ankle
                        
                        
                        %% Lateral and frontal tilt from waist accelerometer data
                        acc = accWaistTiming;
                        %phi - tilt (frontal) inclination
                        phiwaist = (180/pi)*atan2(acc(:,    2),acc(:,1));    %ATAN2 does not suffer from sensitivity issues
                        %alpha - roll (lateral) inclination
                        alphawaist = (180/pi)*atan2(acc(:,3),acc(:,1));
                        
                        t = 0:1/Fs:(length(acc)/Fs-1/Fs);
                        
                        ft = 0.5; %cut-off freq
                        [B,A] = butter(1, 2*ft/Fs);   %1st order, cutoff frequency 7Hz (Normalized by 2*pi*Sf) [rad/s]
                        phiwaistf = filtfilt(B,A,phiwaist);   %the filtered version of the signal
                        phiwaistf = phiwaistf*pi/180; %[rad]
                        
                        %                     figure('name','Waist Phi Filtered angles'); subplot(121)
                        %                     plot(t,phiwaistf); hold on
                        %                     title('Phi filtered'); xlim([0 t(end)+1]);
                        
                        figure(anklefig), subplot(211), hold on
                        plot(t,phiwaistf)
                        
                        phiwaist = phiwaistf;
                        
                        %% Extract waist peaks
                        
                        phiwaist0 = phiwaist - mean(phiwaist);
                        phiwaist = phiwaist0;           %zero-mean waist frontal angle
                        dphiwaist = diff(phiwaist);     %waist velocity
                        
                        %filter velocity
                        ft = 2; %cut-off freq
                        [Bvw,Avw] = butter(2, 2*ft/Fs);   %2nd order, cutoff frequency 6Hz (Normalized by 2*pi*Sf) [rad/s]
                        dphiwaistf = filtfilt(Bvw,Avw,dphiwaist);   %the filtered version of the signal
                        dphiwaist = dphiwaistf;
                        
                        figure(anklefig)
                        subplot(212), hold on, plot(t(1:end-1),dphiwaist,'b--','LineWidth',2); legend('Ankle','Waist')
                        
                        % waistfig = figure; subplot(211), hold on, plot(t,phiwaist,'b','LineWidth',2); title('Waist angle')
                        % subplot(212), hold on, plot(t(1:end-1), dphiwaistf,'b','LineWidth',2); title('Filtered Waist velocity')
                        
                        %detect zero crossing for dPhi
                        signPhi = [];
                        for k=1:length(dphiwaist)-1
                            signPhi(k) = dphiwaist(k)*dphiwaist(k+1);
                        end
                        
                        ind0 = find(signPhi < 0);
                       
                        %Original Optimization
%                         ind0opt = [];
%                         for k=1:length(ind0)
%                             [~,ik] = min(dphiwaist(max(ind0(k)-1,1):min(ind0(k)+1,length(dphiwaist))));
%                             ind0opt(k)=max(ind0(k)+ik-2,1);
%                             
%                         end
%                         ind0 = ind0opt; %indices of min and max values of phi
                        
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
                        
                        figure(anklefig), hold on
                        subplot(211); hold on, plot(ind0M/Fs,dphiwaist(ind0M),'b+','MarkerSize',6)
                        %                     legend('Ankle','Ankle peak','Waist','Waist peak')
                        
                        %optimize peaks with iterative algorithm
                        for k = 1:length(ind0M)
                            while(phiwaist(ind0M(k)) < phiwaist(min(ind0M(k)+1,length(phiwaist))))
                                ind0M(k) = ind0M(k)+1;
                            end
                            
                            while(phiwaist(ind0M(k)) < phiwaist(max(1,ind0M(k)-1)))
                                ind0M(k) = ind0M(k)-1;
                            end
                        end
                        
                        % figure(anklefig), hold on
                        % subplot(211); hold on, plot(ind0M/Fs,dphiwaist(ind0M),'g*','MarkerSize',6)
                        
                        
                        %Optimization 2: find local maximum
                        Wsize = 30;
                        for k = 1:length(ind0M)
                            [~,ik] = max(phiwaist(max(ind0M(k)-Wsize,1):min(ind0M(k)+Wsize,length(phiwaist))));
                            ind0M(k) = max(1,ind0M(k)+ik-(Wsize+1));        %to prevent indices out of bound 
                            ind0M(k) = min(ind0M(k),length(dphiankle));
                        end
                        
                        % figure(anklefig), hold on
                        % subplot(211); hold on, plot(ind0M/Fs,dphiwaist(ind0M),'g*','MarkerSize',6)
                        
                        tM_Waist = ind0M/Fs;    %times of max peaks for waist
                        
                        %% Extract time differences
                        twa_swing = []; twa_stance = [];
                        
                        %                     %loop over ankle peaks
                        %                     for i = 1:length(tM_Ankle)
                        %                         d = tM_Waist - tM_Ankle(i); %time differences between peaks
                        %                         dneg = d((d<0));   dpos = d((d>0));
                        %                         if ~isempty(dneg) && ~isempty(dpos)
                        %                             twa_swing = [twa_swing abs(max(dneg))];     %max trunk tilt to max ankle extension (swing ipsilateral)
                        %                             twa_stance =[twa_stance min(dpos)];    %max ankle extension to max trunk tilt (stance ipsilateral)
                        %                         end
                        %                     end
                        
                        %loop over ankle peaks - Exclude 1st and last peak
                        %Exclude data if two ankle peaks are found between dneg and dpos
                        for i = 1:length(tM_Ankle)
                            d = tM_Waist - tM_Ankle(i); %time differences between peaks
                            dneg = d((d<0));   dpos = d((d>0));
                            if ~isempty(dneg) && ~isempty(dpos)
                                A = abs(max(dneg)); B = min(dpos);          %A and B are the two waist peaks including the ankle peak
                                nextp = min(i+1,length(tM_Ankle));          %A and B should include only one ankle peak
                                if tM_Ankle(nextp) > B
                                    twa_swing = [twa_swing A];     %max trunk tilt to max ankle extension (swing ipsilateral)
                                    twa_stance =[twa_stance B];    %max ankle extension to max trunk tilt (stance ipsilateral)
                                end
                            end
                        end
                        
                        
                        Metrics(num,7)=mean(twa_swing);
                        Metrics(num,8)=std(twa_swing);
                        Metrics(num,9)=mean(twa_stance);
                        Metrics(num,10)=std(twa_stance);
                        
                        %Uncomment this line to save Twa_swing/Stance times in
                        %separate file
                        %                     save([Patient '_' Dates{indDates} '_' Test{numTest} '_Minute' num2str(minute) '_Section' num2str(num) '_twa.mat'], 'twa_swing', 'twa_stance')
                        
                    end
                end
                %% Calculate Index
                % calculates the expertise index based on extracted features and the
                % optimal/baseline values indicated below.  Update these values as
                % needed to reflect the current values
                
                %         f_opt=0.983853048000000;
                %         f_0=0.617061381000000;
                %         sd_opt=5.75789138400000;
                %         sd_0=0.699106734000000;
                %         E_opt=8.765627103;
                %         E_0=0.693985086;
                %         WT_opt=0.8;
                %         WT_0=0.0330900000000000;
                %         Steps_opt=30;
                %         Steps_0=4;
                %
                %
                %         Metrics(count+1,1)=mean(Metrics(1:count,1));
                %         Metrics(count+1,2)=mean(Metrics(1:count,2));
                %         Metrics(count+1,3)=mean(Metrics(1:count,3));
                %         Metrics(count+1,4)=mean(Metrics(1:count,4));
                %         Metrics(count+1,5)=sum(Metrics(1:count,5));
                %         Metrics(count+1,6)=mean(Metrics(1:count,6));
                %
                %     %     Metrics(count+1,1)=mean(Metrics(1:9,1));
                %     %     Metrics(count+1,2)=mean(Metrics(1:9,2));
                %     %     Metrics(count+1,3)=mean(Metrics(1:9,3));
                %     %     Metrics(count+1,4)=mean(Metrics(1:9,4));
                %     %     Metrics(count+1,5)=mean(Metrics(1:9,5));
                %     %     Metrics(count+1,6)=mean(Metrics(1:9,6));
                %     %
                %     %     Metrics(count+2,1)=mean(Metrics(10:16,1));
                %     %     Metrics(count+2,2)=mean(Metrics(10:16,2));
                %     %     Metrics(count+2,3)=mean(Metrics(10:16,3));
                %     %     Metrics(count+2,4)=mean(Metrics(10:16,4));
                %     %     Metrics(count+2,5)=mean(Metrics(10:16,5));
                %     %     Metrics(count+2,6)=mean(Metrics(10:16,6));
                %     %
                %     %     Metrics(count+3,1)=mean(Metrics(17:count,1));
                %     %     Metrics(count+3,2)=mean(Metrics(17:count,2));
                %     %     Metrics(count+3,3)=mean(Metrics(17:count,3));
                %     %     Metrics(count+3,4)=mean(Metrics(17:count,4));
                %     %     Metrics(count+3,5)=mean(Metrics(17:count,5));
                %     %     Metrics(count+3,6)=mean(Metrics(17:count,6));
                %
                %         for i=1:count+1
                %             index=zeros(1,5);
                %
                %             index(1)=(Metrics(i,1)+f_0-f_opt)/(f_0);
                %             if index(1)>1
                %                 index(1)=1;
                %             elseif index(1)<0
                %                 index(1)=0;
                %             end
                %             index(1)=index(1)*.3;
                %
                %             index(2)=(sd_opt/Metrics(i,2)-sd_0)/(1-sd_0);
                %             if index(2)>1
                %                 index(2)=1;
                %             elseif index(2)<0
                %                 index(2)=0;
                %             end
                %             index(2)=index(2)*.1;
                %
                %             index(3)=(E_opt/Metrics(i,3)-E_0)/(1-E_0);
                %             if index(3)>1
                %                 index(3)=1;
                %             elseif index(3)<0
                %                 index(3)=0;
                %             end
                %             index(3)=index(3)*.1;
                %
                %             index(4)=(Metrics(i,5)-WT_0)/(WT_opt-WT_0);
                %             if index(4)>1
                %                 index(4)=1;
                %             elseif index(4)<0
                %                 index(4)=0;
                %             end
                %             index(4)=index(4)*.25;
                %
                %             index(5)=(Metrics(i,6)-Steps_0)/(Steps_opt-Steps_0);
                %             if index(5)>1
                %                 index(5)=1;
                %             elseif index(5)<0
                %                 index(5)=0;
                %             end
                %             index(5)=index(5)*.25;
                %
                %             Metrics(i,7)=sum(index);
                %         end
                currentRow=1;
                MetricsTMP=zeros(1,numFeat);
                for i=1:count+1
                    if ~Metrics(i,1)==0
                        MetricsTMP(currentRow,:)=Metrics(i,:);
                        currentRow=currentRow+1;
                    end
                end
                %% Last row of Metrics contains mean over each walking section (for current minute)
                MetricsTMP(currentRow,1)=mean(MetricsTMP(1:currentRow-1,1));
                MetricsTMP(currentRow,2)=mean(MetricsTMP(1:currentRow-1,2));
                MetricsTMP(currentRow,3)=mean(MetricsTMP(1:currentRow-1,3));
                MetricsTMP(currentRow,4)=sum(MetricsTMP(1:currentRow-1,4));
                MetricsTMP(currentRow,5)=sum(MetricsTMP(1:currentRow-1,5));
                MetricsTMP(currentRow,6)=mean(MetricsTMP(1:currentRow-1,6));
                MetricsTMP(currentRow,7)=mean(MetricsTMP(1:currentRow-1,7));
                MetricsTMP(currentRow,8)=mean(MetricsTMP(1:currentRow-1,8));
                MetricsTMP(currentRow,9)=mean(MetricsTMP(1:currentRow-1,9));
                MetricsTMP(currentRow,10)=mean(MetricsTMP(1:currentRow-1,10));
                
                Metrics=MetricsTMP;
                clear MetricsTMP
                
                save([MetricsPathDet Patient '_' Dates{indDates} '_' Test{numTest} '_Minute' num2str(minute) '_Metrics.mat'], 'Metrics')
                
                %Aggregate Results from every minute (Mean) into one matrix
                MetricsAll = [MetricsAll; Metrics(end,:)];
            end
        end
    end
    
    %save all results
    
    if numOfTests==1
        Test={'Post'};
    else
        Test={'Pre'};
    end
    
    save([MetricsPath Patient '_' Dates{indDates} '_' Test{1} '_MetricsAll.mat'], 'MetricsAll')
    
    %Display summary for every minute
    figure('name','Performance'), plot(MetricsAll(:,[4 6]),'-o','LineWidth',2,'MarkerSize',4)
    legend('Walking length [s]','Number of steps')
    figure('name','Performance 2'), plot(MetricsAll(:,[1 5]),'-o','LineWidth',2,'MarkerSize',4)
    legend('Step Frequency','Walking/TotTime')
    figure('name','Energy'), plot(MetricsAll(:,[2 3]),'-o','LineWidth',2,'MarkerSize',4)
    legend('Std dev Waist','Energy/Step')
    figure('name','Ankle-Waist'), plot(MetricsAll(:,7:10),'-o','LineWidth',2,'MarkerSize',4)
    legend('Mean twa_swing','Stddev twa_swing','Mean twa_stance','Stddev twa_stance')
  
    clear TestTimes Metrics MetricsAll

end