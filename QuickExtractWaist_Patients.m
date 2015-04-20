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
% 7. Calculated index for walking section

% Final row has average results for training session
% Be sure to indicate the appropriate sampliong frequency below
close all, clear all

Fs = 30;   %Sampling freq
minSectionLength=400; %Minimum length of walking section that must be within the window (in samples)
numFeat=7; %Number of features (columns) in Metrics matrix
addedSamples=0.5*Fs;  %Samples to add to beginning to window in timing data

MetricsPath = './MetricsData/Patients/R09/'; %folder where Avg Results per test are saved
MetricsPathDet = './MetricsData/Patients/R09/Detailed/'; %folder for each walk section results
datapathTT = './TestTimes/Patients/R09/'; %datapath of TestTimes Data
datapathacc = './TestTimes/Patients/R09/Rawaccdata/'; %datapath of raw acc data

removed = 0;    %variable accounting for data points removed from Metrics

%read all files in directory (should contain only data from one patient!)
filenames = dir(strcat(datapathTT,'*.mat'));

%Loop through all the sessions (by date)
for indDates = 1:length(filenames)
    
    MetricsMean = [];    %Init Matrix with results
    
    %LOAD TEST TIMES DATA
    disp('Loading Test Time file')
    TestTimes = load(strcat(datapathTT,filenames(indDates).name)); TestTimes = TestTimes.TestTimes;
    disp(['File ' strcat(datapathTT,filenames(indDates).name) ' loaded']);
    Patient = filenames(indDates).name(1:3);    %Patient Code
    Date = filenames(indDates).name(5:14);     %Test date
    
    %LOAD RAW WAIST and ANKLE ACC DATA 
%   (Should match format RXX_Waist(2014-01-01)RAW.mat and RXX_Ankle(2014-01-01)RAW.mat)
    disp('Loading Acc data file')
    accraw = load([datapathacc Patient '_Waist(2014-' Date(6:end) ')RAW.mat']); accraw = accraw.accraw;
    disp(['File ' datapathacc Patient '_Waist(2014-' Date(6:end) ')RAW.mat' ' loaded']);
    
    accWaist = cell2mat(accraw(:,2:end));
%     accAnkle = cell2mat(accrawAnkle(:,2:end));
%     accWrist = cell2mat(accrawWrist(:,2:end));
    
    %plot raw data
%     t = 0:1/Fs:(length(accWaist)/Fs-1/Fs);
%     figure
%     set(0,'defaultlinelinewidth',2)
%     plot(t,accWaist,'LineWidth',2); legend('x axis (g)','y axis (frontal)','z axis (perp)')
%     xlabel('Time [s]'), ylabel('acc [g]')

    %% Extract number of tests from Test Times (for patients each date consists of one test session only!)
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
%         Test={'Test1', 'Test2', 'Test3'};
        error('multiple Tests found - Check Test Times')
    end
    
    for numTest=1:numOfTests
        if numTest<numOfTests
            testCount=testStart(numTest+1)-testStart(numTest);
        else
            testCount=count-testStart(numTest)+1;
        end
        
        accWaist = accraw;
                
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
        
%         accAnkle = accrawAnkle(indstart:indend,:);
%         accAnkle = cell2mat(accAnkle(:,2:end));
        
        acctrim=accWaist;
%         acctrimAnkle=accAnkle;
        numMinutes=(indend-indstart)/(Fs*60);
        %%
        % for minute=1:numMinutes
        
        Metrics=zeros(testCount+1, numFeat);
        for num=testStart(numTest):(testCount+testStart(numTest)-1)
            %                     t1 = TestTimes{num,3}; t2 = TestTimes{num,4};
            %                     if t1<(60*Fs*(minute-1)+1)
            %                         t1=60*Fs*(minute-1)+1;
            %                     elseif t1>(60*Fs*minute)
            %                         t1=60*Fs*minute;
            %                     end
            %                     if t2<(60*Fs*(minute-1)+1)
            %                         t2=60*Fs*(minute-1)+1;
            %                     elseif t2>(60*Fs*minute)
            %                         t2=60*Fs*minute;
            %                     end
            % if (t2-t1)>minSectionLength
            %% Run from here for a single walk section
            % set num and minute variables for the section you want
            
            close all
            t1 = TestTimes{num,3}; t2 = TestTimes{num,4};
            %                         if TestTimes{num, 3}<(Fs*60*(minute-1)+1)
            %                             t1=Fs*60*(minute-1)+1;
            %                         end
            %                         if TestTimes{num,4}>Fs*60*minute
            %                             t2=Fs*60*minute;
            %                         end
            accwalk = acctrim(t1:t2,:);
            accWaist = accwalk;
            
            accwalkTiming = acctrim(max(t1-addedSamples,1):t2,:);
            accWaistTiming = accwalkTiming;
            
%             accwalkAnkle = acctrimAnkle(max(t1-addedSamples,1):t2,:);
%             accAnkle = accwalkAnkle;
            
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
            
            Metrics(num-testStart(numTest)+1,1:2)=[Nsteps, ps];
            Metrics(num-testStart(numTest)+1, 4:5)=[length(accWaist)/Fs, length(accWaist)/(Fs*60*numMinutes)];
            
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
                Metrics(num-testStart(numTest)+1, 3)=Estep;
                Metrics(num-testStart(numTest)+1, 6)=Nsteps;
                
                clear E
                
            else
                
                Estep = 0;
                Metrics(num-testStart(numTest)+1, 3)=Estep;
                Metrics(num-testStart(numTest)+1, 6)=Nsteps;
                
            end
            
             
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
            
%             figure(anklefig), subplot(211), hold on
%             plot(t,phiwaistf)
            
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
            
%             figure(anklefig)
%             subplot(212), hold on, plot(t(1:end-1),dphiwaist,'b--','LineWidth',2); legend('Ankle','Waist')
            
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
            
            
            %optimize peaks with iterative algorithm
            for k = 1:length(ind0M)
                while(phiwaist(ind0M(k)) < phiwaist(min(ind0M(k)+1,length(phiwaist))))
                    ind0M(k) = ind0M(k)+1;
                end
                
                while(phiwaist(ind0M(k)) < phiwaist(max(1,ind0M(k)-1)))
                    ind0M(k) = ind0M(k)-1;
                end
            end
                        
            
            %Optimization 2: find local maximum
            Wsize = 30;
            for k = 1:length(ind0M)
                [~,ik] = max(phiwaist(max(ind0M(k)-Wsize,1):min(ind0M(k)+Wsize,length(phiwaist))));
                ind0M(k) = max(1,ind0M(k)+ik-(Wsize+1));        %to prevent indices out of bound
                ind0M(k) = min(ind0M(k),length(dphiwaist));
            end
            
            % figure(anklefig), hold on
            % subplot(211); hold on, plot(ind0M/Fs,dphiwaist(ind0M),'g*','MarkerSize',6)
            
            tM_Waist = ind0M/Fs;    %times of max peaks for waist
            
        end
        %% Calculate Index
        % calculates the expertise index based on extracted features and the
        % optimal/baseline values indicated below.  Update these values as
        % needed to reflect the current values
        
        currentRow=testCount+1;
        MetricsTMP=Metrics;
        
        %% Last row of Metrics contains weighted mean over all walking section
  
        %Remove Outliers for Step Freq
        indStepf = find( MetricsTMP(1:end,1) > 1 | MetricsTMP(1:end,1) < 0.25);
        MetricsTMP(indStepf,:) = [];
        removed = removed + 1;  %counts number of removed datapoints
        currentRow = size(MetricsTMP,1)+1; 
   
%           %Remove data points outside two std dev from the mean for each
%         %feature
%         for f =1:3
%             mu = mean(MetricsTMP(1:end-1,f)); sdev = std(MetricsTMP(1:end-1,f));
%             ind = find(MetricsTMP(1:end-1,f) > mu+2.7*sdev | MetricsTMP(1:end-1,f) < mu-2.7*sdev);
%             MetricsTMP(ind,:) = [];
%             removed = removed + 1;  %counts number of removed datapoints
%         end
%         
        

        MetricsTMP(currentRow,4)=sum(MetricsTMP(1:currentRow-1,4)); %Tot Length of Walk sections (sum(Ti))
        MetricsTMP(currentRow,5)=sum(MetricsTMP(1:currentRow-1,5)); %Tot ratio of Walking/TestTime 
                
        %last row contains weighted mean of each feature 
        MetricsTMP(end,1)=sum(MetricsTMP(1:end-1,1).*MetricsTMP(1:end-1,4))/MetricsTMP(end,4);
        MetricsTMP(end,2)=sum(MetricsTMP(1:end-1,2).*MetricsTMP(1:end-1,4))/MetricsTMP(end,4);
        MetricsTMP(end,3)=sum(MetricsTMP(1:end-1,3).*MetricsTMP(1:end-1,4))/MetricsTMP(end,4);
        MetricsTMP(end,6)=mean(MetricsTMP(1:end-1,6));
        
        
%         MetricsTMP(currentRow,1)=sum(MetricsTMP(1:currentRow-1,1).*MetricsTMP(1:currentRow-1,4))/MetricsTMP(currentRow,4);
%         MetricsTMP(currentRow,2)=sum(MetricsTMP(1:currentRow-1,2).*MetricsTMP(1:currentRow-1,4))/MetricsTMP(currentRow,4);
%         MetricsTMP(currentRow,3)=sum(MetricsTMP(1:currentRow-1,3).*MetricsTMP(1:currentRow-1,4))/MetricsTMP(currentRow,4);
%         MetricsTMP(currentRow,6)=mean(MetricsTMP(1:currentRow-1,6));
%         
        %Ankle-Waist metrics
%         MetricsTMP(currentRow,7)=sum(MetricsTMP(1:currentRow-1,7).*MetricsTMP(1:currentRow-1,4))/MetricsTMP(currentRow,4);
%         MetricsTMP(currentRow,8)=sum(MetricsTMP(1:currentRow-1,8).*MetricsTMP(1:currentRow-1,4))/MetricsTMP(currentRow,4);
%         MetricsTMP(currentRow,9)=sum(MetricsTMP(1:currentRow-1,9).*MetricsTMP(1:currentRow-1,4))/MetricsTMP(currentRow,4);
%         MetricsTMP(currentRow,10)=sum(MetricsTMP(1:currentRow-1,10).*MetricsTMP(1:currentRow-1,4))/MetricsTMP(currentRow,4);
        
        Metrics=MetricsTMP;
        clear MetricsTMP
        
        %save metrics for each walk section
        disp(['Saving ' MetricsPathDet Patient '_' Date '_Metrics.mat']);
        save([MetricsPathDet Patient '_' Date '_Metrics.mat'], 'Metrics')


        %save weighted average metrics (last row of Metrics.mat)
        MetricsMean = Metrics(end,:);
        disp(['Saving ' MetricsPath Patient '_' Date '_MetricsMean.mat']);
        save([MetricsPath Patient '_' Date '_MetricsMean.mat'], 'MetricsMean')
      
        clear Metrics
    end
end

%save all results

% if numOfTests==1
%     Test={'Post'};
% else
%     Test={'Multi'};
% end

% save([MetricsPath Patient '_' Date '_MetricsAll.mat'], 'MetricsAll')

%