% Extracts relevant features from the acceleration data for the patient and
% dates listed below.  The results for each training session are saved as
% files labeled "Patient_(Date)_Metrics.mat"

%TODO: Make separate Extraction Script for Healthy Controls

% Columns in the output matrix Metrics:
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Number of steps per minute

% Final row has average results for training session
% Be sure to indicate the appropriate sampling frequency below
close all, clear all

Fs = 100;   %Original Sampling freq
Resample = 0;   %if resample to 30 Hz
minSectionLength=4*Fs; %Minimum length of walking section that must be within the window (in samples)
numFeat=4; %Number of features (columns) in Metrics matrix

MetricsPath = './MetricsData/Experts/SDD/'; %folder where Avg Results per test are saved
MetricsPathDet = './MetricsData/Experts/SDD/Detailed/'; %folder for each walk section results
datapathTT = './TestTimes/Experts/SDD/'; %datapath of TestTimes Data
datapathacc = './TestTimes/Experts/SDD/'; %datapath of raw acc data

removed = 0;    %variable accounting for data points removed from Metrics

%read all files in directory (should contain only data from one patient!)
filenames = dir(strcat(datapathTT,'*.mat'));

%Loop through all the sessions (by date)
Fs1=Fs;
for indDates = 1:length(filenames)
    
    MetricsMean = [];    %Init Matrix with results
    
    %LOAD TEST TIMES DATA - Format RXX_2014-01-01
    disp('Loading Test Time file')
    TestTimes = load(strcat(datapathTT,filenames(indDates).name)); TestTimes = TestTimes.TestTimes;
    disp(['File ' strcat(datapathTT,filenames(indDates).name) ' loaded']);
    Patient = filenames(indDates).name(1:3);    %Patient Code
    % Date = filenames(indDates).name(5:14);     %Test date - Format RXX_2014-01-01
    Date = filenames(indDates).name(11:20);     %Test date - Format RXX_Waist(2014-01-01)
    
    %LOAD RAW WAIST ACC DATA
    %   (Should match format RXX_Waist(2014-01-01)RAW.csv)
    file = [datapathacc Patient '_Waist(' Date(1:end) ')RAW.csv'];
    disp(['Loading file ' file])
    accrawcsv = readtable(file,'ReadVariableNames',0,'HeaderLines',11);
    accraw = table2cell(accrawcsv);
    accWaist = cell2mat(accraw(:,2:end));
    accWaist1=accWaist;
    disp([file ' Loaded']);
    clear accrawcsv
    
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
    
    for numTest=1:numOfTests
        
        if numOfTests == numTest
             testCount=count-testStart(numTest)+1; %number of walking sections within test
        else
             testCount=testStart(numTest+1)-testStart(numTest);
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
        scale=Fs1/30; % factor to rescale TestTimes up to 100 Hz
        
        if Resample
            Fs = 30;    %new sampling rate
            disp('Interpolating to 30 Hz...')
            x=0:1/Fs1:(length(accWaist)/Fs1-1/Fs1); %original time vector
            xi=0:1/Fs:(length(accWaist)/Fs1-1/Fs1);   %interpol ated time vector
            accWaist=interp1(x, accWaist, xi);
            scale=1;
        end
        
        TestLength=Fs*60*6; %Total samples per test
        TimeStart=ceil(TestTimes{testStart(numTest),3}*scale); % index of test start
        acctrim=accWaist;
        numMinutes=6;   %length of each test
        
        Metrics=zeros(numMinutes, numFeat); %Initialize Detailed Metrics matrix (one row per minute of each test)

        %%
        for minute=1:numMinutes
            
%             Metrics=zeros(testCount+1, numFeat);

            for num=1:testCount
                t1 = ceil(TestTimes{num+testStart(numTest)-1,3}*scale);
                t2 = ceil(TestTimes{num+testStart(numTest)-1,4}*scale);
                if t1<(60*Fs*(minute-1)+TimeStart)
                    t1=60*Fs*(minute-1)+TimeStart;
                elseif t1>(60*Fs*minute+TimeStart-1)
                    t1=60*Fs*minute+TimeStart-1;
                end
                if t2<(60*Fs*(minute-1)+TimeStart)
                    t2=60*Fs*(minute-1)+TimeStart;
                elseif t2>(60*Fs*minute+TimeStart-1)
                    t2=60*Fs*minute+TimeStart-1;
                end
                if (t2-t1)>minSectionLength
                    %% Run from here for a single walk section
                    % set num and minute variables for the section you want
                    
                    % First line is for TestTimes at 100Hz (for R10)
                    accwalk = acctrim(t1:t2,:);
                    accWaist = accwalk;
                    
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
                    if L<250
                        [~,is] = max(Pyy(2:round(fstepsMax*L/Fs))); %approx count of steps per sec
                        Nsteps = f(is+1);
                    else
                        [~,is] = max(Pyy(3:round(fstepsMax*L/Fs))); %approx count of steps per sec
                        Nsteps = f(is+2);
                    end
                    if isempty(Nsteps)  %if # of steps can not be reliably computed
                        Nsteps = 0;
                    end
                    
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
                    
                    Metrics(minute,1:2)=[Nsteps, ps];
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
                        Nsteps = Nsteps*t(end);
                        Estep = Etot/Nsteps;
                        Metrics(minute, 3)=Estep;
                        Metrics(minute, 4)=Nsteps;
                        
                        clear E
                        
                    else
                        Estep = 0;
                        Metrics(minute, 3)=Estep;
                        Metrics(minute, 4)=Nsteps;
                    end
                    Metrics(minute,5)=t2-t1;
 
                end

            end
        end
        %save metrics for each walk section
        disp(['Saving ' MetricsPathDet Patient '_' Date '_Test' num2str(numTest) '_Metrics.mat']);
        save([MetricsPathDet Patient '_' Date '_Test' num2str(numTest) '_Metrics.mat'], 'Metrics')
        
        %Average over the test
        MetricsMean = [MetricsMean;mean(Metrics(:,1:numFeat))];
        
        clear Metrics
    
    end
    
    %save MetricsMean
    disp(['Saving ' MetricsPath Patient '_' Date '_MetricsMean.mat']);
    save([MetricsPath Patient '_' Date '_MetricsMean.mat'], 'MetricsMean')
        
    
end