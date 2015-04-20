%Generate training dataset for classier from multiple data files 
%Provide sampling Rate of data 
function ClassDataCreate

Fs = 30;    %***SAMPLING RATE OF THE DATA***

clear all, close all

matfiles = dir('./ClassifierData/*.mat');
csvfiles = dir('./ClassifierData/*.csv');

if length(matfiles) ~= length(csvfiles)
    error('number of csv and mat files not matching!')
end

%Load mat file and extract Start and End time

Xall = {};  %the cell array with the classifier data from all files

%Open up a parallel pool on the local machine
p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

datapath = './ClassifierData/';
tic
parfor k=1:length(matfiles)
    
    
    filename = matfiles(k).name;
    disp(filename);
    date = matfiles(k).name(5:14);  %to be changed later with an id character to find date
    
    %Start and End Times walk data 
    TestTimes = load(strcat(datapath,filename)); TestTimes = TestTimes.TestTimes;    
 
    %Find and load csv file for that date
    ok = 0; csvi = 1;
    while ~ok
        filename = csvfiles(csvi).name;
        s = find(filename == '(');  %char index marking beginning of date 
        datecsv = filename(s+1:s+10);   %csv file date
        ok = strcmp(datecsv,date);
        csvi = csvi+1;
    end

    T = readtable(strcat(datapath,filename),'ReadVariableNames',0,'HeaderLines',10); 
    accraw = table2cell(T);
    
    %Data for classifier
    X = ClassDataExtr(accraw,TestTimes,Fs);
    Xfilename = strcat(matfiles(k).name(1:15),'ClassData.mat');
%     save(Xfilename,'X');    
%     Xall = [Xall;X];    %concatenate results from each file
    Xall{k} = X;
end

save('ClassDataALL.mat','Xall');
disp('Saved')
    
t = toc


