clear all
datapath = '.\TestTimes\Experts\EH0\';
% datapath = 'C:\WORK\Rewalk\Data\';
filename = 'EH0_Waist (2015-05-15)RAW.csv';
datafile = [datapath filename];
ds = datastore(datafile,'NumHeaderLines',10,'ReadVariableNames',1);
preview(ds)
%Find start and end time of experiment and save it as a new file
% ds.SelectedVariableNames = 'Timestamp';
ds.RowsPerRead = 3E5;

Startdate = '5/15/2015 16:00:00.000';
Enddate = '5/15/2015 16:42:00.000';

%scan for timestamps
% ist = []; iend = [];
% rowsread = 0;     %counts # of rows read
% acc = [];
startfound = 0; endfound = 0;
Data = [];

while hasdata(ds)
    disp('read chunk')
    T = read(ds);
    chklen = size(T,1); disp(chklen)
    disp(T(1,1))
    disp(T(end,1))
    
    if ~startfound && ~endfound
        ism = ismember(table2cell(T(:,1)),{Startdate Enddate}); %search start and end
        ise = find(ism);
        if size(ise,1) > 1
            disp('start and end found')
            startfound = 1; endfound = 1;
            Data = T(ise(1):ise(2),:);
            break
        elseif size(ise,1) == 1
            disp('start found')
            startfound = 1;
            Data = T(ise(1):end,:);
        else
            continue
        end
        
    elseif startfound && ~endfound
        ism = strcmp(table2cell(T(:,1)),Enddate);
        ie = find(ism);
        if isempty(ie)
            Data = [Data;T];    %store data
        else
            endfound = 1;
            disp('End found')
            Data = [Data;T(1:ie,:)];
            break
        end
    else
        break
    end
end

filename2 = 'EH0_Waist(2015-05-15)RAW.csv';
writetable(Data,[datapath filename2])





%% find start and end absolute indices
% while hasdata(ds) || (~ist && ~iend)
%     disp(ds.RowsPerRead)
%     disp('read chunk')
%     T = read(ds);
%     chklen = size(T,1); disp(chklen)
%     disp(T(1,1))
%     disp(T(end,1))
%     %search start index
%     if ~startfound
%         ist = strcmp(table2cell(T(:,1)),Startdate);
%         ist = find(ist);
%         if ~isempty(ist)
%             indstart = rowsread+ist;    %absolute index
%             startfound = 1; 
%         end
%     end
%     %search end index
%     if ~endfound
%         ist = strcmp(table2cell(T(:,1)),Enddate);
%         ist = find(ist);
%         if ~isempty(ist)
%             indend = rowsread+ist;
%             endfound = 1;
%         end
%     end
%
%     rowsread = rowsread+chklen;
% end
%
%read the rows found and save
