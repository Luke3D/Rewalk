
%extract data for tree from healthy control raw acc data
Fs = 100;
Data = ClassDataExtr(accraw,TestTimes,Fs);
XH = cell2mat(Data(:,3)); YH = cell2mat(Data(:,2));
Ypred = predict(tree,XH)
err =  sum(abs(YH-Ypred))/(size(YH,1));





