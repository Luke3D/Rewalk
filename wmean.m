%compute weighted mean of features for bootstrap
%Data contains the features of a clip in each row

function wm = wmean(Data)

T = Data(:,4);  %Duration of each clip

%weighted features (Step F, Sd Phi, Energy)
for f = 1:3
    wm(f) = sum(Data(:,f).*T)/sum(T);
end
    
%Other features not weigthed (Twalk, TWalk/Ttest, Nsteps)
wm(4) = mean(Data(:,4));    %avg duration of walking                 
wm(5) = sum(Data(:,5));     %walking ratio 
wm(6) = mean(Data(:,6));    %avg # of consecutive steps

end