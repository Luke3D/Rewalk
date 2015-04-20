function S=Slope(Data)
 
b=glmfit(Data(:,1),Data(:,2));
S=b(2);