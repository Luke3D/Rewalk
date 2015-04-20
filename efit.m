%Fit exp of the form f(x) = a*exp(b*x)
function a = efit(Data)

x = Data(:,1); y = Data(:,2);
f = fit(x,y,'exp1'); a = f.a; 
%b = fit.b;