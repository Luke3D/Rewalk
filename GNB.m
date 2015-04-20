function I = GNB(Data,muH,sdH,MuPsih,SdPsih,Wratio)

X = zeros(1,5); %Feature vector (Features = [1 2 3 5 6])

%compute wmean from data 
T = Data(:,4);  %Duration of each clip (Twalk)

%weighted features (Step F, Sd Phi, Energy)
for f = 1:3
    X(f) = sum(Data(:,f).*T)/sum(T);
end
    
%Features not weigthed (Wratio, Nsteps)
% X(4) = mean(Data(:,4));    %avg duration of walking                 
% X(5) = sum(Data(:,5));   %walking ratio 
% X(4) = Wratio(1);          %Sum of walking ratios (constant)     
X(4) = sum(Data(:,5));       %sum of Wratio over the bootstrapped sample
X(5) = mean(Data(:,6));    %avg # of consecutive steps

%Healthy features (constants)
muH = muH(1,:); sdH = sdH(1,:);
MuPsih = MuPsih(1); SdPsih = SdPsih(1);

logP = -sum( 0.5*log(2*pi*sdH.^2) + ((X-muH).^2)./(2*sdH.^2) );  %sum of Z-scores
I = (logP-MuPsih)./SdPsih;  %Expertiese index for session s


%Features List
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Length of walking session (in seconds)
% 5. Ratio of time spent walking to total window time
% 6. Number of steps in walking section
