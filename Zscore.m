function I = Zscore(X,muH,sdH,MuPsih,SdPsih)

%Healthy features (constants)
% muH = muH(1,:); sdH = sdH(1,:);

logP = -sum( 0.5*log(2*pi*sdH.^2) + ((X-muH).^2)./(2*sdH.^2), 2);  %sum of Z-scores
I = (logP-MuPsih)./SdPsih;  %Expertiese index for session s


%Features List
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Total steps per minute