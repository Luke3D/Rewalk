%compute Z-score of GNB on One feature only (Walk ratio)
function I = GNBOne(Data,muH,sdH,MuPsih,SdPsih)

Feature = 5;
    
X = sum(Data(:,Feature));       %sum of Wratio over the bootstrapped sample

%Healthy features (constants) - Feature 4 is not present
muH = muH(1,Feature-1); sdH = sdH(1,Feature-1);
MuPsih = MuPsih(1,Feature-1); SdPsih = SdPsih(1,Feature-1);

logP = -sum( 0.5*log(2*pi*sdH.^2) + ((X-muH).^2)./(2*sdH.^2) );  %sum of Z-scores
I = (logP-MuPsih)./SdPsih;  %Expertiese index for session s


%Features List
% 1. Step Frequency
% 2. StdDev of frontal tilt
% 3. Energy per Step
% 4. Length of walking session (in seconds)
% 5. Ratio of time spent walking to total window time
% 6. Number of steps in walking section
