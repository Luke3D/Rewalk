%Extract features to classify walk sections
function f = featureextr(X,Fs)

f = [];

muX = mean(X);   %mean x,y,z
stdX = std(X);   %std dev x,y,z
minX = min(X); 
maxX = max(X);

%Power spectra
phi = (180/pi)*atan2(X(:,2),X(:,1));    %frontal tilt
alpha = (180/pi)*atan2(X(:,3),X(:,1));  %lateral tilt
L = length(phi);    %signal length
Y = fft(phi,L);     %to improve DFT alg performance set L = NFFT = 2^nextpow2(L)
Y2 = fft(alpha,L);
Pyy = Y.*conj(Y)/L; Pyy = Pyy./sum(Pyy);    %normalized Power spectrum
Pyy2 = Y2.*conj(Y2)/L;   Pyy2 = Pyy2./sum(Pyy2);   %normalized Power spectrum
f = Fs/2*linspace(0,1,L/2+1);   %frequency axis

[Pmax_phi,fmax_phi] = max(Pyy(2:L/2+1));          %exclude DC component (1st freq component)
[Pmax_alpha,fmax_alpha] = max(Pyy2(2:L/2+1)); 
fmax_phi = Fs/(2*L)*(fmax_phi+1);                 %add one because it starts from 2nd freq component
fmax_alpha = Fs/(2*L)*(fmax_alpha+1);

%power spectrum up to max freq
fmax = 5;       %max freq to run analysis
Pphi = Pyy(2:fmax*L/Fs+1);  
Palpha = Pyy2(2:fmax*L/Fs+1);

%aggregate features in a row
f = [muX stdX minX maxX fmax_phi Pmax_phi fmax_alpha Pmax_alpha Pphi' Palpha'];





