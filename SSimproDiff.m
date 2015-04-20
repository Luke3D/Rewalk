function Esdiff = SSimproDiff(SSI)

%all features if 1st col; Walk ratio is 2nd col
SSIm = mean(SSI); SSIsd = std(SSI);
EffsizeAll = SSIm(1)/SSIsd(1)/sqrt(length(SSI(:,1)));
EffsizeOne = SSIm(2)/SSIsd(2)/sqrt(length(SSI(:,2)));
Esdiff = EffsizeAll-EffsizeOne;
