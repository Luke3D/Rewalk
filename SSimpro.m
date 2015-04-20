function SSIm2 = SSimpro(SSI)

SSIm = mean(SSI); SSIsd = std(SSI);
SSIm2 =SSIm/SSIsd/sqrt(length(SSI));
