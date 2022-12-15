function [ result ] = evaluation( imgA, imgB, imgF )
%%  ��Եǿ��
result(1) = Edge_Intensity(imgF);

result(2) = OverallCrossEntropy(imgA, imgB, imgF);  % ���彻����
result(3) = MySF(imgF);                             % �ռ�Ƶ��

[EN,Qabf,SCD,FMI_w,FMI_dct,SSIM, MS_SSIM,FMI_pixel,Nabf,MI] = analysis_Reference(imgF,imgA,imgB);

result(4:13) = [EN,Qabf,SCD,FMI_w,FMI_dct,SSIM, MS_SSIM,FMI_pixel,Nabf,MI];
result(14)  = (vifvec(im2double(imgA),im2double(imgF))+vifvec(im2double(imgB),im2double(imgF)))/2;

end

