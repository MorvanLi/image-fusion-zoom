function [ result ] = evaluation_xmy( imgA, imgB, imgF )
%UNTITLED ͼ���ں�����ָ��


result(1) = Definition(imgF);                       % ������
result(2) = Edge_Intensity(imgF);                   % ��Եǿ��EI
result(3) = MySF(imgF);                             % �ռ�Ƶ��SF
result(4) = metricMI(imgA,imgB,double(imgF),1);     % MI����Ϣ
result(5) = metricWang(imgA, imgB, double(imgF));   % NCIE
result(6) = metricPeilla(imgA, imgB, imgF, 1);      % SSIM
result(7) = Qabf(imgA, imgB, imgF); 				% Q^{AB/F}
result(8)=entropy(imgF);%EN��
% result(9)=metricPWW(imgA, imgB,imgF);%mssim
result(10)=avg_gradient(imgF);%ƽ���ݶ�
result(11)=variance(imgF);%��׼�� ������mse
result(12)=metricXydeas(imgA, imgB,imgF);%Q_G �ݶ�
end

