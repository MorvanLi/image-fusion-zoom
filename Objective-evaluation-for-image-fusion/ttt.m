function [ result ] = evaluation( imgA, imgB, imgF )
%%  ��Եǿ��
result(1) = Edge_Intensity(imgF);

result(2) = OverallCrossEntropy(imgA, imgB, imgF);  % ���彻����


end

