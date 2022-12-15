function [evals] = evaluation(oriA, oriB, imgf)
%% ���ö�������ָ����ں�Ч����������

% ��ֵMean Value��ָͼ�����������ػҶ�ֵ������ƽ���������۷�ӳΪƽ�����ȡ�
% imgfMEAN = mean2(imgf);
% sprintf('�ں�ͼ��ľ�ֵΪ : %.4f ',imgfMEAN)

% ��׼��Standard Deviation����ӳͼ��Ҷ�����ڻҶȾ�ֵ����ɢ�̶ȣ�����������ͼ��
% ����Ĵ�С����׼��Խ�󣬻Ҷȼ�Խ��ɢ��ͼ�񷴲�Խ��(�Աȶ�Խ��)���������������Ϣ��
imgfSTD = std2(imgf);
% imgfSTD = MyStd(imgf);
sprintf('�ں�ͼ��ı�׼��Ϊ : %.4f ',imgfSTD)
evals(1)=imgfSTD;

% �����SNR - �ź��������ķ���֮�ȣ�ֵԽ���ں�ͼ���Ӿ�Ч��Խ�á�
% imgfSNR = 20*log10(255/imgfSTD);
% sprintf('�ں�ͼ��������Ϊ : %.4f ',imgfSNR)

% ��Entropy - ��ֵ��ӳ��ͼ��Я����Ϣ���Ķ���,��ֵԽ��˵����������Ϣ��Խ�࣬�ں�Ч��Խ�á�
% imgfH = MyEntropy(imgf);
imgfH = entropy(imgf);
sprintf('�ں�ͼ�����Ϊ : %.4f ',imgfH)
evals(2)=imgfH;

% ������Clarity - ������(��Ƶ�ɷ�)��ӳͼ���΢Сϸ�ڷ������������������Խ����ͼ���ں�Ч��Խ�á�
% ƽ���ݶȷ�ӳ��ͼ��ϸ�ڷ���̶Ⱥ�����仯������һ����˵��AGֵԽ�󣬱���ͼ��Խ�������ں�Ч��Խ�á�
imgfG = Definition(imgf);
sprintf('�ں�ͼ���������(ƽ���ݶ�)Ϊ : %.4f ',imgfG)
evals(3)=imgfG;

% �ռ�Ƶ��Space Frequency - �ռ�Ƶ�ʷ�ӳͼ��ռ����ȫ���Ծˮƽ����ֵԽ��ͼ��Խ���������ں�Ч��Խ�á�
imgfSF = MySF(imgf);
sprintf('�ں�ͼ��Ŀռ�Ƶ��Ϊ : %.4f ',imgfSF)  
evals(4)=imgfSF;
% ��Եǿ�����ں���ͼ���Եϸ�ڵķḻ�̶ȣ���ֵԽ�����ں�ͼ��ı�ԵԽ������Ч��Խ�á�
imgfEI = Edge_Intensity(imgf);
sprintf('�ں�ͼ��ı�Եǿ��Ϊ : %.4f ',imgfEI)
evals(5)=imgfEI;

% ����Ϣ����Ϊ��������֮������Ե����Ȼ�һ������������һ��������Ϣ�������ȣ�����ϢԽ�����ں�ͼ�����Դͼ����ϢԽ�ࡣ
[MIInf] = MIabf(oriA, oriB, uint8(imgf));
sprintf('Դͼ�����ں�ͼ��֮��Ļ���ϢΪ : %.4f ',MIInf)
evals(6)=MIInf;
%%%%
% 
SSIM_P = metricPeilla(oriA, oriB, imgf, 1);      % SSIM
sprintf('Դͼ�����ں�ͼ��֮���SSIM_PΪ : %.4f ',SSIM_P)
evals(7)=SSIM_P;
SSIM_yang= metricYang(oriA, oriB, imgf);      % SSIM-metricYang
sprintf('Դͼ�����ں�ͼ��֮���SSIM_yangΪ : %.4f ',SSIM_yang)
evals(8)=SSIM_yang;
SSIM_Cvejic = metricCvejic(oriA, oriB, imgf,2);      % SSIM-metricCvejic
sprintf('Դͼ�����ں�ͼ��֮���SSIM_CvejicΪ : %.4f ',SSIM_Cvejic)
evals(9)=SSIM_Cvejic;
% % ����Դͼ����ں�ͼ��֮��Ľṹ�����ԣ���ֵԽ����ͼ���ں�Ч��Խ��
% % �ο����ף�Z. Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli. Image quality assessment:from error 
% % visibility to structural similarity, IEEE Transactios on Image Processing��2004��13(4):600�C612.
% [ssim1] = (ssim(oriA(:,:,1), imgf(:,:,1))+ssim(oriB(:,:,1), imgf(:,:,1)))/2;
% [ssim2] = (ssim(oriA(:,:,1), imgf(:,:,2))+ssim(oriB(:,:,2), imgf(:,:,2)))/2;
% [ssim3] = (ssim(oriA(:,:,1), imgf(:,:,3))+ssim(oriB(:,:,3), imgf(:,:,3)))/2;
% SSIM = (ssim1+ssim2+ssim3)/3;
% sprintf('Դͼ�����ں�ͼ��֮��Ľṹ������SSIMΪ : %.4f ',SSIM)
% ƽ��������Mean Cross Entropy - ֵԽС˵���ں�ͼ���Դͼ������ȡ����ϢԽ�࣬�ں�Ч��Խ��
% [MCE] = OverallCrossEntropy(oriA, oriB, uint8(imgf));
% sprintf('Դͼ�����ں�ͼ��֮���ƽ��������Ϊ : %.4f ',MCE)

% % ����Qoָ��ȡֵ����ֵԽ�ӽ�1������ͼ���ں�Ч��Խ��
% Qo1= (MyQo(oriA(:,:,1),imgf(:,:,1))+MyQo(oriB(:,:,1),imgf(:,:,1)))/2;
% Qo2= (MyQo(oriA(:,:,1),imgf(:,:,2))+MyQo(oriB(:,:,2),imgf(:,:,2)))/2;
% Qo3= (MyQo(oriA(:,:,1),imgf(:,:,3))+MyQo(oriB(:,:,3),imgf(:,:,3)))/2;
% Qo= (Qo1+Qo2+Qo3)/3;
% sprintf('Դͼ�����ں�ͼ��֮���QoֵΪ : %.4f ',Qo)
% 
% % ����Qwָ��ȡֵ����ֵԽ�ӽ�1������ͼ���ں�Ч��Խ��
% Qw1 = MyQw(oriA(:,:,1), oriB(:,:,1), imgf(:,:,1));
% Qw2 = MyQw(oriA(:,:,1), oriB(:,:,2), imgf(:,:,2));
% Qw3 = MyQw(oriA(:,:,1), oriB(:,:,3), imgf(:,:,3));
% Qw = (Qw1+Qw2+Qw3)/3;
% sprintf('Դͼ�����ں�ͼ��֮���QwֵΪ : %.4f ',Qw)
% 
% % �ں�ͼ����������ָ��Qabf����Sobel��Ե��������������ж��ٱ�Ե��Ϣ��Դͼ��ת�Ƶ����ں�ͼ��
% % ���ۺ�����Դͼ�����ں�ͼ��֮��Ľṹ���ƶȣ���ֵԽ�ӽ�1��˵��ͼ���ں�Ч��Խ�á�
% [Q1] = Qabf(oriA(:,:,1), oriB(:,:,1), imgf(:,:,1));
% [Q2] = Qabf(oriA(:,:,1), oriB(:,:,2), imgf(:,:,2));
% [Q3] = Qabf(oriA(:,:,1), oriB(:,:,3), imgf(:,:,3));
% Q = (Q1+Q2+Q3)/3;
% sprintf('Դͼ�����ں�ͼ��֮���QabfֵΪ : %.4f ',Q)
% 

% 
% % ����Ť����Warping Degree - ֱ�ӷ�ӳ���ں�Ӱ��Ĺ���ʧ��̶ȣ�ֵԽС�ں�Ч��Խ��
% % ƫ��ָ��Bias Index - ��ʾ�ں�Ӱ��͵ͷֱ��ʶನ��ԴӰ���ƫ��̶ȣ�ֵԽС�ں�Ч��Խ��
% [warp1] = analyse(imgf(:,:,1), oriB(:,:,1));
% [warp2] = analyse(imgf(:,:,2), oriB(:,:,2));
% [warp3] = analyse(imgf(:,:,3), oriB(:,:,3));
% warp = (warp1+warp2+warp3)/3;
% sprintf('Դͼ�����ں�ͼ��֮��Ĺ���Ť����Ϊ : %.4f ',warp)