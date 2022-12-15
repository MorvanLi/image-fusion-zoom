function Edgein = Edge_Intensity(imgf) 
%% Edge Intensity of the Fusion Image - �����ں�ͼ��ı�Եǿ��
% ��Եǿ�ȣ���Ҫ��ָ��Ե���ڽ����صĶԱ�ǿ�ȡ�ͼ��ϸ��Խ�ḻ�����ԵҲ��Խ������
img = double(imgf); 
% Create horizontal sobel matrix - ����ˮƽSobel���� 
w = fspecial('sobel'); 

% Get the size of img 
[M,N] = size(img); 

% ����3*3��Sobel(һ�ֱ�Ե����˲���)������ȡͼ���Ե��������ͳ��
gx = imfilter(img,w,'replicate'); 
gy = imfilter(img,w','replicate'); 

for i = 1:M
    for j = 1:N
        g(i,j) = sqrt(gx(i,j)*gx(i,j) + gy(i,j)*gy(i,j));
    end
end
Edgein = mean2(g); 