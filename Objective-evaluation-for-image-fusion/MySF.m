function SF = MySF(imgf)
%% SF(�ռ�Ƶ��) of the Fusion Image - �����ں�ͼ��Ŀռ�Ƶ��SF
% image fusion evaluate parameter - ͼ���ں����۲���
%    Example��
%      I = imread('imagename.jpg/bmp');
%      SF = MySF(I);
s = size(size(imgf));
if s(2) == 3
    imgf = rgb2gray(imgf);
end    

G = double(imgf);
[m,n] = size(G);
c1 = 0;
c2 = 0;

% ������Ƶ
for i = 1:m
    for j = 2:n
        w1 = G(i,j)-G(i,j-1);
        c1 = c1+w1^2;
    end
end
r = sqrt(c1/(m*n));

% ������Ƶ
for i = 2:m
    for j = 1:n
        w2 = G(i,j)-G(i-1,j);
        c2 = c2+w2^2;
    end
end
c = sqrt(c2/(m*n));

% ����ͼ��Ŀռ�Ƶ��
SF = sqrt(r^2+c^2);

%% 
% rf = 0;cf = 0;
% for i = 1:m
%     for j = 2:n
%         rf = rf+(imgf(i,j)-imgf(i,j-1))^2;
%     end
% end
% rf = rf/(m*n);    %������Ƶ
% 
% for j = 1:n
%     for i = 2:m
%         cf = cf+(imgf(i,j)-imgf(i-1,j))^2;
%     end
% end
% cf = cf/(m*n);    %������Ƶ
% 
% SF = sqrt(double(rf+cf))