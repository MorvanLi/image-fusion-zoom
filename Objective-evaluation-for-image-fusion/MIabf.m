function [h] = MIabf(image_1,image_2,image_F)
% [h] = MI(image_1,image_2,image_F) - ����ͼ��֮��Ļ���Ϣ
% image_1 - ����ͼ��1(Դͼ��)���� 
% image_2 - ����ͼ��2(Դͼ��)����
% image_F - ����ͼ��F���ں�ͼ������
% h - ���ػ�õĻ���Ϣ
%%%%%%%%%%%%%%%%%%%���÷�ʽ%%%%%%%%%%%%%%%%%%%%%%%%%%
% image_1 = imread('test1.jpg');
% image_2 = imread('test2.jpg');
% image_F = imread('fusion.jpg');
% [h] = MI(image_1,image_2,image_F);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ��������ͼ�����ݳߴ��Ƿ����
[row1 col1] = size(image_1);
[row2 col2] = size(image_2);
[rowF colF] = size(image_F);
% if (row1 ~= row2) || (col1 ~= col2)||(rowF ~= row2)||(colF ~= col2)
%     error('ͼ�����ݳߴ粻��ȣ�����');
% end;

% ��������ͼ�����ݵ���Ϣ��
e1 = ENTROPY(image_1);
e2 = ENTROPY(image_2);
eF = ENTROPY(image_F);

% �ֱ����Դͼ�����ں�ͼ��֮���������Ϣ��
eF1 = Hab(image_1,image_F);
eF2 = Hab(image_2,image_F);
% �ֱ����Դͼ�����ں�ͼ��֮��Ļ���Ϣ
MIF1 = e1+eF-eF1;
MIF2 = e2+eF-eF2;
% ���㲢���ػ���Ϣ
h = MIF1+MIF2;
sprintf('�ں�ͼ��Ļ���ϢMI��ֵΪ : %.4f ',h);  

function [e] = ENTROPY(image)
% [e] = ENTROPY(image) - ��������ͼ�����ݵ���Ϣ��
[row,col] = size(image);
counter = zeros(256,256);

% ͳ��ֱ��ͼ
image = image+1;
for i=1:row
    for j=1:col
        index = image(i,j);
        counter(index) = counter(index)+1;
    end
end
% ����ͼ����Ϣ��
total= row*col;
index = find(counter~=0);
p = counter./total;

% ��ò�����ͼ����Ϣ��
e = sum(sum(-p(index).*log2(p(index))));   

function [HabR] = Hab(image_1,image_2)
% [HabR] = Hab(image_1,image_2) - ��������ͼ���������Ϣ��
% image_1 - ����ͼ��1���� 
% image_2 - ����ͼ��2���� 

% ��ȡͼ�����ݵĴ�С�����ɼ�����
[row,col] = size(image_1);
counter = zeros(256,256);

% ͳ��ֱ��ͼ
image_1 = image_1+1;
image_2 = image_2+1;
for i=1:row
    for j=1:col
        index_1 = image_1(i,j);
        index_2 = image_2(i,j);
        counter(index_1,index_2) = counter(index_1,index_2)+1;  %����ֱ��ͼ
    end
end
% counter=counter./(row*col);
% 
% % ����AB��������
% HabR=0;
% for i=1:256
%     for j=1:256
%         if counter(i,j)~=0;
%             HabR=HabR-counter(i,j)*log2(counter(i,j));
%         end
%     end
% end
% ����������Ϣ��
total= row*col;
index = find(counter~=0);
p = counter./total;

% ��ò�����������Ϣ��
HabR = sum(sum(-p(index).*log2(p(index))));   