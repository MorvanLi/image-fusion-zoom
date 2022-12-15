function f=MyEntroy(h)
%H return entropy of the image          ��ͼ�����
%input must be a imagehandle            ����ͼ����
%image fusion evaluate parameter        ͼ���ں����۲���
%    example
%      i=imread('imagename.bmp');
%      f=H(i); 
% ��ֵ�Ĵ�С��ʾͼ����������ƽ����Ϣ���Ķ��٣���ӳ��ͼ���о��в�ͬ�Ҷ�ֵ���صĸ��ʷֲ�
% ��Խ��˵��ͼ����ں�Ч��Խ�á�
s=size(size(h));
if s(2)==3
    h1=rgb2gray(h);
else
    h1=h;
end    
h1=double(h1);
[m,n]=size(h1);
X=zeros(1,256);
result=0;
%ͳ��ԭͼ���Ҷȼ�����
for i=1:m
    for j=1:n
        X(h1(i,j)+1)=X(h1(i,j)+1)+1;
    end
end
%������Ҷȼ����س��ֵĸ���
for k=1:256
    P(k)=X(k)/(m*n);
    if (P(k)~=0)
        result=P(k)*log2(P(k))+result;
    end
end
result=-result;
f=result;
end
