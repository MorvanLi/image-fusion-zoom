function y=mySNR(F,R)
%% ����ʵ�ּ���ͼ�������ȣ�
 % FΪ�ں�ͼ��
 % RΪ�ο�ͼ��
 % yΪ�����
%%
F=double(F);
R=double(R);
[m,n]=size(F);
temp=0;
temp1=0;
for i=1:m
    for j=1:n
        tp=R(i,j)-F(i,j);
        temp=temp+tp^2;
        temp1=temp1+F(i,j)^2;
    end 
end
y=10*log10(temp1./temp);
end
        


