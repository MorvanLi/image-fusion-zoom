function result=myOCE(A,B,F)
%% ����myOCE����ͼ��Ľ�����
 % AΪԴͼ��A��
 % BΪԴͼ��B��
 % FΪ�ں�ͼ��
 % resultΪ�����أ�
%%
[p q k]=size(F);
if k==3
F=rgb2gray(F);
end
[M,N]=size(A);
H_tempF=zeros(1,256);
%% ��ͼ��ĻҶ�ֵ��[0,255]����ͳ��
for m=1:M;
    for n=1:N;
        if F(m,n)==0;
            i=1;
        else
            i=F(m,n);
        end
        H_tempF(i)=H_tempF(i)+1;
    end
end
H_tempF=H_tempF./(M*N);
%% ���صĶ���������
H_tempA=zeros(1,256);
%% ��ͼ��ĻҶ�ֵ��[0,255]����ͳ��
for m=1:M;
    for n=1:N;
        if A(m,n)==0;
            i=1;
        else
            i=A(m,n);
        end
        H_tempA(i)=H_tempA(i)+1;
    end
end
H_tempA=H_tempA./(M*N);
%%
result1=0;
for i=1:length(H_tempF)
    if H_tempF(i)==0||H_tempA(i)==0 
        result1=result1;
    else
        result1=result1+H_tempA(i)*log2(H_tempA(i)/H_tempF(i));
    end
end
H_tempB=zeros(1,256);
%% ��ͼ��ĻҶ�ֵ��[0,255]����ͳ��
for m=1:M;
    for n=1:N;
        if B(m,n)==0;
            i=1;
        else
            i=B(m,n);
        end
        H_tempB(i)=H_tempB(i)+1;
    end
end
H_tempB=H_tempB./(M*N);
%% ���㽻����
result2=0;
for i=1:length(H_tempF)
    if H_tempF(i)==0||H_tempB(i)==0 
        result2=result2;
    else
        result2=result2+H_tempB(i)*log2(H_tempB(i)/H_tempF(i));
    end
end
result=(result1+result2)/2;
end