%% ����ͼ��ı�׼��
%%
function S = SD(img)
img = double(img);
[m,n] = size(img);
average = mean2(img);

s1 = 0;
for i = 1:m
    for j = 1:n
        s1 = s1+(img(i,j)-average)^2;
    end 
end
S = sqrt(s1/(m*n));
sprintf('�ں�ͼ��ı�׼��Ϊ : %.4f ',S);