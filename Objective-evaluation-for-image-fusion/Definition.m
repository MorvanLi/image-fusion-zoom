function deltaG = Definition(imgf)
%% Definition - ������(ƽ���ݶ�)
% ��ͼ���У�ĳһ����ĻҶȼ��仯��Խ��ƽ���ݶ�Avg_GradientҲԽ��,�����еط�ӳ��ͼ����΢Сϸ��
% ���������任����,��������ͼ����������(ģ��)�̶ȡ�ƽ���ݶ�Խ��ͼ����Խ�࣬Ҳ��Խ������
img = double(imgf);
[M,N] = size(img);
Grad = 0;
for i = 1:(M-1)
    for j = 1:(N-1)
        diffx = img(i,j)-img(i,j+1);     %����ͼ��F��x��y�����ϵĲ��
        diffy = img(i,j)-img(i+1,j);
        Grad = Grad+sqrt((diffx.^2+diffy.^2)/2);
    end
end

deltaG = Grad./((M-1)*(N-1));  %�����ںϺ�ͼ�������Ȧ�G