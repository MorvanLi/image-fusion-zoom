%   ��л����ʹ�ô˴��룬�˴�����������������~(@^_^@)~
%   û����Ļ���������һ������Ϣ�����������1��Ǯ��������Ĵ����ṩ1��Ǯ��Ʒ����(�䨌`��)Ŷ~
%   ��¼�Ա����̡������������ҡ������ȡ
%   �ǵģ��������û�п�������ͷƤ���������1��Ǯ�Ϳ��Խ����\(^o^)/YES!
%   С����ͰѴ����Ÿ������ǵ�Ҫ�ղغ�Ŷ(�ţ�3��)�Ũq��
%   �����ţ�https://item.taobao.com/item.htm?spm=a1z10.1-c.w4004-15151018122.5.uwGoq5&id=538759553146
%   ���������ʧЧ�����׿�����������Ҫ���ͷ�MM��������ɧ��Ŷ~(*/�بv*)
function [img_var] = variance(img)
% 
img = double(img); 
% Get rows and colums of img 
[r c] = size(img); 
     
% Mean value 
img_mean = mean(mean(img)); 
 
% Variance 
img_var = sqrt(sum(sum((img - img_mean).^2)) / (r * c ));
end