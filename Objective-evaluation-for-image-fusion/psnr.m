%   ��л����ʹ�ô˴��룬�˴�����������������~(@^_^@)~
%   û����Ļ���������һ������Ϣ�����������1��Ǯ��������Ĵ����ṩ1��Ǯ��Ʒ����(�䨌`��)Ŷ~
%   ��¼�Ա����̡������������ҡ������ȡ
%   �ǵģ��������û�п�������ͷƤ���������1��Ǯ�Ϳ��Խ����\(^o^)/YES!
%   С����ͰѴ����Ÿ������ǵ�Ҫ�ղغ�Ŷ(�ţ�3��)�Ũq��
%   �����ţ�https://item.taobao.com/item.htm?spm=a1z10.1-c.w4004-15151018122.5.uwGoq5&id=538759553146
%   ���������ʧЧ�����׿�����������Ҫ���ͷ�MM��������ɧ��Ŷ~(*/�بv*)
function psnr = psnr(originalimg, restoredimg)
%PSNR Summary of this function goes here
%   Detailed explanation goes here
%Peak Signal to Noise Ratio (PSNR)��ֵ�����

md = (originalimg - restoredimg).^2;
mdsize = size(md);
summation = 0;
for  i = 1:mdsize(1)
    for j = 1:mdsize(2)
        summation = summation + abs(md(i,j));
    end
end

psnr = size(originalimg, 1) * size(originalimg, 2) * max(max(originalimg.^2))/summation;
psnr = 10 * log10(psnr);
%}

end

