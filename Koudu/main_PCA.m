clear
filename = '../wine/wine.txt';
[cls,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13] = textread(filename,'%n%n%n%n%n%n%n%n%n%n%n%n%n%n','delimiter',',');
total = [cls,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13];
%% ��������
N = size(total,1);
%% ÿ��������
N1 = sum(total(:,1)==1);
N2 = sum(total(:,1)==2);
N3 = sum(total(:,1)==3);
%% ÿ������
cls1_data = total(1:N1,2:end);
cls2_data = total(N1+1:N1+N2,2:end);
cls3_data = total(N1+N2+1:N,2:end); 
%% ������
cls_data = [cls1_data;cls2_data;cls3_data];
%% ���ݱ�׼��
[Z,mu,sigma] = zscore([cls1_data;cls2_data;cls3_data]);
%% ���ϵ��
covmat = cov(Z);
[V, L] = eig(covmat);
%% ��������ֵ������
d = diag(L);
[Yt,index] = sort(d,'descend');%����
V = V(:,index);
D = d(index);
rat1 = D./sum(D);
rat2 = cumsum(D)./sum(D);
%% ��������ֵ�������ʣ��ۼƹ�����
result1(1,:)={'����ֵ','������','�ۼƹ�����'};
result1(2:2+size(total,2)-2,1)=num2cell(D);
result1(2:2+size(total,2)-2,2)=num2cell(rat1);
result1(2:2+size(total,2)-2,3)=num2cell(rat2);
%% ���ɷ��غ�
threshold = 0.85;
index = find(rat2 > threshold);
result2(:,1) = {'����';'x1';'x2';'x3';'x4';'x5';'x6';'x7';'x8';'x9';'x10';'x11';'x12';'x13'};
result2(1,2:2+index(1)-1)={'Z6','Z5','Z4','Z3','Z2','Z1'};
result2(2:2+size(total,2)-2,2:2+index(1)-1)=num2cell(V(:,1:index(1)));
%% ��άΪindex(1)������
data_pca = Z*V(:,1:index(1));
%% �õ�������
label = total(:,1);
new = [label data_pca];
save('PCA.mat','data_pca','label')

