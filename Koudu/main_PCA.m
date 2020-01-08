clear
filename = '../wine/wine.txt';
[cls,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13] = textread(filename,'%n%n%n%n%n%n%n%n%n%n%n%n%n%n','delimiter',',');
total = [cls,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13];
%% 样本总数
N = size(total,1);
%% 每类样本数
N1 = sum(total(:,1)==1);
N2 = sum(total(:,1)==2);
N3 = sum(total(:,1)==3);
%% 每类样本
cls1_data = total(1:N1,2:end);
cls2_data = total(N1+1:N1+N2,2:end);
cls3_data = total(N1+N2+1:N,2:end); 
%% 总样本
cls_data = [cls1_data;cls2_data;cls3_data];
%% 数据标准化
[Z,mu,sigma] = zscore([cls1_data;cls2_data;cls3_data]);
%% 相关系数
covmat = cov(Z);
[V, L] = eig(covmat);
%% 计算特征值贡献率
d = diag(L);
[Yt,index] = sort(d,'descend');%降序
V = V(:,index);
D = d(index);
rat1 = D./sum(D);
rat2 = cumsum(D)./sum(D);
%% 调出特征值，贡献率，累计贡献率
result1(1,:)={'特征值','贡献率','累计贡献率'};
result1(2:2+size(total,2)-2,1)=num2cell(D);
result1(2:2+size(total,2)-2,2)=num2cell(rat1);
result1(2:2+size(total,2)-2,3)=num2cell(rat2);
%% 主成分载荷
threshold = 0.85;
index = find(rat2 > threshold);
result2(:,1) = {'变量';'x1';'x2';'x3';'x4';'x5';'x6';'x7';'x8';'x9';'x10';'x11';'x12';'x13'};
result2(1,2:2+index(1)-1)={'Z6','Z5','Z4','Z3','Z2','Z1'};
result2(2:2+size(total,2)-2,2:2+index(1)-1)=num2cell(V(:,1:index(1)));
%% 降维为index(1)个特征
data_pca = Z*V(:,1:index(1));
%% 得到新数据
label = total(:,1);
new = [label data_pca];
save('PCA.mat','data_pca','label')

