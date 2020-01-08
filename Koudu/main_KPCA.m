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
[patterns,mu,sigma] = zscore([cls1_data;cls2_data;cls3_data]);
% patterns = cls_data;
cov_size = N;
%% 计算核矩阵
rbfvar = 10000;
for i=1:cov_size
    for j=1:cov_size
        K(i,j) = exp(-norm(patterns(i,:)-patterns(j,:))^2/rbfvar);
        K(j,i) = K(i,j);
    end
end
unit = ones(cov_size, cov_size)/cov_size;
%% 中心化核矩阵
K_n = K - unit * K - K * unit + unit * K * unit;
%% 特征值分解
[V,D] = eig(K_n/cov_size);
% [x,index] = sort(real(diag(evaltures_1)));
% evals=flipud(x);
% index=flipud(index);
%% 将特征向量按特征值的大小顺序排序
eigValue = diag(D);
[Yt, index] = sort(eigValue, 'descend');  
eigVector = V(:,index);
eigValue = eigValue(index);

D = eigValue;
rat1 = D./sum(D);
rat2 = cumsum(D)./sum(D);
%% 调出特征值，贡献率，累计贡献率
result1(1,:)={'特征值','贡献率','累计贡献率'};
result1(2:1+size(total,1),1)=num2cell(D);
result1(2:1+size(total,1),2)=num2cell(rat1);
result1(2:1+size(total,1),3)=num2cell(rat2);
%% 主成分载荷
threshold = 0.85;
index = find(rat2 > threshold);
%% Normalization
norm_eigVector = sqrt(sum(eigVector.^2));
eigVector = eigVector./repmat(norm_eigVector,size(eigVector,1),1);
%% dimension reduction
eigVector = eigVector(:,1:index(1));
data_kpca = K * eigVector;
%% 得到新数据
label = total(:,1);
new = [label data_kpca];
save('KPCA.mat','data_kpca','label')




        
        
        
        