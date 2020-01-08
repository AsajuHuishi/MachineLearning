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
%% 计算期望
E_cls1 = mean(cls1_data);
E_cls2 = mean(cls2_data);
E_cls3 = mean(cls3_data);
E_all = mean([E_cls1;E_cls2;E_cls3]);
%% 计算类间散度矩阵
x1 = E_all - E_cls1;
x2 = E_all - E_cls2;
x3 = E_all - E_cls3;
Sb = N1*x1'*x1/N + N2*x2'*x2/N + N3*x3'*x3/N;
%% 计算类内散度矩阵
y1 = 0;
for i = 1:N1
    y1 = y1+(cls1_data(i,:)-E_cls1)'*(cls1_data(i,:)-E_cls1);
end
y2 = 0;
for i = 1:N2
    y2 = y2+(cls2_data(i,:)-E_cls2)'*(cls2_data(i,:)-E_cls2);
end
y3 = 0;
for i = 1:N3
    y3 = y3+(cls3_data(i,:)-E_cls3)'*(cls3_data(i,:)-E_cls3);
end
Sw = N1*y1/N + N2*y2/N + N3*y3/N;
%% 求特征值和特征向量
[V,L] = eig(inv(Sw)*Sb);
% [a,b] = max(max(L));
% newspace = V(:,b);%最大特征向量
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
threshold = 0.75;
index = find(rat2 > threshold);
result2(:,1) = {'变量';'x1';'x2';'x3';'x4';'x5';'x6';'x7';'x8';'x9';'x10';'x11';'x12';'x13'};
result2(1,2:2+index(1)-1)={'Z2','Z1'};
result2(2:2+size(total,2)-2,2:2+index(1)-1)=num2cell(V(:,1:index(1)));
%% 降维为index(1)个特征
data_lda = Z*V(:,1:index(1));
%% 得到新数据
label = total(:,1);
new = [label data_lda];
figure;
new_1 = cls1_data*V(:,1:index(1));
new_2 = cls2_data*V(:,1:index(1));
new_3 = cls3_data*V(:,1:index(1));
for i=1:size(new_1,1)
    h1 = plot(new_1(i,1),new_1(i,2),'.r');
    hold on;
end
for i=1:size(new_2,1)
    h2 = plot(new_2(i,1),new_2(i,2),'*b');
    hold on;
end
for i=1:size(new_3,1)
    h3 = plot(new_3(i,1),new_3(i,2),'vc');
    hold on;
end
legend([h1(1),h2(1),h3(1)],'class1','class2','class3')
title('LDA');

data_lda = [new_1;new_2;new_3];
save('LDA.mat','data_lda','label')

hold on;
