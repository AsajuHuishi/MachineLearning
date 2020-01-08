clear;
load('KLDA.mat')
data_lda = data_klda;
%% 样本总数
N = size(data_lda,1);
%% 每类样本数
N1 = sum(label(:,1)==1);
N2 = sum(label(:,1)==2);
N3 = sum(label(:,1)==3);
%% 每类样本
cls1_data = data_lda(1:N1,:);
cls2_data = data_lda(N1+1:N1+N2,:);
cls3_data = data_lda(N1+N2+1:N,:); 
cls1_label = label(1:N1,:);
cls2_label = label(N1+1:N1+N2,:);
cls3_label = label(N1+N2+1:N,:); 
%% 总样本
cls_data = [cls1_data;cls2_data;cls3_data];
%% 数据标准化
% cls1_data = zscore(cls1_data);
% cls2_data = zscore(cls2_data);
% cls3_data = zscore(cls3_data);
% cls_data = zscore(cls_data);
%% 挑选训练样本
train_data = [cls1_data(1:24,:);cls1_data(30:59,:);cls2_data(1:30,:);cls2_data(38:71,:);cls3_data(1:22,:);cls3_data(28:48,:)];
train_label = [cls1_label(1:24,:);cls1_label(30:59,:);cls2_label(1:30,:);cls2_label(38:71,:);cls3_label(1:22,:);cls3_label(28:48,:)];
train_data1 = [cls1_data(1:24,:);cls1_data(30:59,:)];
train_data2 = [cls2_data(1:30,:);cls2_data(38:71,:)];
train_data3 = [cls3_data(1:22,:);cls3_data(28:48,:)];
train_label1 = [cls1_label(1:24,:);cls1_label(30:59,:)];%%1
train_label2 = [cls2_label(1:30,:);cls2_label(38:71,:)];%%2
train_label3 = [cls3_label(1:22,:);cls3_label(28:48,:)];%%3
%% 挑选测试样本
test_data = [cls1_data(25:29,:);cls2_data(31:37,:);cls3_data(23:27,:)];
test_label = [cls1_label(25:29,:);cls2_label(31:37,:);cls3_label(23:27,:)];
% test_data1 = cls1_data(55:59,:);
% test_data2 = cls2_data(65:71,:);
% test_data3 = cls3_data(44:48,:);
% test_label1 = label(55:59,:);%%1
% test_label2 = label(65:71,:);%%2
% test_label3 = label(44:48,:);%%3
%% 变成二分类问题
% step1
step1_pos = train_data1;
step1_neg = [train_data2;train_data3];
step1_pos_label = ones(54,1);
step1_neg_label = zeros(64+43,1);
train_step1_data = [step1_pos; step1_neg];
train_step1_label = [step1_pos_label; step1_neg_label];
%%%%%%%%%%%%%%%%%%%%
% step2
step2_pos = [train_data1;train_data2];
step2_neg = train_data3;
step2_pos_label = ones(54+64,1);
step2_neg_label = zeros(43,1);
train_step2_data = [step2_pos; step2_neg];
train_step2_label = [step2_pos_label; step2_neg_label];
%% 测试
model1 = glmfit(train_step1_data,train_step1_label,'binomial', 'link', 'logit');
Y1 = glmval(model1,test_data,'logit');
model2 = glmfit(train_step2_data,train_step2_label,'binomial', 'link', 'logit');
Y2 = glmval(model2,test_data,'logit');
%% 结果
re(:,1) = {'logistic_step1','logistic_step2','logistic预测','test_label'};
re(1,2:18) = num2cell(Y1);
re(2,2:18) = num2cell(Y2);
for i = 1:17
    if(Y1(i)>0.9&&Y2(i)>0.9)
        L(i)=1;
    elseif(Y1(i)<1e-3&&Y2(i)>0.9)
        L(i)=2;
    elseif(Y1(i)<1e-3&&Y2(i)<1e-3)
        L(i)=3;
    else
        L(i)=NaN;
    end
end
re(3,2:18) = num2cell(L);
re(4,2:18) = num2cell(test_label');
