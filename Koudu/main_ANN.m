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
% train_label = [cls1_label(1:24,:);cls1_label(30:59,:);cls2_label(1:30,:);cls2_label(38:71,:);cls3_label(1:22,:);cls3_label(28:48,:)];
train_data1 = [cls1_data(1:24,:);cls1_data(30:59,:)];
train_data2 = [cls2_data(1:30,:);cls2_data(38:71,:)];
train_data3 = [cls3_data(1:22,:);cls3_data(28:48,:)];

N_train = size(train_data,1);
train_label = zeros(N_train,3);
train_label(1:54,1) = 1;
train_label(55:118,2) = 1;
train_label(119:161,3) = 1;
%% 挑选测试样本
test_data = [cls1_data(25:29,:);cls2_data(31:37,:);cls3_data(23:27,:)];
test_label = [cls1_label(25:29,:);cls2_label(31:37,:);cls3_label(23:27,:)];

%% 训练
net = newff(minmax(train_data') , [20 3] , { 'logsig' 'purelin' } , 'traingdx' ) ;
%设置训练参数
net.trainparam.show = 50 ;
net.trainparam.epochs = 250 ;
net.trainparam.goal = 0.01 ;
net.trainParam.lr = 0.01 ;
%开始训练
model = train(net,train_data', train_label');
save('model','model');
% load('model');
Y = sim(model, test_data');
%% 结果
re(:,1) = {'第1类预测','第2类预测','第3类预测','ANN预测','test_label'};
re(1:3,2:18) = num2cell(Y);
[~,ind] = max(Y,[],1);
re(4,2:18) = num2cell(ind);
re(5,2:18) = num2cell(test_label');