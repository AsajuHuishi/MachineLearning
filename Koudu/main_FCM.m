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
%% ���ݱ�׼��
[Z,mu,sigma] = zscore([cls1_data;cls2_data;cls3_data]);
%% FCM 
sum0 = 0;
while(sum0<150)
[center, U] = fcm(Z,3);
U = U';
[~, fcm_label] = max(U,[],2);
label = total(:,1);
sum0 = sum((fcm_label == label)==1);
end
acc = sum0/N
%%
U = U';
re(:,1) = {'��1��Ԥ��','��2��Ԥ��','��3��Ԥ��','FCMԤ��','��ʵ���'};
re(1:3,2:179) = num2cell(U);
[~,ind] = max(U,[],1);
re(4,2:179) = num2cell(ind);
re(5,2:179) = num2cell(label);


