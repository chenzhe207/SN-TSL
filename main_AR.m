clear all
clc
close all
addpath(genpath('.\TSNE')); % add sparse coding algorithem OMP
addpath('D:\0 学习 ☆☆☆\数据库');
addpath('D:\0 学习 ☆☆☆\博士期间工作 ☆☆☆\5. TSL-LSR工作（已投SP）\TSL-LSR 代码\TSLSR (ours)\AR_noised_data')
load AR_120_50_40_Occ_01_SP_001

c = length(unique(Label));

train_num = 10;
%% select training and test samples
for ii = 1 : 10
train_data = []; test_data = []; 
train_label = []; test_label = [];
for i = 1 : c
    index = find(Label == i); 
    randindex = index(randperm(length(index)));
    train_data = [train_data DATA(:,randindex(1 : train_num))];
    train_label = [train_label  Label(randindex(1 : train_num))];
  
    test_data = [test_data DATA(:, randindex(train_num + 1 : end))];
    test_label = [test_label  Label(randindex(train_num + 1 : end))];
end
  
[Pro_Matrix,Mean_Image] = my_pca_1(train_data);
train_data = Pro_Matrix' * train_data;
test_data = Pro_Matrix' * test_data;
train_data = train_data./ repmat(sqrt(sum(train_data .* train_data)), [size(train_data, 1) 1]); %normalize\
test_data = test_data./ repmat(sqrt(sum(test_data .* test_data)), [size(test_data, 1) 1]); %normalize\

   
for i = 1 : size(train_data, 2)
    a = train_label(i);
    H_train(a, i) = 1;
end 

%% parameters
alpha = 1e-5;
beta = 1;
lambda = 1e-1;

[Q, W, Omega, obj_value] = SN_TSL(train_data, c, H_train, alpha, beta, lambda);

%% classfication
T_train = Q * W * train_data;
T_test = Q * W * test_data; 
T_train = T_train./ repmat(sqrt(sum(T_train .* T_train)), [size(T_train, 1) 1]);
T_test = T_test./ repmat(sqrt(sum(T_test .* T_test)), [size(T_test, 1) 1]);

mdl = fitcknn(T_train', train_label);
class_test = predict(mdl, T_test');
acc(ii) = sum(test_label' == class_test)/length(test_label)*100;
fprintf('ii = %d; ', ii);
fprintf('Recognition rate for our DPL1 is: %.03f; ', acc(ii));
fprintf('Max accuracy = %.03f;\n', max(acc));
end






fprintf('accuracy = %.2f±%.2f\n',mean(acc), std(acc));
% fprintf('time = train: %.4f; test: %.4f\n',TrTime,TtTime);




plot(obj_value)

WX = W * train_data;
imagesc(reshape(E(:,2),50,40))
colormap(gray(256))
imagesc(train_data(1:5,1:50))
colormap(gray(256))


