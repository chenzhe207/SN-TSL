function [Pro_Matrix,Mean_Image]=my_pca_1(Train_SET)
%���룺
%Train_SET��ѵ����������ÿ����һ��������ÿ��һ��������Dim*Train_Num
%Eigen_NUM��ͶӰά��

%�����
%Pro_Matrix��ͶӰ����
%Mean_Image����ֵͼ��

[Dim,Train_Num]=size(Train_SET);

%��ѵ����������������ά��ʱ��ֱ�ӷֽ�Э�������
if Dim<=Train_Num
    Mean_Image=mean(Train_SET,2);
    Train_SET=bsxfun(@minus,Train_SET,Mean_Image);
    R=Train_SET*Train_SET'/(Train_Num-1);
    
    [eig_vec,eig_val]=eig(R);
    eig_val=diag(eig_val);
    [~,ind]=sort(eig_val,'descend');
    W=eig_vec(:,ind);
    value = 0; Sum = sum(eig_val);
    for Eigen_NUM = 1 : size(eig_val, 1)
        value = value + eig_val(Eigen_NUM);
        if value / Sum >=0.99
            break;
        end
    end
        
        
    Pro_Matrix=real(W(:,1:Eigen_NUM));
    
else
    %����С���󣬼���������ֵ������������Ȼ��ӳ�䵽�����
    Mean_Image=mean(Train_SET,2);
    Train_SET=bsxfun(@minus,Train_SET,Mean_Image);
    R=Train_SET'*Train_SET/(Train_Num-1);
    
    [eig_vec,eig_val]=eig(R);
    eig_val=diag(eig_val);
    [val,ind]=sort(eig_val,'descend');
    W=eig_vec(:,ind);
    value = 0; Sum = sum(eig_val);
    for Eigen_NUM = 1 : size(eig_val, 1)
        value = value + val(Eigen_NUM);
        if value / Sum >=0.99
            break;
        end
    end
    Pro_Matrix=real(Train_SET*W(:,1:Eigen_NUM)*diag(val(1:Eigen_NUM).^(-1/2)));
end

end
