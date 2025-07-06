<<<<<<< HEAD
# Week2
=======
# US 南瓜价格区间预测
1.项目概述
本项目聚焦于 US 南瓜价格数据的分析与预测，旨在通过构建不同的回归模型来预测南瓜的最低价格和最高价格，并对模型进行优化，对比优化前后模型的性能表现为模型选择提供依据。

2.数据说明
项目使用的数据文件为 US-pumpkins.csv，包含了南瓜的各类属性信息以及价格相关数据。
![image](https://github.com/user-attachments/assets/12af56ee-3073-4da0-bf4c-96626c9126af)

3.实验过程

（1）数据预处理
统计空缺值：使用 isnull().sum() 统计每列的空缺值数目并展示。
    
![image](https://github.com/user-attachments/assets/518a788f-47a4-4d3a-8ab9-e7a2ba451e9e)
    
    删除全空列：识别并删除全部为空白的列。
  
    删除无意义列：删除空缺值比例大于 90% 的无意义列。
  
    缺失值填充：对分类变量使用众数填充缺失值，对数值变量使用中位数填充缺失值。

处理后数据：

![image](https://github.com/user-attachments/assets/d9e9c7f1-c9a5-4833-86a9-2941202b9995)

（2）预测建模

    数据准备：对字符型特征进行独热编码，确定最低价格和最高价格作为目标变量，划分训练集和测试集。
    
    模型定义与训练：定义了多种回归模型，包括 SVR、RandomForest、GradientBoosting、Ridge、Lasso 和 KNN，并分别对每个模型进行训练和预测。

结果分析：

  1.原始模型效果
![image](https://github.com/user-attachments/assets/57632725-d4cc-4280-b64a-8f902a29a15a)
 
  2.原始模型效果对比
![image](https://github.com/user-attachments/assets/13920341-03d7-499b-833c-eccdb34f5111)
 
  3.部分模型优化（SVR、Lasso原始模型表现较差，故选择对其进行优化）
![image](https://github.com/user-attachments/assets/d6e04b6e-7cae-45fb-8c67-517865164c0e)
  
  4.实验结果整合
![image](https://github.com/user-attachments/assets/3ce9cb99-61b7-427b-a512-e1eba7f7e6dc)

5.实验结论

在实际应用中，如果追求高精度的预测，RandomForest、GradientBoosting 等集成学习模型可以作为首选。同时，经过优化后的 SVR 和 Lasso 模型也表现出了良好的性能，可以根据具体情况进行选择。

>>>>>>> origin/master
