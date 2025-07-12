#基于南瓜价格预测项目的超参数对模型影响的研究
1.数据预处理部分请见主【main主支】

2.实验方法：超参数调优

3.实验检验标准：各个模型的模型效果

4.实验结果：

优化前（含部分模型初步优化）：

<img width="692" height="210" alt="image" src="https://github.com/user-attachments/assets/69a72e17-7776-4334-8a7e-9929125e2d68" />

优化后：

<img width="692" height="184" alt="image" src="https://github.com/user-attachments/assets/67dd0e0f-81e9-408d-bc4f-134a8fe974b2" />

5.实验结论：
多数模型（SVR、GradientBoosting、Lasso 等 ）经优化后，MSE 降低、R² 提升，预测精准度和拟合效果优化明显；RandomForest 虽然MSE 升高、R² 有所降低，但整体仍维持高性能，侧面也反映了超参数的调整不一定都会呈现正向的效果。
不过总体来说，本次的优化策略对提升模型预测效果是有效的，不同模型因算法特性，优化收益有差异，但大多数都呈现正向改进。
