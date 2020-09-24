本节场景为：通过布置在不同房间的传感器获取到穿戴设备的人的移动数据，预测人的移动轨迹（人在哪个房间），场景见文件夹内示意图  
本节内容包含：  
1.数据说明见IndoorMovement\数据说明.txt  
2.如何用pandas加载csv，并且画数数据的折线图，柱状图  
3.用最小二乘法对数据进行线性拟合，并画出图像  
4.数据特征工程:所有MovementAAL_RSS文件中最小的文件包含19条数据，所以默认以19作为数据集维度，故每个文件取最后19条，根据MovementAAL_DatasetGroup中的分组对应关系，将MovementAAL_RSS作为输入，MovementAAL_target作为输出，将文件按关联关系拼成train和test集合  
5.将构建好的，维度为19的数据分别代入7种模型进行评估准确性，7种模型分别为LogisticRegression，KNN，DecisionTreeClassifier，SVM，RandomForestClassifier，GradientBoostingClassifier  
6.重复上述过程，将数据重构成25维，如果MovementAAL_RSS文件中数据不足25条则用0补全，重新用7个模型对25维的数据进行评估，评估结果发现改用25维数据后KNN模型的性能提升最好。  
7.针对性能提升最好的KNN模型，对其参数k进行网格搜索，找到最优值k=7，最后将KNN-7模型加入模型列表，评估8个模型的准确性

