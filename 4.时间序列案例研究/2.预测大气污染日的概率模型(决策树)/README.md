本节场景为：通过"Temperature","Humidity","Light","CO2","HumidityRatio"等换机数据预测房间是否在使用
本节内容包含：  
1.本程序使用了四种模型进行预测，并对四种模型预测效果进行评估测试，分别是：  
袋装决策树（BaggingClassifier）  
额外决策树（ExtraTreesClassifier）  
随机梯度提升（GradientBoostingClassifier）  
随机森林（RandomForestClassifier）  
2.本程序通过对例4中的梯度提升模型调整参数，来提高预测的准确率。分别调整了深度，学习率，采样集，和树数，通过brier skill score值来评价结果