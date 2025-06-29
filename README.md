# 脑电图(EEG)眼睛状态预测系统

## 概述

本项目开发了一个医疗数据分析系统，利用脑电图(EEG)信号预测患者眼睛状态（睁眼/闭眼）。系统构建了标准化的决策树模型，实现了自动化诊断辅助，达到了93%的预测准确率，为眼科诊断提供了高效的自动化辅助工具。

## 技术栈

- Python
- Pandas
- NumPy
- Scikit-learn
- ARFF 数据处理

## 主要功能

1. 从ARFF格式文件中加载EEG数据
2. 数据预处理与标准化
3. 决策树模型训练与超参数优化
4. 模型评估与可视化
5. 模型保存与加载
6. 预测新的EEG数据

## 项目结构
eeg-eye-state-prediction/
├── eeg_eye_state_prediction.py   # 主程序文件
├── eeg_eye_state_model.pkl       # 训练好的模型
├── confusion_matrix.png          # 混淆矩阵图像
├── feature_importance.png        # 特征重要性图像
├── requirements.txt              # 依赖包列表
└── README.md                     # 项目说明文档
## 安装与运行

1. 克隆仓库：git clone https://github.com/yourusername/eeg-eye-state-prediction.git
cd eeg-eye-state-prediction
2. 安装依赖：pip install -r requirements.txt
3. 运行程序：python eeg_eye_state_prediction.py
## 结果展示

运行程序后，系统会自动完成数据加载、预处理、模型训练、评估和保存等步骤，并生成以下结果：

1. 控制台输出模型准确率和分类报告
2. 保存混淆矩阵图像`confusion_matrix.png`
3. 保存特征重要性图像`feature_importance.png`
4. 保存训练好的模型`eeg_eye_state_model.pkl`

## 使用预训练模型进行预测

如果你已经有了训练好的模型，可以直接使用它来预测新的EEG数据：
from eeg_eye_state_prediction import EEGEyeStatePredictor

# 创建预测器实例
predictor = EEGEyeStatePredictor()

# 加载预训练模型
predictor.load_model('eeg_eye_state_model.pkl')

# 假设这是新的EEG数据（14个特征）
new_eeg_data = [0.5, 0.3, 0.7, 0.2, 0.9, 0.1, 0.8, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.5]

# 进行预测
prediction = predictor.predict(new_eeg_data)
print(f"预测结果: {'睁眼' if prediction[0] == 1 else '闭眼'}")
## 贡献

如果你想为这个项目做出贡献，请遵循以下步骤：

1. Fork这个仓库
2. 创建你的特性分支 (`git checkout -b feature/your-feature`)
3. 提交你的更改 (`git commit -m 'Add some feature'`)
4. 将你的分支推送到远程仓库 (`git push origin feature/your-feature`)
5. 打开一个Pull Request

## 许可证

本项目采用MIT许可证。有关详细信息，请参阅LICENSE文件。
    