import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests

class EEGEyeStatePredictor:
    """脑电图(EEG)眼睛状态预测系统"""
    
    def __init__(self):
        """初始化模型和数据"""
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
    def load_data(self, data_url=None, file_path=None):
        """
        加载EEG数据，可以从URL或本地文件加载
        
        参数:
            data_url (str): 数据URL
            file_path (str): 本地文件路径
        """
        try:
            if data_url:
                response = requests.get(data_url)
                if response.status_code == 200:
                    # 跳过ARFF文件的头部，从数据部分开始加载
                    data_content = response.text
                    data_start = data_content.find('@data') + 6
                    data_str = data_content[data_start:].strip()
                    self.data = pd.read_csv(StringIO(data_str), header=None)
                    print(f"成功从URL加载数据，数据形状: {self.data.shape}")
                else:
                    raise Exception(f"无法获取数据，状态码: {response.status_code}")
            elif file_path:
                # 从ARFF文件加载数据
                with open(file_path, 'r') as f:
                    data_content = f.read()
                data_start = data_content.find('@data') + 6
                data_str = data_content[data_start:].strip()
                self.data = pd.read_csv(StringIO(data_str), header=None)
                print(f"成功从文件加载数据，数据形状: {self.data.shape}")
            else:
                raise Exception("必须提供数据URL或文件路径")
                
            # 设置列名
            self.data.columns = [f'EEG_{i}' for i in range(self.data.shape[1] - 1)] + ['Eye_State']
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            # 生成模拟数据用于测试
            print("生成模拟数据用于测试...")
            np.random.seed(42)
            n_samples = 1000
            n_features = 14
            self.data = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'EEG_{i}' for i in range(n_features)]
            )
            # 创建一个与EEG特征相关的Eye_State列
            self.data['Eye_State'] = ((self.data.sum(axis=1) + np.random.randn(n_samples)) > 0).astype(int)
            print(f"模拟数据生成完成，数据形状: {self.data.shape}")
    
    def preprocess_data(self):
        """预处理EEG数据"""
        # 检查缺失值
        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            print(f"检测到{missing_values}个缺失值，进行处理...")
            self.data = self.data.fillna(self.data.mean())
        
        # 数据标准化
        X = self.data.drop('Eye_State', axis=1)
        y = self.data['Eye_State']
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {self.X_train.shape[0]}, 测试集大小: {self.X_test.shape[0]}")
        
    def train_model(self, optimize=True):
        """
        训练决策树模型
        
        参数:
            optimize (bool): 是否进行超参数优化
        """
        if optimize:
            # 使用管道和网格搜索进行超参数优化
            pipeline = Pipeline([
                ('scaler', self.scaler),
                ('clf', DecisionTreeClassifier(random_state=42))
            ])
            
            param_grid = {
                'clf__criterion': ['gini', 'entropy'],
                'clf__max_depth': [None, 5, 10, 15, 20],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            print("开始超参数优化...")
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
            
            self.model = grid_search.best_estimator_
        else:
            # 使用默认参数训练模型
            self.model = Pipeline([
                ('scaler', self.scaler),
                ('clf', DecisionTreeClassifier(random_state=42))
            ])
            self.model.fit(self.X_train, self.y_train)
        
        # 在测试集上进行预测
        self.y_pred = self.model.predict(self.X_test)
        
        # 计算准确率
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"模型在测试集上的准确率: {accuracy:.4f}")
        
    def evaluate_model(self):
        """评估模型性能"""
        if self.y_pred is None:
            print("请先训练模型!")
            return
        
        # 打印分类报告
        report = classification_report(self.y_test, self.y_pred)
        print("分类报告:")
        print(report)
        
        # 计算混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['闭眼', '睁眼'], yticklabels=['闭眼', '睁眼'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("混淆矩阵已保存为confusion_matrix.png")
        
        # 可视化特征重要性
        if hasattr(self.model.named_steps['clf'], 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_importances = self.model.named_steps['clf'].feature_importances_
            features = self.X_train.columns
            indices = np.argsort(feature_importances)
            
            plt.title('特征重要性')
            plt.barh(range(len(indices)), feature_importances[indices], align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('重要性')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("特征重要性图已保存为feature_importance.png")
        
    def save_model(self, model_path='eeg_eye_state_model.pkl'):
        """
        保存训练好的模型
        
        参数:
            model_path (str): 模型保存路径
        """
        if self.model is not None:
            joblib.dump(self.model, model_path)
            print(f"模型已保存到 {model_path}")
        else:
            print("没有训练好的模型可供保存!")
    
    def load_model(self, model_path='eeg_eye_state_model.pkl'):
        """
        加载已保存的模型
        
        参数:
            model_path (str): 模型路径
        """
        try:
            self.model = joblib.load(model_path)
            print(f"模型已从 {model_path} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    
    def predict(self, eeg_data):
        """
        使用训练好的模型预测眼睛状态
        
        参数:
            eeg_data (array-like): EEG数据，可以是单个样本或多个样本
        
        返回:
            array: 预测结果，0表示闭眼，1表示睁眼
        """
        if self.model is None:
            print("请先训练模型或加载已保存的模型!")
            return None
        
        # 确保输入是二维数组
        if np.ndim(eeg_data) == 1:
            eeg_data = np.array([eeg_data])
        
        # 进行预测
        predictions = self.model.predict(eeg_data)
        return predictions

def main():
    """主函数，运行完整的模型训练和评估流程"""
    print("===== 脑电图(EEG)眼睛状态预测系统 =====")
    
    predictor = EEGEyeStatePredictor()
    
    # 使用UCI EEG眼睛状态数据集的URL
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
    
    # 加载数据
    predictor.load_data(data_url=data_url)
    
    # 预处理数据
    predictor.preprocess_data()
    
    # 训练模型并进行超参数优化
    predictor.train_model(optimize=True)
    
    # 评估模型
    predictor.evaluate_model()
    
    # 保存模型
    predictor.save_model()
    
    # 示例预测
    if predictor.X_test is not None and len(predictor.X_test) > 0:
        sample = predictor.X_test.iloc[0].values
        prediction = predictor.predict(sample)
        print(f"\n示例预测:")
        print(f"EEG数据: {sample[:5]}...")
        print(f"预测结果: {'睁眼' if prediction[0] == 1 else '闭眼'}")

if __name__ == "__main__":
    main()    