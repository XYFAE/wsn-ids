import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 模拟数据生成函数（可替换为真实数据）
def generate_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'packet_size': np.random.uniform(32, 1500, num_samples),
        'transmission_frequency': np.random.uniform(0.1, 10, num_samples),  # 次/秒
        'route_deviation': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]),
        'energy_consumption': np.random.uniform(0.1, 2.0, num_samples),
        'label': np.random.choice(['normal', 'DoS', 'MITM'], size=num_samples,
                                  p=[0.6, 0.2, 0.2])
    }
    df = pd.DataFrame(data)
    return df

# 数据预处理
def preprocess(df):
    # 标签编码
    df['label'] = df['label'].map({'normal': 0, 'DoS': 1, 'MITM': 2})
    X = df.drop('label', axis=1)
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 主程序
if __name__ == '__main__':
    # 生成模拟数据
    df = generate_data(num_samples=2000)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = preprocess(df)

    # 初始化并训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 预测与评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy * 100:.2f}%")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['normal', 'DoS', 'MITM']))

    # 特征重要性可视化
    importance = model.feature_importances_
    features = X_train.columns
    plt.figure(figsize=(8, 4))
    plt.barh(features, importance)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()