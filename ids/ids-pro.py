import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# 生成模拟数据（实际使用时替换为真实数据如Kitsune/CICIDS2017）
def generate_synthetic_data(samples=10000, features=50):
    # 正常流量（80%样本）
    normal_data = np.random.normal(loc=0, scale=1, size=(int(samples*0.8), features))
    normal_labels = np.zeros(int(samples*0.8))
    
    # 攻击流量（20%样本，包含DoS、僵尸网络等）
    attack_data = np.random.uniform(low=-3, high=3, size=(int(samples*0.2), features))
    attack_labels = np.ones(int(samples*0.2))
    
    X = np.vstack([normal_data, attack_data])
    y = np.hstack([normal_labels, attack_labels])
    return X, y

# 数据预处理
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test

# 定义混合深度学习模型（Transformer + CNN + VAE-LSTM）
def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Transformer分支
    x1 = MultiHeadAttention(num_heads=4, key_dim=8)(inputs, inputs)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + inputs)
    
    # CNN分支
    x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x2 = Conv1D(filters=64, kernel_size=3, activation='relu')(x2)
    x2 = Flatten()(x2)
    
    # LSTM分支（简化版VAE-LSTM）
    x3 = LSTM(64, return_sequences=True)(inputs)
    x3 = LSTM(32)(x3)
    
    # 特征融合
    combined = tf.keras.layers.concatenate([x1[:, -1, :], x2, x3])
    
    # 分类层
    outputs = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 主流程
if __name__ == "__main__":
    # 生成数据
    X, y = generate_synthetic_data(samples=10000, features=50)
    
    # 预处理
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # 构建模型
    model = build_hybrid_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # 训练
    history = model.fit(X_train, y_train, 
                       epochs=15, 
                       batch_size=64, 
                       validation_split=0.1,
                       verbose=1)
    
    # 评估
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1 Score: {f1_score(y_test, y_pred):.4f}")