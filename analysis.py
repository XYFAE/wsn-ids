import pandas as pd
import matplotlib.pyplot as plt

def detect_anomalies():
    # 加载日志数据
    df = pd.read_csv("logs/network_traffic.csv")
    
    # 统计每个节点的数据包数量
    node_counts = df['node_id'].value_counts().reset_index()
    node_counts.columns = ['node_id', 'packet_count']
    
    # 检测异常：数据包数量显著低于平均值（假设恶意节点丢包率50%）
    avg_count = node_counts['packet_count'].mean()
    threshold = avg_count * 0.6  # 低于平均值的60%视为异常
    anomalies = node_counts[node_counts['packet_count'] < threshold]
    
    # 输出结果
    if not anomalies.empty:
        print("[ALERT] Detected potential malicious nodes:")
        print(anomalies)
    else:
        print("No anomalies detected.")
    
    # 可视化
    plt.bar(node_counts['node_id'], node_counts['packet_count'], color='skyblue')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
    plt.xlabel('Node ID')
    plt.ylabel('Packet Count')
    plt.title('WSN Node Packet Count Analysis')
    plt.legend()
    plt.savefig('logs/packet_analysis.png')
    plt.show()

if __name__ == "__main__":
    detect_anomalies()