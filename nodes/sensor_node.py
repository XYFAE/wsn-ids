import time
import random
from scapy.all import IP, UDP, Raw, send

class SensorNode:
    def __init__(self, node_id, sink_ip, is_malicious=False):
        self.node_id = node_id
        self.sink_ip = sink_ip
        self.is_malicious = is_malicious
        self.packet_count = 0  # 记录发送数据包数量

    def send_data(self):
        """模拟传感器数据发送（温度值）"""
        while True:
            temp = random.randint(20, 30)  # 生成20-30°C的随机温度
            pkt = IP(dst=self.sink_ip)/UDP(dport=1234)/Raw(load=f"{self.node_id}:{temp}")
            
            # 恶意节点行为：选择性转发（50%概率丢弃数据包）
            if self.is_malicious and random.random() < 0.5:
                print(f"[Malicious] Node {self.node_id} dropped a packet.")
            else:
                send(pkt, verbose=0)
                self.packet_count += 1
            
            time.sleep(5)  # 每5秒发送一次

if __name__ == "__main__":
    # 示例：启动一个正常节点（ID=1）和一个恶意节点（ID=5）
    # 正常节点
    # node = SensorNode(node_id="node1", sink_ip="127.0.0.1")
    # 恶意节点
    node = SensorNode(node_id="node5", sink_ip="127.0.0.1", is_malicious=True)
    node.send_data()