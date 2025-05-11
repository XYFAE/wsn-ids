from scapy.all import sniff, UDP
import pandas as pd
from datetime import datetime

class SinkNode:
    def __init__(self):
        self.log = []  # 存储接收记录

    def packet_handler(self, pkt):
        """处理接收到的数据包"""
        if UDP in pkt and pkt[UDP].dport == 1234:
            src_ip = pkt[IP].src
            payload = pkt[Raw].load.decode()
            node_id, temp = payload.split(":")
            self.log.append({
                "timestamp": datetime.now(),
                "node_id": node_id,
                "temperature": int(temp),
                "src_ip": src_ip
            })
            print(f"Received from {node_id}: Temp={temp}°C")

    def start(self):
        """启动监听"""
        print("Sink node started. Listening on UDP port 1234...")
        sniff(filter="udp port 1234", prn=self.packet_handler, store=0)

    def save_logs(self):
        """保存日志到CSV"""
        df = pd.DataFrame(self.log)
        df.to_csv("logs/network_traffic.csv", index=False)
        print("Logs saved to logs/network_traffic.csv")

if __name__ == "__main__":
    sink = SinkNode()
    try:
        sink.start()
    except KeyboardInterrupt:
        sink.save_logs()