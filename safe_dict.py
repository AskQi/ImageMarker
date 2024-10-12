import threading
import queue


class ThreadSafeQueueDict:
    def __init__(self):
        self.queue = queue.Queue()  # 创建一个队列
        self.lock = threading.Lock()  # 创建一个锁
        self.data = {}  # 字典存储数据

    def set(self, key, value):
        """设置字典中的值，线程安全"""
        with self.lock:
            self.data[key] = value
            self.queue.put(key)  # 将键放入队列

    def get(self, key):
        """获取字典中的值，线程安全"""
        with self.lock:
            return self.data.get(key)

    def remove(self, key):
        """从字典中删除键，线程安全"""
        with self.lock:
            if key in self.data:
                del self.data[key]

    def __str__(self):
        with self.lock:
            return str(self.data)