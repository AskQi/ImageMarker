import time
from datetime import timedelta
from functools import wraps

from logger import HiLogger

logger = HiLogger(__name__, 'ImageMarker')


def time_cost(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        formatted_time = str(timedelta(seconds=elapsed_time))  # 使用 timedelta 计算时分秒

        logger.info(f"{func.__name__} 执行耗时: {formatted_time}")
        return result

    return wrapper
