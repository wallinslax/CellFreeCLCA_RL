import time
import threading
import queue

# Worker 類別，負責處理資料
class Worker(threading.Thread):
  def __init__(self, queue, num):
    threading.Thread.__init__(self)
    self.queue = queue
    self.num = num

  def run(self):
    while self.queue.qsize() > 0:
      # 取得新的資料
      msg = self.queue.get()

      # 處理資料
      print("Worker %d: %s" % (self.num, msg))
      time.sleep(1)

# 建立佇列
my_queue = queue.Queue()

# 將資料放入佇列
for i in range(10):
  my_queue.put("Data %d" % i)

# 建立兩個 Worker
my_worker1 = Worker(my_queue, 1)
my_worker2 = Worker(my_queue, 2)

# 讓 Worker 開始處理資料
my_worker1.start()
my_worker2.start()

# 等待所有 Worker 結束
my_worker1.join()
my_worker2.join()

print("Done.")