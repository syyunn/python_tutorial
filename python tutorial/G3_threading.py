# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:40:05 2018

@author: 기훈
"""

import threading

def sum(low, high):
    total = 0
    for i in range(low, high):
        total += i
    print("Subthread", total)
    
t = threading.Thread(target=sum, args=(1, 100000))
t.start()

print("Main Thread")

#%%

import threading, requests, time

def getHtml(url):
    resp = requests.get(url)
    time.sleep(1)
    print(url, len(resp.text),'chars')

t1 = threading.Thread(target=getHtml, args =('http://google.com',))
t1.start()


#%%
import threading, requests, time
 
def getHtml(url):
    resp = requests.get(url)
    time.sleep(1)
    print(url, len(resp.text), ' chars')
 
# 데몬 쓰레드
t1 = threading.Thread(target=getHtml, args=('http://google.com',))
t1.daemon = True 
t1.start()
 
print("### End ###")

#%%
'''
thread.start_new_thread ( function, args[, kwargs] )

'''
threading.start_new_thread
#%%
import _thread
import threading
import time
#%%
def print_time(threadName, delay):
    count = 0
    while count < 5:
        time.sleep(delay)
        count += 1
        print(threadName+" : "+time.ctime(time.time()))

try:
    _thread.start_new_thread(print_time, ("Thread-1",2,))
    _thread.start_new_thread(print_time, ("Thread-2",4,))
except:
    print("Error:unable to start thread")
    
while 1:
    pass
#%%
threading.activeCount()
threading.currentThread()
threading.enumerate()

#%%
import threading
import time

exitFlag = 0

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print ("Starting " + self.name)
      print_time(self.name, self.counter, 5)
      print ("Exiting " + self.name)

def print_time(threadName, delay, counter):
   while counter:
      if exitFlag:
         threadName.exit()
      time.sleep(delay)
      print ("%s: %s" % (threadName, time.ctime(time.time())))
      counter -= 1

# Create new threads
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# Start new Threads
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print ("Exiting Main Thread")












