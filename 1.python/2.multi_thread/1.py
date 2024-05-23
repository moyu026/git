import time
import os
import threading

start = time.perf_counter()

def do_something(seconds):
    print(f'sleep {seconds} second...')
    time.sleep(seconds)
    print('done sleep...')
    # print("当前子进程的id: {}".format(os.getpid()))
    # print("当前子进程的父进程id: {}".format(os.getppid()))


if __name__ == '__main__':

    threads = []
    for _ in range(10):
        t = threading.Thread(target=do_something, args=[1.5])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} seconds')