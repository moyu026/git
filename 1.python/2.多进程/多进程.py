import time
import multiprocessing
import os

start = time.perf_counter()

def do_something():
    print('sleep 1 second...')
    time.sleep(1)
    print('done sleep...')
    print("当前子进程的id: {}".format(os.getpid()))
    print("当前子进程的父进程id: {}".format(os.getppid()))

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=do_something)
    p2 = multiprocessing.Process(target=do_something)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} seconds')