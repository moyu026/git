import time
import os
import multiprocessing

start = time.perf_counter()

def do_something(seconds):
    print(f'sleep {seconds} second...')
    time.sleep(seconds)
    print('done sleep...')
    # print("当前子进程的id: {}".format(os.getpid()))
    # print("当前子进程的父进程id: {}".format(os.getppid()))


if __name__ == '__main__':

    processes = []
    for _ in range(10):
        p = multiprocessing.Process(target=do_something, args=[1.5])
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} seconds')