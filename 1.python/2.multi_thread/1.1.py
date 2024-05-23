import time
import os
import concurrent.futures


def do_something(seconds):
    print(f'sleep {seconds} second...')
    time.sleep(seconds)
    print('done sleep...')
    # print("当前子进程的id: {}".format(os.getpid()))
    # print("当前子进程的父进程id: {}".format(os.getppid()))


if __name__ == '__main__':
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        secs = [5, 4, 3, 2, 1]
        results = [executor.submit(do_something, sec) for sec in secs]

    finish = time.perf_counter()

    print(f'Finished in {round(finish - start, 2)} seconds')
