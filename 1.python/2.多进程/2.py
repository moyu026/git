import time
import multiprocessing
import concurrent.futures

start = time.perf_counter()


def do_something(seconds):
    print(f'sleep {seconds} second...')
    time.sleep(seconds)
    print('done sleep...')
    return seconds


if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [5, 4, 3, 2, 1]
        ##############
        # results = [executor.submit(do_something, sec) for sec in secs]
        # for f in concurrent.futures.as_completed(results):
        #     print(f.result())

        ##########
        results = executor.map(do_something, secs)
        for result in results:
            print(result)

        ###############
        # f1 = executor.submit(do_something, 1)
        # f2 = executor.submit(do_something, 1)
        # print(f1.result())
        # print(f2.result())



    finish = time.perf_counter()

    print(f'Finished in {round(finish - start, 2)} seconds')
