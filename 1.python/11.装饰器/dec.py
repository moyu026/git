import time

def timeit(f):

    def wrapper(x):
        start = time.time()
        ret = f(x)
        print(time.time() - start)
        return ret
    return wrapper

@timeit
def my_func(x):
    time.sleep(x)

my_func(1)