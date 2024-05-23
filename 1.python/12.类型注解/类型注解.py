from typing import List, Union

list_type = List[Union[int, float]]  # 类型注解


def my_sum(alist: list_type) -> Union[int, float]:  # 函数返回值类型注解
    ret = 0
    for a in alist:
        ret = ret + a
    return ret


ints = [1, 3, 5, 7]
print(my_sum(ints))

floats = [1.0, 3.0, 5.0, 7.0]
print(my_sum(floats))

nums = [1, 3.0, 5, 7.0]
print(my_sum(nums))

strs = ['a', 'b', 'c']
print(my_sum(strs))
