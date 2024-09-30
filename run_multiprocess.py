# coding : utf-8
# Author : yuxiang Zeng
import multiprocessing


def multi_thread_function():
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    def function(inputs):
        idx, now_input = inputs
        return idx, now_input + 1

    inputList = [(idx, 0) for (idx, j) in enumerate(range(100000))]

    result_list = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(function, inputs) for inputs in inputList]
        for future in tqdm(as_completed(futures), total=len(inputList)):
            idx, output = future.result()
            result_list.append([idx, output])

    # Sort result_list by idx
    final_result = sorted(result_list, key=lambda x: x[0])
    print('Done!')
    return final_result


def accumulate(start, chunk_size):
    """Accumulate a range of integers from start to start+chunk_size-1"""
    end = start + chunk_size
    return sum(range(start, end))

def multiprocess():
    total_numbers = 10  # 我们需要计算从0到9的和
    num_processes = 4  # 使用的进程数

    # 确定每个进程需要处理的数字数量
    chunk_size = (total_numbers + num_processes - 1) // num_processes

    # 创建一个进程池
    pool = multiprocessing.Pool(processes=num_processes)

    # 分配任务给不同进程，并传递 chunk_size 参数
    tasks = [(i * chunk_size, chunk_size) for i in range(num_processes)]
    results = pool.starmap(accumulate, tasks)  # 使用 starmap 而不是 map 来传递多个参数

    # 关闭进程池并等待所有进程完成
    pool.close()
    pool.join()
    final_result = sum(results)
    return final_result


if __name__ == '__main__':
    # result = multi_thread_function()
    # print(result)

    final_result = multiprocess()
    print("The sum is:", final_result)
