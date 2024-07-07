# coding : utf-8
# Author : yuxiang Zeng
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


if __name__ == '__main__':
    result = multi_thread_function()
    print(result)

