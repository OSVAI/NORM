from __future__ import print_function

import sys
import os
import argparse
import socket
import time

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_gpus(nums_needed=1, mem_min=0, time_sleep=60):

    while True:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
        # If there is no ~ in the path, return the path unchanged
        with open(os.path.expanduser('~/.tmp_free_gpus') , 'r') as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [(idx, int(x.split()[2])) for idx, x in enumerate(frees)]
        idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
        if len(idx_freeMemory_pair) < 1 or len(idx_freeMemory_pair[0]) < 1 or idx_freeMemory_pair[0][1] < mem_min:
            time.sleep(time_sleep)
            # if len(idx_freeMemory_pair) > 0:
            #     eprint(os.getpid(), time.time(), ' Waiting for avarilabel GPU, top free GPU and free mem:{} aquired mem>{}'.format(idx_freeMemory_pair[0], mem_min))
            # else:
            #     eprint("idx_freeMemory_pair len {}".format(len(idx_freeMemory_pair)))
        else:
            # eprint(' Avarilabel GPU, top free GPU and free mem:{} aquired mem>{}'.format(idx_freeMemory_pair[0], mem_min))
            break

    usingGPUs = [str(idx_memory_pair[0]) for idx_memory_pair in idx_freeMemory_pair[:nums_needed] ]
    usingGPUs = ','.join(usingGPUs)
    # print('using GPU idx: #', usingGPUs)

    return usingGPUs

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(nums_needed=1, mem_min=4000, time_sleep=60)
    parser = argparse.ArgumentParser('')
    parser.add_argument('-n', '--nums_needed', type=int, default=1)
    parser.add_argument('-m', '--mem_min', type=int, default=5000)
    parser.add_argument('-t', '--time_sleep', type=int, default=60)
    opt = parser.parse_args()
    # print(opt)
    gpu_id = find_gpus(nums_needed=opt.nums_needed, mem_min=opt.mem_min, time_sleep=opt.time_sleep)
    print(gpu_id)
    # sys.exit(gpu_id)
    # print(0)