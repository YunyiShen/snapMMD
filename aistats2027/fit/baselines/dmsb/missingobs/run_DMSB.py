# we used a slightly adapted version of the code to run in notebook
import sys
import torch
import numpy as np
sys.path.append('../../DMSB/')
from runner_adp import Runner
import options_adp
device = 'cuda' if torch.cuda.is_available() else "cpu"
import random
import os
import time
import csv

def main():
    seeds = [40, 41, 42, 43, 44, 1, 2, 3, 4, 5]
    # grab command line arguments 
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    # determine which task to run
    task_name = sys.argv[3]

    data = np.load(f"./data/{task_name}_data.npz")
    N_steps = data['N_steps']
    Xs =[torch.tensor(data["Xs"][i]).float().to(device) for i in range(N_steps-1)] # training data
    my_seeds = seeds[my_task_id:len(seeds):num_tasks]
    for seed in my_seeds:
        print(f"task {task_name} with seed {seed}")
        # set seed
        varr = .1
        
        iters = 10
        #opt = options_adp.set("gmm", f"{task_name}_gpu_vscale.01_var{varr}_iter{iters}")
        opt = options_adp.set("gmm", f"{task_name}_gpu_vscale.01_var{varr}")
        
        opt.v_scale = .01
        opt.var = varr
        opt.seed = seed # for saving sake
        opt.num_stage = iters
        
        run = Runner(opt, Xs)

        # actually setting up the seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        start_time = time.time()
        run.sb_alternate_train(opt)
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(f'./interpolation/DMSB_{task_name}_{seed}_timing_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['algorithm','task', 'seed','elapsed'])
            # Write timing result
            writer.writerow(['DMSB', task_name, seed, elapsed_time])


if __name__ == '__main__':
    main()