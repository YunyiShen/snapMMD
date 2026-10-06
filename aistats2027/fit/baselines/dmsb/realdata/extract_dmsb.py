import os
import shutil

def check_nine_files(directory): # this will check if the job failed to finish
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        
        # Filter out non-file entries
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
        
        # Check if the number of files is exactly 9
        return len(files) >= 8
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    

seeds = [40, 41, 42, 43, 44, 1, 2, 3, 4, 5]
tasks = ["pbmc","GoM"] #["LV20", "LV50", "repres20", "repres50", "eb", "hESC", "cmu"]

for task in tasks:
    for seed in seeds:
        var = 0.1
        results_folder = f"./results/{task}_gpu_vscale.01_var{var}_seed{seed}"
        out_file = f"./interpolation/{task}_dmsb_seed{seed}.npy"
        successful = check_nine_files(results_folder + "/backward")
        if successful:
            shutil.copy(results_folder+"/replaced_traj.npy", out_file)