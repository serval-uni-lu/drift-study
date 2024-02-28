import os

path = "/scratch/users/tsimonetto/drift-study/models/lcld_201317_ds_time"
not_here = 0
here = 0
for model in ["rf_lcld", "rf_lcld_400"]:
    for j in ([""]):
        for win_size in [50000, 100000, 200000, 400000, 100000000000]:
            for end_idx in range(405000, 1115000, 5000):
                start_idx = max(end_idx-win_size, 0)
                l_path = f"{path}/{model}{j}_{start_idx}_{end_idx}.joblib.pred.hdf5"
                if not os.path.exists(l_path):
                    not_here +=1
                    print(f"{l_path} not found")
                else:
                    here +=1

print(here)
print(not_here)