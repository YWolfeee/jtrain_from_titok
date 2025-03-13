import numpy as np
# import matplotlib.py as plt
import os
import sys
root_path = "results_finetune"
scripts = sys.argv[1]
p_mean = sys.argv[2]    
print(scripts, p_mean)
# scripts = "l512"
# p_mean = 0.25
# if p_mean == 0.25:
# L512 comparison
# steps = 0
steps = 240999
file_list = [
    f"titok_b512_4096_12+method=ours+p_mean={p_mean}+anneal=False+finetune=True",
    f"titok_b512_4096_12+method=ours+p_mean={p_mean}+anneal=True+finetune=True",
    f"titok_b512_4096_12+method=ours+p_mean={p_mean}+anneal=False+finetune=False",
    f"titok_b512_4096_12+method=ours+p_mean={p_mean}+anneal=True+finetune=False",
]

print(f"steps = {steps}")

baseline_rate = float(p_mean)
baseline_recon = 1.871828
mode="-eval"

for name in file_list:
    print("*" * 16)
    print(name)

    try:
        recon_error = np.load(f"./{root_path}/{name}/recon_matrix/recon_matrix-{steps}{mode}.npy")
    except Exception as e:
        if mode == "-train":
            recon_error = np.load(f"./{root_path}/{name}/recon_matrix/recon_matrix-{steps}.npy")
        else:
            raise e
    
    print(recon_error.shape)

    if os.path.exists(f"./{root_path}/{name}/policy_recon_arr/policy_recon_arr-{steps}{mode}.npy"):
        policy_arr = np.load(f"./{root_path}/{name}/policy_recon_arr/policy_recon_arr-{steps}{mode}.npy")
    else:
        policy_arr = None
    # recon_error = np.load(f"./temp/{name}/recon_matrix/recon_matrix-{steps}{mode}.npy")
    # recon_error = np.load("./results_try_new_design/titok_b512_4096_12+rate_bug+nll_only=0.5+rate_weight=1+elbo_lower=0.2+elbo_upper=1.0/recon_matrix/recon_matrix-74499.npy")
    # recon_error = np.load("./results_try_new_design/titok_b512_4096_12+nll_only=0.5+rate_weight=1+elbo_lower=0.5+elbo_upper=0.5/recon_matrix/recon_matrix-34499.npy")

    # recon_error = np.load("./results_try_new_design/titok_b512_4096_12_no_crop+rate_bug+nll_only=0.5+rate_weight=1+elbo_lower=0.2+elbo_upper=1.0/recon_matrix/recon_matrix-59999.npy")
    # recon_error = np.load("./results_try_new_design/elastic+titok_b256_4096_12/recon_matrix/recon_matrix-84999.npy")
    N = recon_error.shape[1]
    rate_error = 1 - np.arange(N) / N

    if "elbo_lower=0.5+elbo_upper=0.5" in name: # only take 0.5
        selections = recon_error[:, N // 2]
        rate_loss = 0.5 * np.ones_like(selections) 

        baseline_recon = selections.mean()
        baseline_rate = rate_loss.mean()

        print(f"beta={0.0:1f}, rate_loss={rate_loss.mean():4f}, rate_std={rate_loss.std():4f}, recon={selections.mean():4f}")
    else:
        if policy_arr is not None:
            print("policy recon error", policy_arr.shape, policy_arr.mean())

        # indices = np.ones_like(recon_error[:, 0], dtype=np.int32) * N // 2
        # rate_loss = 1 - indices / N
        # selections = recon_error[np.arange(recon_error.shape[0]), 
        #                             indices]
        # print(f"beta={0.0:1f}, rate_loss={rate_loss.mean():4f}, rate_std={rate_loss.std():4f}, recon={selections.mean():4f}")
        

        for beta in np.arange(0, 10, 0.1):
            total = recon_error + beta * rate_error[None]
            indices = total.argmin(axis=1)
            rate_loss = 1 - indices / N
            selections = recon_error[np.arange(recon_error.shape[0]), 
                                    indices]
            # print(rate_loss.shape, selections.shape)
            if np.abs(rate_loss.mean() - baseline_rate) < 0.005 or \
                np.abs(selections.mean() - baseline_recon) < 0.01:
                print(f"beta={beta:1f}, rate_loss={rate_loss.mean():4f}, rate_std={rate_loss.std():4f}, recon={selections.mean():4f}")
