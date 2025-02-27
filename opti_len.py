import numpy as np
# import matplotlib.py as plt
import os
import sys
root_path = "results_try_new_design"
scripts = sys.argv[1]
print(scripts)
# scripts = "s256"

if scripts == "l512":
# L512 comparison
    steps = 103999
    file_list = [
        "titok_l512_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.5+elbo_upper=0.5",
        "titok_l512_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0"
    ]
elif scripts == 'b512':
    # B512 comparison
    # '''
    # steps = 152999
    steps = 76999
    file_list = [
        "titok_b512_4096_12+nll_only=0.5+rate_weight=1+elbo_lower=0.5+elbo_upper=0.5",      # BASELINE
        # "elastic+titok_b512_4096_12",
        # "titok_b512_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=0.8",
        "titok_b512_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0",
        "titok_b512_4096_12+elbo_mode=0.4+0.6+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0",
        "titok_b512_4096_12+elbo_mode=0.1_in_0.5+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0",
        "titok_b512_4096_12+elbo_mode=0.1_in_px+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0",
    #     # "titok_b512_4096_12+elbo_mode=upto_px+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0",
    #     # "titok_b512_4096_12+elbo_mode=downto_px+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0",
    ]
# '''
elif scripts == 'b256':

# B256 comparison
# '''
    steps = 100999
    file_list = [
        "titok_b256_4096_12+nll_only=0.5+rate_weight=1+elbo_lower=0.5+elbo_upper=0.5",      # BASELINE
        "elastic+titok_b256_4096_12",
        "titok_b256_4096_12+rate_bug+nll_only=0.5+rate_weight=1+elbo_lower=0.2+elbo_upper=1.0",
        "titok_b256_4096_12+elbo_mode=0.1_in_0.5+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0",
    ]
# '''

elif scripts == 'l256':
# L256 comparison
    steps = 91999
    file_list = [
        "titok_l256_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.5+elbo_upper=0.5",
        "titok_l256_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0"
    ]

elif scripts == 's512':
    steps = 100999
    file_list = [
        "titok_s512_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.5+elbo_upper=0.5",
        "titok_s512_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0"
    ]

elif scripts == 's256':
    steps = 119999
    file_list = [
        "titok_s256_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.5+elbo_upper=0.5",
        "titok_s256_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.0+elbo_upper=1.0"
    ]
print(f"steps = {steps}")

baseline_rate = 0.5
baseline_recon = 0
mode="-train"

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
        

        for beta in np.arange(0, 4, 0.04):
            total = recon_error + beta * rate_error[None]
            indices = total.argmin(axis=1)
            rate_loss = 1 - indices / N
            selections = recon_error[np.arange(recon_error.shape[0]), 
                                    indices]
            # print(rate_loss.shape, selections.shape)
            if np.abs(rate_loss.mean() - baseline_rate) < 0.005 or \
                np.abs(selections.mean() - baseline_recon) < 0.01:
                print(f"beta={beta:1f}, rate_loss={rate_loss.mean():4f}, rate_std={rate_loss.std():4f}, recon={selections.mean():4f}")
