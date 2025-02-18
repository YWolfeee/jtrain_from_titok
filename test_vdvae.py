import os
import numpy as np
import matplotlib.pyplot as plt
from vdvae.hps import Hyperparams
from vdvae.vae import VAE
from PIL import Image
import torch
import torchvision.transforms as transforms
from vdvae.data import set_up_imagenet64_preprocess_func

def init_vae_settings():
    cifar10 = Hyperparams()
    cifar10.width = 384
    cifar10.lr = 0.0002
    cifar10.zdim = 16
    cifar10.wd = 0.01
    cifar10.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
    cifar10.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
    cifar10.warmup_iters = 100
    cifar10.dataset = 'cifar10'
    cifar10.n_batch = 16
    cifar10.ema_rate = 0.9999

    i32 = Hyperparams()
    i32.update(cifar10)
    i32.dataset = 'imagenet32'
    i32.ema_rate = 0.999
    i32.dec_blocks = "1x2,4m1,4x4,8m4,8x9,16m8,16x19,32m16,32x40"
    i32.enc_blocks = "32x15,32d2,16x9,16d2,8x8,8d2,4x6,4d4,1x6"
    i32.width = 512
    i32.n_batch = 8
    i32.lr = 0.00015
    i32.grad_clip = 200.
    i32.skip_threshold = 300.
    i32.epochs_per_eval = 1
    i32.epochs_per_eval_save = 1

    i64 = Hyperparams()
    i64.update(i32)
    i64.n_batch = 4
    i64.grad_clip = 220.0
    i64.skip_threshold = 380.0
    i64.dataset = 'imagenet64'
    i64.dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12"
    i64.enc_blocks = "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5"
    
    i64.bottleneck_multiple = 0.25
    i64.no_bias_above = 64
    i64.num_mixtures = 10
    return i64

H = init_vae_settings()
H, preprocess_fn = set_up_imagenet64_preprocess_func(H)
vae = VAE(H)
state_dict = torch.load('imagenet64-iter-1600000-model.th', map_location='cpu')
new_state_dict = {}
l = len('module.')
for k in state_dict:
    if k.startswith('module.'):
        new_state_dict[k[l:]] = state_dict[k]
    else:
        new_state_dict[k] = state_dict[k]
state_dict = new_state_dict
vae.load_state_dict(state_dict)

# Create temp_results directory if it doesn't exist
os.makedirs('temp_results', exist_ok=True)

# Create figure with 8 subplots in one row
plt.figure(figsize=(32, 4))

for i in range(8):
    img = Image.open(f'vdvae/test_imgs/test{i}.png').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64
        transforms.ToTensor(),  # Convert PIL image to tensor and scale to [0,1]
        transforms.Lambda(lambda x: x * 255)  # Scale back to [0,255] range
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    img = img.permute(0, 2, 3, 1).contiguous() # convert back to (B, H, W, C)

    # preprocess the image
    inp, out = preprocess_fn(img)
    # inp, out shape: (1, 64, 64, 3); mean = 0, std = 1
    vae = vae.cuda()
    stats = vae.forward(inp, out)

    for key in stats:
        stats[key] = stats[key].item()

    # Plot in the corresponding subplot position
    plt.subplot(1, 8, i+1)
    plt.imshow(img[0].cpu().numpy().astype(np.uint8))
    
    # Add stats text above the image
    stats_text = '\n'.join([f'{k}: {v:.2f}' for k,v in stats.items()])
    plt.title(stats_text)

# Save the complete figure with all images
plt.savefig('temp_results/all_tests_with_stats.png')
plt.close()
