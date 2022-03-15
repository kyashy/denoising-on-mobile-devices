import numpy as np
import matplotlib.pyplot as plt
import cv2
import rawpy

def noise_model(input_array, k, sigma2):
    output = k * np.random.poisson(input_array / k) + np.random.normal(0, np.sqrt(sigma2), input_array.shape)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

path = './images/10106_00_30s.arw'
raw = rawpy.imread(path)
raw_post = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8)

k = 30         # config
sigma2 = 30    # config

# オリジナル
original_image = cv2.cvtColor(raw_post, cv2.COLOR_BGR2RGB)
cv2.imwrite('./results/original_image.png', original_image)

# ノイズ付加
noisy_image = noise_model(raw_post, k, sigma2)
noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
cv2.imwrite('./results/noisy_image.png', noisy_image)
