from align_image_code import align_images
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

# First load images

def read_image(path):
    img = Image.open(path)
    out = np.asarray(img, dtype=np.float64)
    out = out / out.max()
    return out

def gaussian2d(ksize=9, sigma=1.6):
    g1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    G = g1d @ g1d.T
    return G.astype(np.float64)

def gray(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b

img_pairs = [
    # ["../img/2.2/nutmeg.jpg", "../img/2.2/derek.jpg", 7, 7]
    # ["../img/2.2/1-dog.jpg", "../img/2.2/1-paul.jpg", 3, 8]
    ["../img/2.2/3-steph.jpg", "../img/2.2/1-paul.jpg", 3, 8]
    # ["../img/2.2/2-tern.jpg", "../img/2.2/2-weeknd.jpg", 3, 3]
]

for path1, path2, sigma1, sigma2 in img_pairs:
    
    # Next align images (this code is provided, but may be improved)
    img1_rgb, img2_rgb = align_images(read_image(path1), read_image(path2))

    img1 = gray(img1_rgb)
    img2 = gray(img2_rgb)

    s1 = max(int(3 * sigma1), 3)
    k1 = 2 * s1 + 1
    G1 = gaussian2d(k1, s1)

    s2 = max(int(3 * sigma2), 3)
    k2 = 2 * s2 + 1
    G2 = gaussian2d(k2, s2)

    img1_blur = signal.convolve2d(img1, G1, mode="same", boundary="symm")
    img1_highfreq = img1 - img1_blur
    img2_lowfreq = signal.convolve2d(img2, G2, mode="same", boundary="symm")

    hybrid = np.clip(img1_highfreq + img2_lowfreq, 0., 1.)

    # fig, axes = plt.subplots(2, 5, figsize=(16, 4))

    # axes[0, 0].imshow(img1, cmap="gray")
    # axes[0, 0].set_title(f"\"{path1}\"")

    # axes[0, 1].imshow(img2, cmap="gray")
    # axes[0, 1].set_title(f"\"{path2}\"")
    
    # axes[0, 4].imshow(hybrid, cmap="gray")
    # axes[0, 4].set_title("Hybrid image")

    # axes[1, 0].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img1)))), cmap="gray")
    # axes[1, 0].set_title(f"\"{path1}\"")

    # axes[1, 1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img2)))), cmap="gray")
    # axes[1, 1].set_title(f"\"{path2}\"")

    # axes[1, 2].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img1_highfreq)))), cmap="gray")
    # axes[1, 2].set_title(f"\"{path1}\" high frequency")

    # axes[1, 3].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img2_lowfreq)))), cmap="gray")
    # axes[1, 3].set_title(f"\"{path2}\" low frequency")

    # axes[1, 4].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid)))), cmap="gray")
    # axes[1, 4].set_title("Hybrid image")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(f"\"{path1}\"")

    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(f"\"{path2}\"")
    
    axes[2].imshow(hybrid, cmap="gray")
    axes[2].set_title("Hybrid image")

    for a in axes.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()

    ## Compute and display Gaussian and Laplacian Pyramids
    ## You also need to supply this function
    # N = 5 # suggested number of pyramid levels (your choice)
    # pyramids(hybrid, N)