from atomai.utils import create_lattice_mask, extract_patches_
from sklearn.model_selection import train_test_split
from atomai.transforms import datatransform
import numpy as np
import matplotlib.pyplot as plt

print(" ")
print("##### MOS2 #####")

dataset4 = np.load('mos2_labels_imagem4.npz') # lê o arquivo .npz gerado ao final do notebook anterior
# essa estrutura de arquivo está relacionada ao armazenamento de múltiplos arrays na biblioteca Numpy

image4 = dataset4['image']
label4 = dataset4['labels']

print(f"INÍCIO: Shape (Image, label e label defects - Imagem 4): {image4.shape}, {label4.shape}")

dataset6 = np.load('mos2_labels_imagem6.npz') 

image6 = dataset6['image']
label6 = dataset6['labels']

print(f"INÍCIO: Shape (Image, label e label defects - Imagem 6): {image6.shape}, {label6.shape}")

for b in range(10):
    print(" ", flush=True)
    print(f"##### BATCH {b} #####", flush=True)

    images_all4, labels_all4 = extract_patches_(
        image4, label4, patch_size=512, num_patches=250, random_state=42+b)

    print("extract_patches - image4:", images_all4.shape, flush=True)
    print("extract_patches - label4:", labels_all4.shape, flush=True)


    images_all6, labels_all6 = extract_patches_(
        image6, label6, patch_size=512, num_patches=250, random_state=123+b)

    print("extract_patches - image6:", images_all6.shape, flush=True)
    print("extract_patches - label6:", labels_all6.shape, flush=True)

    # For a single class case, we still need to explicitly specify the single channel
    labels_all4 = labels_all4[..., None] if np.ndim(labels_all4) == 3 else labels_all4
    # Number of channels in masked data (the training images have a single channel)
    ch = labels_all4.shape[-1]
    # Define image distortion/noise parameters
    zoom = 1.1 # zoom factor
    poisson = [30, 40] # P noise range (scaled units)
    gauss = [20, 100] # G noise range (scaled units)
    blur = [1, 40] # Blurring range (scaled units)
    contrast = [5, 14] # contrast range (< 10 is brighter, > 10 is darker)
    salt_and_pepper = [1, 10] # min/max amount of salted/peppered pixels (scaled units)
    # Run the augmentor
    imaug = datatransform(
        n_channels=ch, dim_order_in='channel_last', dim_order_out='channel_first', 
        gauss_noise=gauss, poisson_noise=poisson, salt_and_pepper=salt_and_pepper,
        contrast=contrast, blur=blur, zoom=zoom, rotation=True,
        squeeze_channels=True, seed=42)
    print("Augmentor4 definido")
    images_all4, labels_all4 = imaug.run(images_all4, labels_all4)

    print(f"DATA AUGMENTATION: Shape (Image, label e label defects - Imagem 4): {images_all4.shape}, {label4.shape}")


    # For a single class case, we still need to explicitly specify the single channel
    labels_all6 = labels_all6[..., None] if np.ndim(labels_all6) == 3 else labels_all6
    # Number of channels in masked data (the training images have a single channel)
    ch = labels_all6.shape[-1]
    # Define image distortion/noise parameters
    zoom = 1.1 # zoom factor
    poisson = [30, 40] # P noise range (scaled units)
    gauss = [20, 100] # G noise range (scaled units)
    blur = [1, 40] # Blurring range (scaled units)
    contrast = [5, 14] # contrast range (< 10 is brighter, > 10 is darker)
    salt_and_pepper = [1, 10] # min/max amount of salted/peppered pixels (scaled units)
    # Run the augmentor
    imaug = datatransform(
        n_channels=ch, dim_order_in='channel_last', dim_order_out='channel_first', 
        gauss_noise=gauss, poisson_noise=poisson, salt_and_pepper=salt_and_pepper,
        contrast=contrast, blur=blur, zoom=zoom, rotation=True,
        squeeze_channels=True, seed=42)
    print("Augmentor6 definido")
    images_all6, labels_all6 = imaug.run(images_all6, labels_all6)

    print(f"DATA AUGMENTATION: Shape (Image, label e label defects - Imagem 6): {images_all6.shape}, {label6.shape}")

    n = 5

    n = n + 1
    fig = plt.figure( figsize=(30, 8))
    for i in range(1, n):   
        ax = fig.add_subplot(2, n, i)
        ax.imshow(images_all4[i-1].squeeze(), cmap='gray')
        ax.set_axis_off()
        ax.set_title('Image ' + str(i-1) )
        ax.grid(alpha = 0.5)
        ax = fig.add_subplot(2, n, i+n)
        ax.set_axis_off()
        if labels_all4.shape[1] == 1:
            ax.imshow(labels_all4[i-1, 0], cmap='jet', interpolation='Gaussian')
        else:
            ax.imshow(labels_all4[i-1], cmap='jet', interpolation='Gaussian')
        ax.set_title('Ground truth ' + str(i-1))
        ax.grid(alpha=0.75)
    plt.savefig(f'images/mos2/amostra4_mos2_{b}.png', dpi=900, bbox_inches='tight')

    n = 5

    n = n + 1
    fig = plt.figure( figsize=(30, 8))
    for i in range(1, n):   
        ax = fig.add_subplot(2, n, i)
        ax.imshow(images_all6[i-1].squeeze(), cmap='gray')
        ax.set_axis_off()
        ax.set_title('Image ' + str(i-1) )
        ax.grid(alpha = 0.5)
        ax = fig.add_subplot(2, n, i+n)
        ax.set_axis_off()
        if labels_all6.shape[1] == 1:
            ax.imshow(labels_all6[i-1, 0], cmap='jet', interpolation='Gaussian')
        else:
            ax.imshow(labels_all6[i-1], cmap='jet', interpolation='Gaussian')
        ax.set_title('Ground truth ' + str(i-1))
        ax.grid(alpha=0.75)
    plt.savefig(f'images/mos2/amostra6_mos2_{b}.png', dpi=900, bbox_inches='tight')

    print("Imagens de Amostra Salvas!")

    images_all4, images_test_all4, labels_all4, labels_test_all4 = train_test_split(
        images_all4, labels_all4, test_size=0.2, random_state=42)

    images_all6, images_test_all6, labels_all6, labels_test_all6 = train_test_split(
        images_all6, labels_all6, test_size=0.2, random_state=42)

    images_all = np.concatenate((images_all4, images_all6), axis=0)
    labels_all = np.concatenate((labels_all4, labels_all6), axis=0)

    images_test_all = np.concatenate((images_test_all4, images_test_all6), axis=0)
    labels_test_all = np.concatenate((labels_test_all4, labels_test_all6), axis=0)

    np.savez(f'datasets/mos2/mos2_dataset_{b}.npz', images= images_all, labels = labels_all)
    np.savez(f'datasets/mos2/mos2_test_{b}.npz', images= images_test_all, labels = labels_test_all)

    print("--- Datasets Salvos! ---")
