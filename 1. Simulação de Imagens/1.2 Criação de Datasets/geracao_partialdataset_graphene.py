from atomai.utils import create_lattice_mask, extract_patches_
from sklearn.model_selection import train_test_split
from atomai.transforms import datatransform
import numpy as np
import matplotlib.pyplot as plt

dataset = np.load('graphene_labels.npz') # lê o arquivo .npz gerado ao final do notebook anterior
# essa estrutura de arquivo está relacionada ao armazenamento de múltiplos arrays na biblioteca Numpy

image = dataset['image']
label = dataset['labels']

print(f"INÍCIO: Shape (Image, label e label defects): {image.shape}, {label.shape}", flush=True)

for b in range(10):
    print(" ", flush=True)
    print(f"##### BATCH {b} #####", flush=True)
    images_all, labels_all = extract_patches_(
        image, label, patch_size=512, num_patches=500, random_state=42+b)

    print("extract_patches - images:", images_all.shape, flush=True)
    print("extract_patches - labels:", labels_all.shape, flush=True)


    # For a single class case, we still need to explicitly specify the single channel
    labels_all = labels_all[..., None] if np.ndim(labels_all) == 3 else labels_all
    # Number of channels in masked data (the training images have a single channel)
    ch = labels_all.shape[-1]
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
    print("Augmentor definido", flush=True)
    images_all, labels_all = imaug.run(images_all, labels_all)

    print(f"DATA AUGMENTATION: Shape (Image, label e label defects): {images_all.shape}, {label.shape}", flush=True)


    n = 5

    n = n + 1
    fig = plt.figure( figsize=(30, 8))
    for i in range(1, n):   
        ax = fig.add_subplot(2, n, i)
        ax.imshow(images_all[i-1].squeeze(), cmap='gray')
        ax.set_axis_off()
        ax.set_title('Image ' + str(i-1) )
        ax.grid(alpha = 0.5)
        ax = fig.add_subplot(2, n, i+n)
        ax.set_axis_off()
        if labels_all.shape[1] == 1:
            ax.imshow(labels_all[i-1, 0], cmap='jet', interpolation='Gaussian')
        else:
            ax.imshow(labels_all[i-1], cmap='jet', interpolation='Gaussian')
        ax.set_title('Ground truth ' + str(i-1))
        ax.grid(alpha=0.75)
    plt.savefig(f'images/graphene/amostra_graphene{b}.png', dpi=900, bbox_inches='tight')

    print("Imagens de amostra salvas!", flush=True)

    images_all, images_test_all, labels_all, labels_test_all = train_test_split(
        images_all, labels_all, test_size=0.2, random_state=42)


    np.savez(f'datasets/graphene/graphene_dataset{b}.npz', images= images_all, labels = labels_all)
    np.savez(f'datasets/graphene/graphene_test{b}.npz', images= images_test_all, labels = labels_test_all)

    print("--- Datasets salvos! ---", flush=True)

