from PIL import Image
import os
import matplotlib.pyplot as plt
from typing import Iterator
import numpy as np
import jax.numpy as jnp
import random
from diffusion import forward_diffusion
import random

WORKING_DIR="./"

def load_image(image_path, size=None):
    """加载图片，调整尺寸，并转换为numpy数组
             [0, 255] -> [-1, 1]
    """
    with Image.open(image_path) as img:
        if size:
            img = img.resize(size)
        if random.random() > 0.5:  # 50% 概率应用翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:  # 50% 概率应用旋转
            img = img.rotate(random.choice([90, 180, 270]), expand=True)

        # print('np.array(img).max(), np.array(img).min()', np.array(img).max(), np.array(img).min()) 
        img_array = (np.array(img)/255.0 - 0.5)*2
        return img_array

def test_load_image():
    img_array = load_image(f'{WORKING_DIR}/train_set/6953297_8576bf4ea3.jpg')
    print('np.array(img).max(), np.array(img).min()', img_array.max(), img_array.min()) 

test_load_image()
# def load_images(directory_path, size=(32, 32)):
#     """加载目录中所有图片并转换为具有统一尺寸的numpy数组的列表"""
#     image_arrays = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据需要添加其他图片格式
#             file_path = os.path.join(directory_path, filename)
#             image_arrays.append(load_image(file_path, size=size))
#     return np.stack(image_arrays)  # 使用 np.stack 确保能够正确地形成一个多维数组

def image_generator(filenames, size=(32, 32), batch_size=20, num_files_limit=None) -> Iterator[jnp.ndarray]:
    # filenames = [f for f in os.listdir(directory_path) if f.endswith('.jpg') or f.endswith('.png')]
    num_files = len(filenames)
    if num_files_limit:
        num_files = num_files_limit
        filenames = filenames[:num_files]

    def finite_generator():
        for i in range(0, num_files, batch_size):
            batch_filenames = filenames[i:i + batch_size]
            batch_images = [load_image(f'{WORKING_DIR}/train_set/{f}') for f in batch_filenames]
            yield jnp.stack(batch_images)
    return finite_generator()

def inverse_transform(image):
    """Convert tensors from [-1., 1.] to [0., 255.] and ensure type uint8 for correct image display."""
    image = ((jnp.clip(image, -1, 1) + 1.0) / 2.0) * 255.0
    return image.astype(jnp.uint8)  # Convert to uint8 to match expected [0, 255] integer range
    # image = ((jnp.clip(image, -1, 1) + 1.0) / 2.0) 
    # # print(image)
    # return image  # Convert to uint8 to match expected [0, 255] integer range

def normalize_image(image):
    print('image.max(), image.min()', image.max(), image.min())
    normalized_image = (image - image.min()) / (image.max() - image.min())
    print('normalized_image.max(), normalized_image.min()', normalized_image.max(), normalized_image.min())
    print()
    return normalized_image



def visualize_images(images):
    num_images = images.shape[0]
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(normalize_image(images[i]))
        # ax.imshow(inverse_transform(images[i]))
        ax.axis('off')
    # plt.show()
    plt.savefig('inferred_images.png',dpi=300)

# 使用上面的函数可视化图片
# visualize_images(dataset)

def show_vague_images():
    
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]
    example_dataset = image_generator('./train_set', batch_size=5)
    flattened_dataset = next(example_dataset)
    # num_pictures = flattened_dataset.shape[0]
    print('flattened_dataset.shape', flattened_dataset.shape)

    noisy_images = []
    for specific_timestep in specific_timesteps:
        xt, _ = forward_diffusion(sd=sd, x0=flattened_dataset, timestep=specific_timestep)
        
        noisy_images.append(inverse_transform(xt)) 

    # Plot and see samples at different timesteps
    NUM_ROWS = min(5, len(flattened_dataset))  # Ensure we do not exceed the number of available images
    fig, axes = plt.subplots(NUM_ROWS, len(specific_timesteps), figsize=(12, NUM_ROWS))  # Each image is 3 inches wide

    for i in range(NUM_ROWS):
        for j in range(len(specific_timesteps)):
            img = noisy_images[j][i,:,:,:].astype(np.uint8)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

    plt.tight_layout()  # Adjust layout
    plt.show()
# show_vague_images()