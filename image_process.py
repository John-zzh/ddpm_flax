from PIL import Image
import os
import matplotlib.pyplot as plt
from typing import Iterator
import numpy as np
import jax.numpy as jnp
import random

def load_image(image_path, size=None):
    """加载图片，调整尺寸，并转换为numpy数组
             [0, 255] -> [-1, 1]
    """
    with Image.open(image_path) as img:
        if size:
            img = img.resize(size)
        img_array = (np.array(img)/255.0 - 0.5)*2
        return img_array

# def load_images(directory_path, size=(32, 32)):
#     """加载目录中所有图片并转换为具有统一尺寸的numpy数组的列表"""
#     image_arrays = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据需要添加其他图片格式
#             file_path = os.path.join(directory_path, filename)
#             image_arrays.append(load_image(file_path, size=size))
#     return np.stack(image_arrays)  # 使用 np.stack 确保能够正确地形成一个多维数组

def image_generator(directory_path, size=(32, 32), batch_size=20, num_files_limit=None) -> Iterator[jnp.ndarray]:
    filenames = [f for f in os.listdir(directory_path) if f.endswith('.jpg') or f.endswith('.png')]
    num_files = len(filenames)
    if num_files_limit:
        num_files = num_files_limit
        filenames = filenames[:num_files]

    def finite_generator():
        for _ in range(1000):
            random.shuffle(filenames)  # 打乱文件名的顺序
            for i in range(0, num_files, batch_size):
                batch_filenames = filenames[i:i + batch_size]
                batch_images = [load_image(os.path.join(directory_path, f), size=None) for f in batch_filenames]
                yield jnp.stack(batch_images)
    return finite_generator()

def inverse_transform(image):
    """Convert tensors from [-1., 1.] to [0., 255.] """
    return ((jnp.clip(image, -1, 1) + 1.0) / 2.0) * 255.0

def visualize_images(images):
    """可视化数据集中的图片"""
    plt.figure(figsize=(10, 3))
    for i in range(images.shape[0]):
        ax = plt.subplot(3, 10, i + 1)
        img = inverse_transform(images[i,:,:,:]).astype(np.uint8)
        plt.imshow(img)
        plt.axis("off")
    # plt.show()
    plt.savefig('inferred_images.png',dpi=300)

# 使用上面的函数可视化图片
# visualize_images(dataset)

def show_vague_images():
    
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]
    example_dataset = image_generator(directory_path, batch_size=5)
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