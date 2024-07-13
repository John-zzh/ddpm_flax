import os
from PIL import Image
import numpy as np

def resize_and_save_images(input_directory, output_directory, size=(32, 32)):
    """调整目录中所有图片的尺寸并保存到新目录中"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            
            with Image.open(input_path) as img:
                img_resized = img.resize(size)
                img_resized.save(output_path)

# 使用示例
resize_and_save_images('/home/jowsl/ddpm/archive/flowers/sunflower', '/home/jowsl/ddpm/ddpm_flax/train_set')
