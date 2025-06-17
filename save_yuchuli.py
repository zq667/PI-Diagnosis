from torch.utils.data import Dataset
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pandas as pd
import os
import shutil
import unicodedata
import pydicom
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import SimpleITK as sitk

matplotlib.rc("font", family='Noto Sans CJK JP')



#文件路径的列表
folder_path = 'yb_binguzuida'
folder_path2 = 'yb_jinggujiejie'
folder_path3 = 'yb_ruangufugai'
folder_path4 = 'yb_shizhuangwei'
img_paths = os.listdir(folder_path)
img_paths2 = os.listdir(folder_path2)
img_paths3 = os.listdir(folder_path3)
img_paths4 = os.listdir(folder_path4)


# 修改后的read_dicom2函数
def read_dicom3(sitk_image, target_spacing=None):
    """
    读取SimpleITK图像并将其转换为RGB图像。

    参数：
        sitk_image (SimpleITK.Image)：SimpleITK图像对象。
        target_spacing (tuple)：目标间距。默认为None，表示不进行间距调整。

    返回：
        PIL.Image.Image：RGB图像。
    """
    try:
        # 获取原始间距
        original_spacing = sitk_image.GetSpacing()

        # 将像素数组转换为numpy数组
        np_array = sitk.GetArrayFromImage(sitk_image)

        # 将像素数组转换为float类型
        new_image = np_array.astype(float)

        # 缩放图像
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0

        # 如果目标间距不为空，则调整图像尺寸
        if target_spacing is not None:
            scaling_factor = (original_spacing[0] / target_spacing[0],
                              original_spacing[1] / target_spacing[1])
            scaled_image = np.array(Image.fromarray(scaled_image).resize((int(new_image.shape[1] * scaling_factor[0]),
                                                                          int(new_image.shape[0] * scaling_factor[1]))))

        # 转换为uint8类型
        scaled_image = np.uint8(scaled_image)

        # 转换为PIL Image对象
        final_image = Image.fromarray(scaled_image)

        # 如果尚未转换为RGB，则进行转换
        final_image = final_image.convert('RGB')

        return final_image
    except Exception as e:
        print(f"发生错误：{e}")
        return None

def correct_bias_field(image):
    # 对MRI图像进行偏置场校正
    n4_bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # n4_bias_corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    # n4_bias_corrector.SetConvergenceThreshold(1e-6)
    # n4_bias_corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    corrected_image = n4_bias_corrector.Execute(image)
    return corrected_image

def process_dicom_with_bias_correction(folder_path, file_name, target_spacing=None):
    try:
        # 读取DICOM文件
        file_path = os.path.join(folder_path, file_name)
        ds = pydicom.dcmread(file_path)

        # 将DICOM像素数组转换为SimpleITK图像对象
        sitk_image = sitk.GetImageFromArray(ds.pixel_array)
        sitk_image.SetSpacing(ds.PixelSpacing)

        # 将将像素类型转换为Float32类型
        sitk_image =sitk.Cast(sitk_image,sitk.sitkFloat32)

        # 对DICOM图像进行偏置场校正
        corrected_image = correct_bias_field(sitk_image)

        # 调用read_dicom2函数处理校正后的图像
        rgb_image = read_dicom3(corrected_image, target_spacing=target_spacing)

        return rgb_image
    except Exception as e:
        print(f"发生错误：{e}")
        return None




def center_pad(image, target_width, target_height):
    """
    将图像居中放置到目标尺寸的图像中，并在需要时填充空白像素。

    参数：
        image (PIL.Image.Image)：要放置的图像。
        target_width (int)：目标宽度。
        target_height (int)：目标高度。

    返回：
        PIL.Image.Image：处理后的图像。
    """
    width, height = image.size
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    right = left + width
    bottom = top + height
    padded_image = Image.new(image.mode, (target_width, target_height), color='black')
    padded_image.paste(image, (left, top))
    return padded_image

def show_image(image):
    """
    显示图像。

    参数：
        image (PIL.Image.Image)：要显示的图像。
    """
    plt.imshow(image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 创建新文件夹
output_folder_path1 = 'yb_binguzuida_processed'
output_folder_path2 = 'yb_jinggujiejie_processed'
output_folder_path3 = 'yb_ruangufugai_processed'
output_folder_path4 = 'yb_shizhuangwei_processed'

print("正在创建文件夹...")

os.makedirs(output_folder_path1, exist_ok=True)
os.makedirs(output_folder_path2, exist_ok=True)
os.makedirs(output_folder_path3, exist_ok=True)
os.makedirs(output_folder_path4, exist_ok=True)

print("文件夹创建完成！")

# 定义函数来处理并保存图像
def process_and_save_images(input_folder_path, output_folder_path):
    # 获取文件夹内所有图像的文件名
    img_paths = os.listdir(input_folder_path)

    # 遍历文件夹内的每张图像
    for img_path in img_paths:
        # 处理图像
        print(f"正在处理图像: {img_path}")
        processed_image = center_pad(process_dicom_with_bias_correction(input_folder_path, img_path, [0.85, 0.85]), 224,224)

        # 保存处理后的图像到新文件夹
        output_img_path = os.path.join(output_folder_path, img_path)

        # 替换文件扩展名为 .png
        output_img_path = output_img_path.replace('.dcm', '.png')

        # 保存图像为PNG格式
        processed_image.save(output_img_path)

        print(f"图像 {img_path} 处理完成并保存为 {output_img_path}")


# 对每个文件夹执行处理和保存操作
print("开始处理图像...")
process_and_save_images(folder_path, output_folder_path1)
process_and_save_images(folder_path2, output_folder_path2)
process_and_save_images(folder_path3, output_folder_path3)
process_and_save_images(folder_path4, output_folder_path4)
print("所有图像处理完成！")