from torch.utils.data import Dataset
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pandas as pd
import os
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
from sklearn.metrics import accuracy_score, roc_auc_score
import SimpleITK as sitk
from Transformer import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import json

matplotlib.rc("font", family='Noto Sans CJK JP')




#读取excel表
df = pd.read_excel('biaoqian_xin.xlsx',header=None)
column_data = df.iloc[:, 0].copy()
for i in range(len(column_data)):
    column_data[i] = unicodedata.normalize("NFKD", column_data[i])
df.iloc[:, 0] = column_data




#文件路径的列表
folder_path = 'binguzuida_new_processed'
folder_path2 = 'jinggujiejie_new_processed'
folder_path3 = 'ruangufugai_new_processed'
folder_path4 = 'shizhuangwei_new_processed'
img_paths = os.listdir(folder_path)
img_paths2 = os.listdir(folder_path2)
img_paths3 = os.listdir(folder_path3)
img_paths4 = os.listdir(folder_path4)



#通过文件路径获取文件名字
def getname(origin_name):
    origin_name_normal = unicodedata.normalize("NFKD", origin_name)
    a = origin_name_normal.split('.')
    b = a[0].split('-')
    c = b[:len(b)-2]
    d = ''
    for i in range(len(c)):
        if i < len(c)-1:
            d = d + c[i] + '-'
        if i == len(c)-1:
            d = d + c[i]
    return d

#通过文件路径获取文件的序列类型
def getname2(origin_name):
    origin_name_normal = unicodedata.normalize("NFKD", origin_name)
    a = origin_name_normal.split('.')
    b = a[0].split('-')
    c = b[-2:]
    if len(c) > 2:
        print("name_wrong")
    d = c[0]+'-'+c[1]
    return d

#选择指定的序列组合
def xulie_liter(img_paths,img_paths2,img_paths3,img_paths4):
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    img_paths =sorted(img_paths)
    img_paths2 = sorted(img_paths2)
    img_paths3 = sorted(img_paths3)
    img_paths4 = sorted(img_paths4)
    for i in range(len(img_paths)):
        a = getname2(img_paths[i])
        b = getname2(img_paths2[i])
        c = getname2(img_paths3[i])
        d = getname2(img_paths4[i])
        if a == b == c == d == 'PD-SPAIR':
            p1.append(img_paths[i])
            p2.append(img_paths2[i])
            p3.append(img_paths3[i])
            p4.append(img_paths4[i])
    return p1,p2,p3,p4

# img_paths,img_paths2,img_paths3,img_paths4 = xulie_liter(img_paths,img_paths2,img_paths3,img_paths4)

# print("=======================")
# print(len(img_paths))
# print(len(img_paths2))
# print(len(img_paths3))
# print(len(img_paths4))
# print(img_paths)
# print(img_paths2)
# print(img_paths3)
# print(img_paths4)
# print("=======================")

#千万注意要修改列数
#通过文件名获取标签
def getlabel(name):
    goal = df.loc[df[0] == name]
    label = goal.iloc[0,1]
    return int(label)



#分隔文件路径的列表
img_paths_health = []
img_paths_sick = []

for i in img_paths:
    if getlabel(getname(i)) == 0:
        img_paths_health.append(i)
    if getlabel(getname(i)) == 1:
        img_paths_sick.append(i)

# print(len(img_paths_health))
# print(len(img_paths_sick))

img_paths_health_train_val,img_paths_health_test  = train_test_split(img_paths_health, test_size=0.2, random_state=42)
img_paths_health_train,img_paths_health_val  = train_test_split(img_paths_health_train_val, test_size=0.25, random_state=42)

img_paths_sick_train_val,img_paths_sick_test  = train_test_split(img_paths_sick, test_size=0.2, random_state=42)
img_paths_sick_train,img_paths_sick_val  = train_test_split(img_paths_sick_train_val, test_size=0.25, random_state=42)

img_paths_train = img_paths_health_train + img_paths_sick_train
img_paths_val = img_paths_health_val + img_paths_sick_val
img_paths_test = img_paths_health_test + img_paths_sick_test




#解决四个文件名字编号相同但是尾部不同的问题
def find_real_file_name(paths,file_name):
    real_file_name = 'abc'
    for i in paths:
        if getname(i) == getname(file_name) :
            real_file_name = i
    return  real_file_name

def add_elements(paths1,paths2):
    arr = []
    for i in paths1:
        real_name = find_real_file_name(paths2,i)
        arr.append(real_name)
    return arr

img_paths_train2 = add_elements(img_paths_train,img_paths2)
img_paths_train3 = add_elements(img_paths_train,img_paths3)
img_paths_train4 = add_elements(img_paths_train,img_paths4)

img_paths_val2 = add_elements(img_paths_val,img_paths2)
img_paths_val3 = add_elements(img_paths_val,img_paths3)
img_paths_val4 = add_elements(img_paths_val,img_paths4)

img_paths_test2 = add_elements(img_paths_test,img_paths2)
img_paths_test3 = add_elements(img_paths_test,img_paths3)
img_paths_test4 = add_elements(img_paths_test,img_paths4)


#读取dicom图片
def read_dicom(folder_path,file_name):
    file_path = os.path.join(folder_path, file_name)
    ds = pydicom.dcmread(file_path)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    final_image = final_image.convert('RGB')
    return final_image

def read_dicom2(folder_path, file_name, target_spacing=None):
    """
    读取DICOM文件并将其转换为RGB图像。

    参数：
        folder_path (str)：包含DICOM文件的文件夹路径。
        file_name (str)：DICOM文件的名称。
        target_spacing (tuple)：目标间距。默认为None，表示不进行间距调整。

    返回：
        PIL.Image.Image：RGB图像。
    """
    try:
        file_path = os.path.join(folder_path, file_name)

        # 读取DICOM文件
        ds = pydicom.dcmread(file_path)
        # print(ds.PixelSpacing)

        # 获取原始间距
        original_spacing = (ds.PixelSpacing[0], ds.PixelSpacing[1])

        # 将像素数组转换为float类型
        new_image = ds.pixel_array.astype(float)

        # 缩放图像
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0

        # 如果目标间距不为空，则调整图像尺寸
        if target_spacing is not None:
            scaling_factor = (original_spacing[0] / target_spacing[0],
                              original_spacing[1] / target_spacing[1])
            scaled_image = np.array(Image.fromarray(scaled_image).resize((int(new_image.shape[1] * scaling_factor[0]),
                                                                          int(new_image.shape[0] * scaling_factor[1]))))
            # 更新PixelSpacing
            ds.PixelSpacing = target_spacing

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





#读取预处理后的图像
def read_image(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    image = Image.open(file_path)
    return image



#准备数据集
train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

val_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

test_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

class MyData(Dataset):
    def __init__(self,img_path,img_path2,img_path3,img_path4,transform):
        self.img_path = img_path
        self.img_path2 = img_path2
        self.img_path3 = img_path3
        self.img_path4 = img_path4
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_name2 = self.img_path2[idx]
        img_name3 = self.img_path3[idx]
        img_name4 = self.img_path4[idx]
        img1 = self.transform(read_image(folder_path,img_name))
        img2 = self.transform(read_image(folder_path2,img_name2))
        img3 = self.transform(read_image(folder_path3,img_name3))
        img4 = self.transform(read_image(folder_path4,img_name4))
        label = getlabel(getname(img_name))
        return img1,img2,img3,img4,label

    def __len__(self):
        return len(self.img_path)


train_dataset = MyData(img_paths_train,img_paths_train2,img_paths_train3,img_paths_train4,train_transform)
val_dataset = MyData(img_paths_val,img_paths_val2,img_paths_val3,img_paths_val4,val_transform)
test_dataset = MyData(img_paths_test,img_paths_test2,img_paths_test3,img_paths_test4,test_transform)




# length 长度
train_data_size = len(train_dataset)
val_data_size = len(val_dataset)
test_data_size = len(test_dataset)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("验证数据集的长度为：{}".format(val_data_size))
print("测试数据集的长度为：{}".format(test_data_size))



# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=16,shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16,shuffle=True)


class FusionNet(nn.Module):
    def __init__(self, vgg1, vgg2, vgg3, vgg4):
        super(FusionNet, self).__init__()
        self.vgg1 = vgg1
        self.vgg2 = vgg2
        self.vgg3 = vgg3
        self.vgg4 = vgg4

        self.classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )

    def forward(self, x1, x2, x3, x4):
        feature1 = self.vgg1(x1)
        feature2 = self.vgg2(x2)
        feature3 = self.vgg3(x3)
        feature4 = self.vgg4(x4)
        # print(feature1.shape)

        merged_feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)  # 在通道维度上拼接特征
        # print(merged_feature.shape)

        merged_feature = merged_feature.view(merged_feature.size(0), -1)
        # print(merged_feature.shape)

        output = self.classifier(merged_feature)
        return output


class FusionNet_and_Duo_ren_wu_Net(nn.Module):
    def __init__(self, vgg1, vgg2, vgg3, vgg4):
        super(FusionNet_and_Duo_ren_wu_Net, self).__init__()
        self.vgg1 = vgg1
        self.vgg2 = vgg2
        self.vgg3 = vgg3
        self.vgg4 = vgg4

        self.classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )

    def forward(self, x1, x2, x3, x4):
        feature1 = self.vgg1(x1)
        feature2 = self.vgg2(x2)
        feature3 = self.vgg3(x3)
        feature4 = self.vgg4(x4)
        # print(feature1.shape)

        merged_feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)  # 在通道维度上拼接特征
        # print(merged_feature.shape)

        merged_feature = merged_feature.view(merged_feature.size(0), -1)
        # print(merged_feature.shape)

        output = self.classifier(merged_feature)
        return output



model1 = torch.load('direct_classification_val.pth')
model2 = torch.load('direct_classification+duo_ren_wu_val.pth')

model_list = [model1,model2]


# 定义 bootstrap 计算置信区间的通用函数
def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, random_state=None):
    """
    使用 bootstrap 法计算指标的 95% 置信区间。

    参数:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测值
        metric_fn (function): 用于计算指标的函数
        n_bootstraps (int): 重采样次数
        random_state (int): 随机种子

    返回:
        (float, float): 置信区间的下界和上界
    """
    rng = np.random.default_rng(random_state)
    metrics = []
    for _ in range(n_bootstraps):
        # 有放回随机抽样
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]
        # 计算指标
        metric = metric_fn(bootstrap_true, bootstrap_pred)
        metrics.append(metric)
    metrics = np.array(metrics)
    # 计算 95% 置信区间
    lower_ci = np.percentile(metrics, 2.5)
    upper_ci = np.percentile(metrics, 97.5)
    return lower_ci, upper_ci


# 定义敏感度和特异度的函数
def sensitivity_score(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def specificity_score(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    return TN / (TN + FP) if (TN + FP) > 0 else 0.0


# 模型评估部分
for model in model_list:
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 利用 GPU 训练
    if torch.cuda.is_available():
        model.to('cuda')

    model.eval()
    with torch.no_grad():
        all_preds_test = []
        all_labels_test = []
        running_loss_test = 0

        # 遍历测试数据
        for x1, x2, x3, x4, label in tqdm(test_dataloader):
            x1, x2, x3, x4, label = x1.to('cuda'), x2.to('cuda'), x3.to('cuda'), x4.to('cuda'), label.to('cuda')
            output = model(x1, x2, x3, x4)
            loss = criterion(output, label)
            running_loss_test += loss.item()

            all_preds_test.append(output)
            all_labels_test.append(label)

        epoch_loss_test = running_loss_test / len(test_dataloader.dataset)
        all_preds_test = torch.cat(all_preds_test, dim=0)
        all_labels_test = torch.cat(all_labels_test, dim=0)

        # 转为 NumPy 格式
        y_true = all_labels_test.cpu().numpy()
        y_pred = torch.argmax(all_preds_test, dim=1).cpu().numpy()

        # 准确率及其置信区间
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_ci = bootstrap_metric_ci(y_true, y_pred, accuracy_score)

        # F1 分数及其置信区间
        f1 = f1_score(y_true, y_pred)
        f1_ci = bootstrap_metric_ci(y_true, y_pred, f1_score)

        # 敏感度及其置信区间
        sensitivity = sensitivity_score(y_true, y_pred)
        sensitivity_ci = bootstrap_metric_ci(y_true, y_pred, sensitivity_score)

        # 特异度及其置信区间
        specificity = specificity_score(y_true, y_pred)
        specificity_ci = bootstrap_metric_ci(y_true, y_pred, specificity_score)

        # 计算混淆矩阵和 AUC
        conf_matrix = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, all_preds_test[:, 1].cpu().detach().numpy())
        auc_ci_lower, auc_ci_upper = bootstrap_metric_ci(
            y_true, all_preds_test[:, 1].cpu().detach().numpy(), roc_auc_score
        )

        # 输出结果
        print('Test Loss: ', round(epoch_loss_test, 3))
        print(f'Accuracy: {round(accuracy, 3)} (95% CI: [{round(accuracy_ci[0], 3)} - {round(accuracy_ci[1], 3)}])')
        print(f'F1 Score: {round(f1, 3)} (95% CI: [{round(f1_ci[0], 3)} - {round(f1_ci[1], 3)}])')
        print(
            f'Sensitivity: {round(sensitivity, 3)} (95% CI: [{round(sensitivity_ci[0], 3)} - {round(sensitivity_ci[1], 3)}])')
        print(
            f'Specificity: {round(specificity, 3)} (95% CI: [{round(specificity_ci[0], 3)} - {round(specificity_ci[1], 3)}])')
        print(f'AUC: {round(auc, 3)} (95% CI: [{round(auc_ci_lower, 3)} - {round(auc_ci_upper, 3)}])')
        print('-' * 50)

