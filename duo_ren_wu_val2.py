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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

matplotlib.rc("font", family='Noto Sans CJK JP')

# 读取excel表
df = pd.read_excel('biaoqian_xin.xlsx', header=None)
column_data = df.iloc[:, 0].copy()
for i in range(len(column_data)):
    column_data[i] = unicodedata.normalize("NFKD", column_data[i])
df.iloc[:, 0] = column_data

# 文件路径的列表
folder_path = 'binguzuida_new_processed'
folder_path2 = 'jinggujiejie_new_processed'
folder_path3 = 'ruangufugai_new_processed'
folder_path4 = 'shizhuangwei_new_processed'
img_paths = os.listdir(folder_path)
img_paths2 = os.listdir(folder_path2)
img_paths3 = os.listdir(folder_path3)
img_paths4 = os.listdir(folder_path4)


# 通过文件路径获取文件名字
def getname(origin_name):
    origin_name_normal = unicodedata.normalize("NFKD", origin_name)
    a = origin_name_normal.split('.')
    b = a[0].split('-')
    c = b[:len(b) - 2]
    d = ''
    for i in range(len(c)):
        if i < len(c) - 1:
            d = d + c[i] + '-'
        if i == len(c) - 1:
            d = d + c[i]
    return d


# 通过文件路径获取文件的序列类型
def getname2(origin_name):
    origin_name_normal = unicodedata.normalize("NFKD", origin_name)
    a = origin_name_normal.split('.')
    b = a[0].split('-')
    c = b[-2:]
    if len(c) > 2:
        print("name_wrong")
    d = c[0] + '-' + c[1]
    return d


# 选择指定的序列组合
def xulie_liter(img_paths, img_paths2, img_paths3, img_paths4):
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    img_paths = sorted(img_paths)
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
    return p1, p2, p3, p4


# 筛选固定的序列，不用时注释下面一行代码
# img_paths,img_paths2,img_paths3,img_paths4 = xulie_liter(img_paths,img_paths2,img_paths3,img_paths4)

# 千万注意要修改列数
# 通过文件名获取标签
def getlabel(name, index):
    goal = df.loc[df[0] == name]
    label = goal.iloc[0, index + 1]
    return int(label)


# 分隔文件路径的列表
img_paths_health = []
img_paths_sick = []

img_paths1_0 = []
img_paths1_1 = []
img_paths1_2 = []

img_paths2_0 = []
img_paths2_1 = []
img_paths2_2 = []

img_paths_health3 = []
img_paths_sick3 = []

img_paths_health4 = []
img_paths_sick4 = []

img_paths_health5 = []
img_paths_sick5 = []

img_paths_health6 = []
img_paths_sick6 = []

img_paths_health7 = []
img_paths_sick7 = []

img_paths_health8 = []
img_paths_sick8 = []

img_paths_health9 = []
img_paths_sick9 = []

for i in img_paths:
    if getlabel(getname(i), 0) == 0:
        img_paths_health.append(i)
    if getlabel(getname(i), 0) == 1:
        img_paths_sick.append(i)

for i in img_paths:
    if getlabel(getname(i), 1) == 0:
        img_paths1_0.append(i)
    if getlabel(getname(i), 1) == 1:
        img_paths1_1.append(i)
    if getlabel(getname(i), 1) == 2:
        img_paths1_2.append(i)

for i in img_paths:
    if getlabel(getname(i), 2) == 0:
        img_paths2_0.append(i)
    if getlabel(getname(i), 2) == 1:
        img_paths2_1.append(i)
    if getlabel(getname(i), 2) == 2:
        img_paths2_2.append(i)

for i in img_paths:
    if getlabel(getname(i), 3) == 0:
        img_paths_health3.append(i)
    if getlabel(getname(i), 3) == 1:
        img_paths_sick3.append(i)

for i in img_paths:
    if getlabel(getname(i), 4) == 0:
        img_paths_health4.append(i)
    if getlabel(getname(i), 4) == 1:
        img_paths_sick4.append(i)

for i in img_paths:
    if getlabel(getname(i), 5) == 0:
        img_paths_health5.append(i)
    if getlabel(getname(i), 5) == 1:
        img_paths_sick5.append(i)

for i in img_paths:
    if getlabel(getname(i), 6) == 0:
        img_paths_health6.append(i)
    if getlabel(getname(i), 6) == 1:
        img_paths_sick6.append(i)

for i in img_paths:
    if getlabel(getname(i), 7) == 0:
        img_paths_health7.append(i)
    if getlabel(getname(i), 7) == 1:
        img_paths_sick7.append(i)

for i in img_paths:
    if getlabel(getname(i), 8) == 0:
        img_paths_health8.append(i)
    if getlabel(getname(i), 8) == 1:
        img_paths_sick8.append(i)

for i in img_paths:
    if getlabel(getname(i), 9) == 0:
        img_paths_health9.append(i)
    if getlabel(getname(i), 9) == 1:
        img_paths_sick9.append(i)

# print("=======================")
# print(len(img_paths_health))
# print(len(img_paths_sick))
# print("=======================")
# print(len(img_paths1_0))
# print(len(img_paths1_1))
# print(len(img_paths1_2))
# print("=======================")
# print(len(img_paths2_0))
# print(len(img_paths2_1))
# print(len(img_paths2_2))
# print("=======================")
# print(len(img_paths_health3))
# print(len(img_paths_sick3))
# print("=======================")
# print(len(img_paths_health4))
# print(len(img_paths_sick4))
# print("=======================")
# print(len(img_paths_health5))
# print(len(img_paths_sick5))
# print("=======================")
# print(len(img_paths_health6))
# print(len(img_paths_sick6))
# print("=======================")
# print(len(img_paths_health7))
# print(len(img_paths_sick7))
# print("=======================")
# print(len(img_paths_health8))
# print(len(img_paths_sick8))
# print("=======================")
# print(len(img_paths_health9))
# print(len(img_paths_sick9))
# print("=======================")

img_paths_health_train_val, img_paths_health_test = train_test_split(img_paths_health, test_size=0.2, random_state=42)
img_paths_health_train, img_paths_health_val = train_test_split(img_paths_health_train_val, test_size=0.25,
                                                                random_state=42)

img_paths_sick_train_val, img_paths_sick_test = train_test_split(img_paths_sick, test_size=0.2, random_state=42)
img_paths_sick_train, img_paths_sick_val = train_test_split(img_paths_sick_train_val, test_size=0.25, random_state=42)

img_paths_train = img_paths_health_train + img_paths_sick_train
img_paths_val = img_paths_health_val + img_paths_sick_val
img_paths_test = img_paths_health_test + img_paths_sick_test


# 解决四个文件名字编号相同但是尾部不同的问题
def find_real_file_name(paths, file_name):
    real_file_name = 'abc'
    for i in paths:
        if getname(i) == getname(file_name):
            real_file_name = i
    return real_file_name


def add_elements(paths1, paths2):
    arr = []
    for i in paths1:
        real_name = find_real_file_name(paths2, i)
        arr.append(real_name)
    return arr


img_paths_train2 = add_elements(img_paths_train, img_paths2)
img_paths_train3 = add_elements(img_paths_train, img_paths3)
img_paths_train4 = add_elements(img_paths_train, img_paths4)

img_paths_val2 = add_elements(img_paths_val, img_paths2)
img_paths_val3 = add_elements(img_paths_val, img_paths3)
img_paths_val4 = add_elements(img_paths_val, img_paths4)

img_paths_test2 = add_elements(img_paths_test, img_paths2)
img_paths_test3 = add_elements(img_paths_test, img_paths3)
img_paths_test4 = add_elements(img_paths_test, img_paths4)


# 读取dicom图片
def read_dicom(folder_path, file_name):
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


# 读取预处理后的图像
def read_image(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    image = Image.open(file_path)
    return image


# 准备数据集
train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


class MyData(Dataset):
    def __init__(self, img_path, img_path2, img_path3, img_path4, transform):
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
        img1 = self.transform(read_image(folder_path, img_name))
        img2 = self.transform(read_image(folder_path2, img_name2))
        img3 = self.transform(read_image(folder_path3, img_name3))
        img4 = self.transform(read_image(folder_path4, img_name4))
        label1 = getlabel(getname(img_name), 1)
        label2 = getlabel(getname(img_name), 2)
        label3 = getlabel(getname(img_name), 3)
        label4 = getlabel(getname(img_name), 4)
        label5 = getlabel(getname(img_name), 5)
        label6 = getlabel(getname(img_name), 6)
        label7 = getlabel(getname(img_name), 7)
        label8 = getlabel(getname(img_name), 8)
        label9 = getlabel(getname(img_name), 9)
        labels = [label1, label2, label3, label4, label5, label6, label7, label8, label9]
        return img1, img2, img3, img4, torch.Tensor(labels)

    def __len__(self):
        return len(self.img_path)


train_dataset = MyData(img_paths_train, img_paths_train2, img_paths_train3, img_paths_train4, train_transform)
val_dataset = MyData(img_paths_val, img_paths_val2, img_paths_val3, img_paths_val4, val_transform)
test_dataset = MyData(img_paths_test, img_paths_test2, img_paths_test3, img_paths_test4, test_transform)

# length 长度
train_data_size = len(train_dataset)
val_data_size = len(val_dataset)
test_data_size = len(test_dataset)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("验证数据集的长度为：{}".format(val_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 创建网络模型

vgg1 = torchvision.models.vgg16(pretrained=True)
vgg2 = torchvision.models.vgg16(pretrained=True)
vgg3 = torchvision.models.vgg16(pretrained=True)
vgg4 = torchvision.models.vgg16(pretrained=True)

# 移除vgg1的最后一层全连接层
vgg1 = nn.Sequential(*list(vgg1.children())[:-1])

# 移除vgg2的最后一层全连接层
vgg2 = nn.Sequential(*list(vgg2.children())[:-1])

# 移除vgg3的最后一层全连接层
vgg3 = nn.Sequential(*list(vgg3.children())[:-1])

# 移除vgg3的最后一层全连接层
vgg4 = nn.Sequential(*list(vgg4.children())[:-1])


class Duo_ren_wu_Net(nn.Module):
    def __init__(self, vgg1, vgg2, vgg3, vgg4):
        super(Duo_ren_wu_Net, self).__init__()
        self.vgg1 = vgg1
        self.vgg2 = vgg2
        self.vgg3 = vgg3
        self.vgg4 = vgg4
        self.task1_classifier = nn.Sequential(
            nn.Linear(100352, 3)
        )
        self.task2_classifier = nn.Sequential(
            nn.Linear(100352, 3)
        )
        self.task3_classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )
        self.task4_classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )
        self.task5_classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )
        self.task6_classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )
        self.task7_classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )
        self.task8_classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )
        self.task9_classifier = nn.Sequential(
            nn.Linear(100352, 2)
        )

    def forward(self, x1, x2, x3, x4):
        feature1 = self.vgg1(x1)
        feature2 = self.vgg2(x2)
        feature3 = self.vgg3(x3)
        feature4 = self.vgg4(x4)
        merged_feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)  # 在通道维度上拼接特征
        # print(merged_feature.shape)
        merged_feature = merged_feature.view(merged_feature.size(0), -1)
        # print(merged_feature.shape)
        task1_output = self.task1_classifier(merged_feature)
        task2_output = self.task2_classifier(merged_feature)
        task3_output = self.task3_classifier(merged_feature)
        task4_output = self.task4_classifier(merged_feature)
        task5_output = self.task5_classifier(merged_feature)
        task6_output = self.task6_classifier(merged_feature)
        task7_output = self.task7_classifier(merged_feature)
        task8_output = self.task8_classifier(merged_feature)
        task9_output = self.task9_classifier(merged_feature)
        return task1_output, task2_output, task3_output, task4_output, task5_output, task6_output, task7_output, task8_output, task9_output


model = Duo_ren_wu_Net(vgg1, vgg2, vgg3, vgg4)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# sche_lr = lr_scheduler.StepLR(optim,step_size=10,gamma=0.95)
# sche_lr = lr_scheduler.CosineAnnealingLR(optim,T_max=2)


# 损失函数
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失

# 利用GPU训练
if torch.cuda.is_available():
    model.to('cuda')
torch.cuda.is_available()


def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    aucs = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]
        auc = roc_auc_score(bootstrap_true, bootstrap_pred)
        aucs.append(auc)
    aucs = np.array(aucs)
    lower_ci = np.percentile(aucs, 2.5)
    upper_ci = np.percentile(aucs, 97.5)
    return lower_ci, upper_ci


def bootstrap_auc_ci_duo(y_true, y_pred, n_bootstraps=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    aucs = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]
        try:
            auc = roc_auc_score(bootstrap_true, bootstrap_pred, multi_class='ovo')
            aucs.append(auc)
        except ValueError:  # 如果出现异常
            aucs.append(np.nan)  # 将 auc 赋值为 np.nan
    aucs = [x for x in aucs if not np.isnan(x)]
    aucs = np.array(aucs)
    lower_ci = np.percentile(aucs, 2.5)
    upper_ci = np.percentile(aucs, 97.5)
    return lower_ci, upper_ci


def one_vs_one_auc(y_true, y_pred, class_index):
    # Convert to binary labels for the selected class vs all others
    y_true_binary = np.where(y_true == class_index, 1, 0)
    y_pred_binary = y_pred[:, class_index]  # Assuming y_pred is probabilities for each class
    if np.unique(y_true_binary).shape[0] == 1:
        # If only one class present, return NaN (or any other appropriate value)
        return np.nan
    auc = roc_auc_score(y_true_binary, y_pred_binary)
    return auc


def bootstrap_auc_ci_multiclass(y_true, y_pred, n_bootstraps=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    aucs = []
    for class_index in range(y_pred.shape[1]):
        class_aucs_bootstrap = []
        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_true), len(y_true), replace=True)
            bootstrap_true = y_true[indices]
            bootstrap_pred = y_pred[indices]
            auc = one_vs_one_auc(bootstrap_true, bootstrap_pred, class_index)
            class_aucs_bootstrap.append(auc)
        class_aucs_bootstrap = [x for x in class_aucs_bootstrap if not np.isnan(x)]
        aucs.extend(class_aucs_bootstrap)

    aucs = np.array(aucs)
    lower_ci = np.percentile(aucs, 2.5)
    upper_ci = np.percentile(aucs, 97.5)

    return lower_ci, upper_ci


def cal_duo_sensitivity(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(confusion_matrix)
        # 对角线上是正确预测的
        TP = confusion_matrix[i, i]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(confusion_matrix[:, i]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(confusion_matrix[i, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN
        metrics_result.append(TP / (TP + FN))
    return metrics_result


def cal_duo_specificity(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(confusion_matrix)
        # 对角线上是正确预测的
        TP = confusion_matrix[i, i]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(confusion_matrix[:, i]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(confusion_matrix[i, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN
        metrics_result.append(TN / (TN + FP))
    return metrics_result


best_epoch = 0
best_auc = 0

loss_train = []
all_task1_accuracy_train = []
all_task2_accuracy_train = []
all_task3_accuracy_train = []
all_task4_accuracy_train = []
all_task5_accuracy_train = []
all_task6_accuracy_train = []
all_task7_accuracy_train = []
all_task8_accuracy_train = []
all_task9_accuracy_train = []
all_task1_auc_train = []
all_task2_auc_train = []
all_task3_auc_train = []
all_task4_auc_train = []
all_task5_auc_train = []
all_task6_auc_train = []
all_task7_auc_train = []
all_task8_auc_train = []
all_task9_auc_train = []
all_task1_f1_train = []
all_task2_f1_train = []
all_task3_f1_train = []
all_task4_f1_train = []
all_task5_f1_train = []
all_task6_f1_train = []
all_task7_f1_train = []
all_task8_f1_train = []
all_task9_f1_train = []
all_task1_sensitivity_train = []
all_task2_sensitivity_train = []
all_task3_sensitivity_train = []
all_task4_sensitivity_train = []
all_task5_sensitivity_train = []
all_task6_sensitivity_train = []
all_task7_sensitivity_train = []
all_task8_sensitivity_train = []
all_task9_sensitivity_train = []
all_task1_specificity_train = []
all_task2_specificity_train = []
all_task3_specificity_train = []
all_task4_specificity_train = []
all_task5_specificity_train = []
all_task6_specificity_train = []
all_task7_specificity_train = []
all_task8_specificity_train = []
all_task9_specificity_train = []
all_task1_auc_ci_lower_train = []
all_task1_auc_ci_upper_train = []
all_task2_auc_ci_lower_train = []
all_task2_auc_ci_upper_train = []
all_task3_auc_ci_lower_train = []
all_task3_auc_ci_upper_train = []
all_task4_auc_ci_lower_train = []
all_task4_auc_ci_upper_train = []
all_task5_auc_ci_lower_train = []
all_task5_auc_ci_upper_train = []
all_task6_auc_ci_lower_train = []
all_task6_auc_ci_upper_train = []
all_task7_auc_ci_lower_train = []
all_task7_auc_ci_upper_train = []
all_task8_auc_ci_lower_train = []
all_task8_auc_ci_upper_train = []
all_task9_auc_ci_lower_train = []
all_task9_auc_ci_upper_train = []

loss_val = []
all_task1_accuracy_val = []
all_task2_accuracy_val = []
all_task3_accuracy_val = []
all_task4_accuracy_val = []
all_task5_accuracy_val = []
all_task6_accuracy_val = []
all_task7_accuracy_val = []
all_task8_accuracy_val = []
all_task9_accuracy_val = []
all_task1_auc_val = []
all_task2_auc_val = []
all_task3_auc_val = []
all_task4_auc_val = []
all_task5_auc_val = []
all_task6_auc_val = []
all_task7_auc_val = []
all_task8_auc_val = []
all_task9_auc_val = []
all_task1_f1_val = []
all_task2_f1_val = []
all_task3_f1_val = []
all_task4_f1_val = []
all_task5_f1_val = []
all_task6_f1_val = []
all_task7_f1_val = []
all_task8_f1_val = []
all_task9_f1_val = []
all_task1_sensitivity_val = []
all_task2_sensitivity_val = []
all_task3_sensitivity_val = []
all_task4_sensitivity_val = []
all_task5_sensitivity_val = []
all_task6_sensitivity_val = []
all_task7_sensitivity_val = []
all_task8_sensitivity_val = []
all_task9_sensitivity_val = []
all_task1_specificity_val = []
all_task2_specificity_val = []
all_task3_specificity_val = []
all_task4_specificity_val = []
all_task5_specificity_val = []
all_task6_specificity_val = []
all_task7_specificity_val = []
all_task8_specificity_val = []
all_task9_specificity_val = []
all_task1_auc_ci_lower_val = []
all_task1_auc_ci_upper_val = []
all_task2_auc_ci_lower_val = []
all_task2_auc_ci_upper_val = []
all_task3_auc_ci_lower_val = []
all_task3_auc_ci_upper_val = []
all_task4_auc_ci_lower_val = []
all_task4_auc_ci_upper_val = []
all_task5_auc_ci_lower_val = []
all_task5_auc_ci_upper_val = []
all_task6_auc_ci_lower_val = []
all_task6_auc_ci_upper_val = []
all_task7_auc_ci_lower_val = []
all_task7_auc_ci_upper_val = []
all_task8_auc_ci_lower_val = []
all_task8_auc_ci_upper_val = []
all_task9_auc_ci_lower_val = []
all_task9_auc_ci_upper_val = []

loss_test = []
all_task1_accuracy_test = []
all_task2_accuracy_test = []
all_task3_accuracy_test = []
all_task4_accuracy_test = []
all_task5_accuracy_test = []
all_task6_accuracy_test = []
all_task7_accuracy_test = []
all_task8_accuracy_test = []
all_task9_accuracy_test = []
all_task1_auc_test = []
all_task2_auc_test = []
all_task3_auc_test = []
all_task4_auc_test = []
all_task5_auc_test = []
all_task6_auc_test = []
all_task7_auc_test = []
all_task8_auc_test = []
all_task9_auc_test = []
all_task1_f1_test = []
all_task2_f1_test = []
all_task3_f1_test = []
all_task4_f1_test = []
all_task5_f1_test = []
all_task6_f1_test = []
all_task7_f1_test = []
all_task8_f1_test = []
all_task9_f1_test = []
all_task1_sensitivity_test = []
all_task2_sensitivity_test = []
all_task3_sensitivity_test = []
all_task4_sensitivity_test = []
all_task5_sensitivity_test = []
all_task6_sensitivity_test = []
all_task7_sensitivity_test = []
all_task8_sensitivity_test = []
all_task9_sensitivity_test = []
all_task1_specificity_test = []
all_task2_specificity_test = []
all_task3_specificity_test = []
all_task4_specificity_test = []
all_task5_specificity_test = []
all_task6_specificity_test = []
all_task7_specificity_test = []
all_task8_specificity_test = []
all_task9_specificity_test = []
all_task1_auc_ci_lower_test = []
all_task1_auc_ci_upper_test = []
all_task2_auc_ci_lower_test = []
all_task2_auc_ci_upper_test = []
all_task3_auc_ci_lower_test = []
all_task3_auc_ci_upper_test = []
all_task4_auc_ci_lower_test = []
all_task4_auc_ci_upper_test = []
all_task5_auc_ci_lower_test = []
all_task5_auc_ci_upper_test = []
all_task6_auc_ci_lower_test = []
all_task6_auc_ci_upper_test = []
all_task7_auc_ci_lower_test = []
all_task7_auc_ci_upper_test = []
all_task8_auc_ci_lower_test = []
all_task8_auc_ci_upper_test = []
all_task9_auc_ci_lower_test = []
all_task9_auc_ci_upper_test = []

# 训练模型
num_epochs = 60
for epoch in range(num_epochs):
    all_task1_preds_train = []
    all_task2_preds_train = []
    all_task3_preds_train = []
    all_task4_preds_train = []
    all_task5_preds_train = []
    all_task6_preds_train = []
    all_task7_preds_train = []
    all_task8_preds_train = []
    all_task9_preds_train = []
    all_labels_train = []
    running_loss_train = 0

    model.train()
    for x1, x2, x3, x4, labels in train_dataloader:
        x1, x2, x3, x4, labels = x1.to('cuda'), x2.to('cuda'), x3.to('cuda'), x4.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        task1_output, task2_output, task3_output, task4_output, task5_output, task6_output, task7_output, task8_output, task9_output = model(
            x1, x2, x3, x4)

        labels = torch.round(labels).long()
        # 分别计算各任务的损失
        loss1 = criterion(task1_output, labels[:, 0])
        loss2 = criterion(task2_output, labels[:, 1])
        loss3 = criterion(task3_output, labels[:, 2])
        loss4 = criterion(task4_output, labels[:, 3])
        loss5 = criterion(task5_output, labels[:, 4])
        loss6 = criterion(task6_output, labels[:, 5])
        loss7 = criterion(task7_output, labels[:, 6])
        loss8 = criterion(task8_output, labels[:, 7])
        loss9 = criterion(task9_output, labels[:, 8])

        # 合并各任务的损失
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss_train += loss.item()
            all_task1_preds_train.append(task1_output)
            all_task2_preds_train.append(task2_output)
            all_task3_preds_train.append(task3_output)
            all_task4_preds_train.append(task4_output)
            all_task5_preds_train.append(task5_output)
            all_task6_preds_train.append(task6_output)
            all_task7_preds_train.append(task7_output)
            all_task8_preds_train.append(task8_output)
            all_task9_preds_train.append(task9_output)
            all_labels_train.append(labels)

    epoch_loss_train = running_loss_train / len(train_dataloader.dataset)
    all_task1_preds_train = torch.cat(all_task1_preds_train, dim=0)
    all_task2_preds_train = torch.cat(all_task2_preds_train, dim=0)
    all_task3_preds_train = torch.cat(all_task3_preds_train, dim=0)
    all_task4_preds_train = torch.cat(all_task4_preds_train, dim=0)
    all_task5_preds_train = torch.cat(all_task5_preds_train, dim=0)
    all_task6_preds_train = torch.cat(all_task6_preds_train, dim=0)
    all_task7_preds_train = torch.cat(all_task7_preds_train, dim=0)
    all_task8_preds_train = torch.cat(all_task8_preds_train, dim=0)
    all_task9_preds_train = torch.cat(all_task9_preds_train, dim=0)
    all_labels_train = torch.cat(all_labels_train, dim=0)

    # 计算准确率
    task1_accuracy_train = accuracy_score(all_labels_train[:, 0].cpu(),
                                          torch.argmax(all_task1_preds_train, dim=1).cpu())
    task2_accuracy_train = accuracy_score(all_labels_train[:, 1].cpu(),
                                          torch.argmax(all_task2_preds_train, dim=1).cpu())
    task3_accuracy_train = accuracy_score(all_labels_train[:, 2].cpu(),
                                          torch.argmax(all_task3_preds_train, dim=1).cpu())
    task4_accuracy_train = accuracy_score(all_labels_train[:, 3].cpu(),
                                          torch.argmax(all_task4_preds_train, dim=1).cpu())
    task5_accuracy_train = accuracy_score(all_labels_train[:, 4].cpu(),
                                          torch.argmax(all_task5_preds_train, dim=1).cpu())
    task6_accuracy_train = accuracy_score(all_labels_train[:, 5].cpu(),
                                          torch.argmax(all_task6_preds_train, dim=1).cpu())
    task7_accuracy_train = accuracy_score(all_labels_train[:, 6].cpu(),
                                          torch.argmax(all_task7_preds_train, dim=1).cpu())
    task8_accuracy_train = accuracy_score(all_labels_train[:, 7].cpu(),
                                          torch.argmax(all_task8_preds_train, dim=1).cpu())
    task9_accuracy_train = accuracy_score(all_labels_train[:, 8].cpu(),
                                          torch.argmax(all_task9_preds_train, dim=1).cpu())

    # 计算AUC
    task1_auc_train = roc_auc_score(all_labels_train[:, 0].cpu().detach().numpy(),
                                    F.softmax(all_task1_preds_train, dim=1).cpu().detach().numpy(), multi_class='ovo')
    task2_auc_train = roc_auc_score(all_labels_train[:, 1].cpu().detach().numpy(),
                                    F.softmax(all_task2_preds_train, dim=1).cpu().detach().numpy(), multi_class='ovo')
    task3_auc_train = roc_auc_score(all_labels_train[:, 2].cpu().detach().numpy(),
                                    all_task3_preds_train[:, 1].cpu().detach().numpy())
    task4_auc_train = roc_auc_score(all_labels_train[:, 3].cpu().detach().numpy(),
                                    all_task4_preds_train[:, 1].cpu().detach().numpy())
    task5_auc_train = roc_auc_score(all_labels_train[:, 4].cpu().detach().numpy(),
                                    all_task5_preds_train[:, 1].cpu().detach().numpy())
    task6_auc_train = roc_auc_score(all_labels_train[:, 5].cpu().detach().numpy(),
                                    all_task6_preds_train[:, 1].cpu().detach().numpy())
    task7_auc_train = roc_auc_score(all_labels_train[:, 6].cpu().detach().numpy(),
                                    all_task7_preds_train[:, 1].cpu().detach().numpy())
    task8_auc_train = roc_auc_score(all_labels_train[:, 7].cpu().detach().numpy(),
                                    all_task8_preds_train[:, 1].cpu().detach().numpy())
    task9_auc_train = roc_auc_score(all_labels_train[:, 8].cpu().detach().numpy(),
                                    all_task9_preds_train[:, 1].cpu().detach().numpy())

    # 计算 F1 分数
    task1_f1_train = f1_score(all_labels_train[:, 0].cpu(), torch.argmax(all_task1_preds_train, dim=1).cpu(),
                              average='macro')
    task2_f1_train = f1_score(all_labels_train[:, 1].cpu(), torch.argmax(all_task2_preds_train, dim=1).cpu(),
                              average='macro')
    task3_f1_train = f1_score(all_labels_train[:, 2].cpu(), torch.argmax(all_task3_preds_train, dim=1).cpu(),
                              average='binary')
    task4_f1_train = f1_score(all_labels_train[:, 3].cpu(), torch.argmax(all_task4_preds_train, dim=1).cpu(),
                              average='binary')
    task5_f1_train = f1_score(all_labels_train[:, 4].cpu(), torch.argmax(all_task5_preds_train, dim=1).cpu(),
                              average='binary')
    task6_f1_train = f1_score(all_labels_train[:, 5].cpu(), torch.argmax(all_task6_preds_train, dim=1).cpu(),
                              average='binary')
    task7_f1_train = f1_score(all_labels_train[:, 6].cpu(), torch.argmax(all_task7_preds_train, dim=1).cpu(),
                              average='binary')
    task8_f1_train = f1_score(all_labels_train[:, 7].cpu(), torch.argmax(all_task8_preds_train, dim=1).cpu(),
                              average='binary')
    task9_f1_train = f1_score(all_labels_train[:, 8].cpu(), torch.argmax(all_task9_preds_train, dim=1).cpu(),
                              average='binary')

    # 计算混淆矩阵以计算敏感度和特异度
    task1_conf_matrix_train = confusion_matrix(all_labels_train[:, 0].cpu(),
                                               torch.argmax(all_task1_preds_train, dim=1).cpu())
    task2_conf_matrix_train = confusion_matrix(all_labels_train[:, 1].cpu(),
                                               torch.argmax(all_task2_preds_train, dim=1).cpu())
    task3_conf_matrix_train = confusion_matrix(all_labels_train[:, 2].cpu(),
                                               torch.argmax(all_task3_preds_train, dim=1).cpu())
    task4_conf_matrix_train = confusion_matrix(all_labels_train[:, 3].cpu(),
                                               torch.argmax(all_task4_preds_train, dim=1).cpu())
    task5_conf_matrix_train = confusion_matrix(all_labels_train[:, 4].cpu(),
                                               torch.argmax(all_task5_preds_train, dim=1).cpu())
    task6_conf_matrix_train = confusion_matrix(all_labels_train[:, 5].cpu(),
                                               torch.argmax(all_task6_preds_train, dim=1).cpu())
    task7_conf_matrix_train = confusion_matrix(all_labels_train[:, 6].cpu(),
                                               torch.argmax(all_task7_preds_train, dim=1).cpu())
    task8_conf_matrix_train = confusion_matrix(all_labels_train[:, 7].cpu(),
                                               torch.argmax(all_task8_preds_train, dim=1).cpu())
    task9_conf_matrix_train = confusion_matrix(all_labels_train[:, 8].cpu(),
                                               torch.argmax(all_task9_preds_train, dim=1).cpu())

    task1_sensitivity_train = np.mean(cal_duo_sensitivity(task1_conf_matrix_train))
    task2_sensitivity_train = np.mean(cal_duo_sensitivity(task2_conf_matrix_train))
    task3_sensitivity_train = task3_conf_matrix_train[1, 1] / (
            task3_conf_matrix_train[1, 1] + task3_conf_matrix_train[1, 0])
    task4_sensitivity_train = task4_conf_matrix_train[1, 1] / (
            task4_conf_matrix_train[1, 1] + task4_conf_matrix_train[1, 0])
    task5_sensitivity_train = task5_conf_matrix_train[1, 1] / (
            task5_conf_matrix_train[1, 1] + task5_conf_matrix_train[1, 0])
    task6_sensitivity_train = task6_conf_matrix_train[1, 1] / (
            task6_conf_matrix_train[1, 1] + task6_conf_matrix_train[1, 0])
    task7_sensitivity_train = task7_conf_matrix_train[1, 1] / (
            task7_conf_matrix_train[1, 1] + task7_conf_matrix_train[1, 0])
    task8_sensitivity_train = task8_conf_matrix_train[1, 1] / (
            task8_conf_matrix_train[1, 1] + task8_conf_matrix_train[1, 0])
    task9_sensitivity_train = task9_conf_matrix_train[1, 1] / (
            task9_conf_matrix_train[1, 1] + task9_conf_matrix_train[1, 0])

    task1_specificity_train = np.mean(cal_duo_specificity(task1_conf_matrix_train))
    task2_specificity_train = np.mean(cal_duo_specificity(task2_conf_matrix_train))
    task3_specificity_train = task3_conf_matrix_train[0, 0] / (
            task3_conf_matrix_train[0, 0] + task3_conf_matrix_train[0, 1])
    task4_specificity_train = task4_conf_matrix_train[0, 0] / (
            task4_conf_matrix_train[0, 0] + task4_conf_matrix_train[0, 1])
    task5_specificity_train = task5_conf_matrix_train[0, 0] / (
            task5_conf_matrix_train[0, 0] + task5_conf_matrix_train[0, 1])
    task6_specificity_train = task6_conf_matrix_train[0, 0] / (
            task6_conf_matrix_train[0, 0] + task6_conf_matrix_train[0, 1])
    task7_specificity_train = task7_conf_matrix_train[0, 0] / (
            task7_conf_matrix_train[0, 0] + task7_conf_matrix_train[0, 1])
    task8_specificity_train = task8_conf_matrix_train[0, 0] / (
            task8_conf_matrix_train[0, 0] + task8_conf_matrix_train[0, 1])
    task9_specificity_train = task9_conf_matrix_train[0, 0] / (
            task9_conf_matrix_train[0, 0] + task9_conf_matrix_train[0, 1])

    # 计算 AUC 的 95% CI
    task1_auc_ci_lower_train, task1_auc_ci_upper_train = bootstrap_auc_ci_duo(
        all_labels_train[:, 0].cpu().detach().numpy(), F.softmax(all_task1_preds_train, dim=1).cpu().detach().numpy())
    task2_auc_ci_lower_train, task2_auc_ci_upper_train = bootstrap_auc_ci_duo(
        all_labels_train[:, 1].cpu().detach().numpy(), F.softmax(all_task2_preds_train, dim=1).cpu().detach().numpy())
    task3_auc_ci_lower_train, task3_auc_ci_upper_train = bootstrap_auc_ci(all_labels_train[:, 2].cpu().detach().numpy(),
                                                                          all_task3_preds_train[:,
                                                                          1].cpu().detach().numpy())
    task4_auc_ci_lower_train, task4_auc_ci_upper_train = bootstrap_auc_ci(all_labels_train[:, 3].cpu().detach().numpy(),
                                                                          all_task4_preds_train[:,
                                                                          1].cpu().detach().numpy())
    task5_auc_ci_lower_train, task5_auc_ci_upper_train = bootstrap_auc_ci(all_labels_train[:, 4].cpu().detach().numpy(),
                                                                          all_task5_preds_train[:,
                                                                          1].cpu().detach().numpy())
    task6_auc_ci_lower_train, task6_auc_ci_upper_train = bootstrap_auc_ci(all_labels_train[:, 5].cpu().detach().numpy(),
                                                                          all_task6_preds_train[:,
                                                                          1].cpu().detach().numpy())
    task7_auc_ci_lower_train, task7_auc_ci_upper_train = bootstrap_auc_ci(all_labels_train[:, 6].cpu().detach().numpy(),
                                                                          all_task7_preds_train[:,
                                                                          1].cpu().detach().numpy())
    task8_auc_ci_lower_train, task8_auc_ci_upper_train = bootstrap_auc_ci(all_labels_train[:, 7].cpu().detach().numpy(),
                                                                          all_task8_preds_train[:,
                                                                          1].cpu().detach().numpy())
    task9_auc_ci_lower_train, task9_auc_ci_upper_train = bootstrap_auc_ci(all_labels_train[:, 8].cpu().detach().numpy(),
                                                                          all_task9_preds_train[:,
                                                                          1].cpu().detach().numpy())

    loss_train.append(epoch_loss_train)
    all_task1_accuracy_train.append(task1_accuracy_train)
    all_task2_accuracy_train.append(task2_accuracy_train)
    all_task3_accuracy_train.append(task3_accuracy_train)
    all_task4_accuracy_train.append(task4_accuracy_train)
    all_task5_accuracy_train.append(task5_accuracy_train)
    all_task6_accuracy_train.append(task6_accuracy_train)
    all_task7_accuracy_train.append(task7_accuracy_train)
    all_task8_accuracy_train.append(task8_accuracy_train)
    all_task9_accuracy_train.append(task9_accuracy_train)

    all_task1_auc_train.append(task1_auc_train)
    all_task2_auc_train.append(task2_auc_train)
    all_task3_auc_train.append(task3_auc_train)
    all_task4_auc_train.append(task4_auc_train)
    all_task5_auc_train.append(task5_auc_train)
    all_task6_auc_train.append(task6_auc_train)
    all_task7_auc_train.append(task7_auc_train)
    all_task8_auc_train.append(task8_auc_train)
    all_task9_auc_train.append(task9_auc_train)

    all_task1_f1_train.append(task1_f1_train)
    all_task2_f1_train.append(task2_f1_train)
    all_task3_f1_train.append(task3_f1_train)
    all_task4_f1_train.append(task4_f1_train)
    all_task5_f1_train.append(task5_f1_train)
    all_task6_f1_train.append(task6_f1_train)
    all_task7_f1_train.append(task7_f1_train)
    all_task8_f1_train.append(task8_f1_train)
    all_task9_f1_train.append(task9_f1_train)

    all_task1_sensitivity_train.append(task1_sensitivity_train)
    all_task2_sensitivity_train.append(task2_sensitivity_train)
    all_task3_sensitivity_train.append(task3_sensitivity_train)
    all_task4_sensitivity_train.append(task4_sensitivity_train)
    all_task5_sensitivity_train.append(task5_sensitivity_train)
    all_task6_sensitivity_train.append(task6_sensitivity_train)
    all_task7_sensitivity_train.append(task7_sensitivity_train)
    all_task8_sensitivity_train.append(task8_sensitivity_train)
    all_task9_sensitivity_train.append(task9_sensitivity_train)

    all_task1_specificity_train.append(task1_specificity_train)
    all_task2_specificity_train.append(task2_specificity_train)
    all_task3_specificity_train.append(task3_specificity_train)
    all_task4_specificity_train.append(task4_specificity_train)
    all_task5_specificity_train.append(task5_specificity_train)
    all_task6_specificity_train.append(task6_specificity_train)
    all_task7_specificity_train.append(task7_specificity_train)
    all_task8_specificity_train.append(task8_specificity_train)
    all_task9_specificity_train.append(task9_specificity_train)

    all_task1_auc_ci_lower_train.append(task1_auc_ci_lower_train)
    all_task1_auc_ci_upper_train.append(task1_auc_ci_upper_train)
    all_task2_auc_ci_lower_train.append(task2_auc_ci_lower_train)
    all_task2_auc_ci_upper_train.append(task2_auc_ci_upper_train)
    all_task3_auc_ci_lower_train.append(task3_auc_ci_lower_train)
    all_task3_auc_ci_upper_train.append(task3_auc_ci_upper_train)
    all_task4_auc_ci_lower_train.append(task4_auc_ci_lower_train)
    all_task4_auc_ci_upper_train.append(task4_auc_ci_upper_train)
    all_task5_auc_ci_lower_train.append(task5_auc_ci_lower_train)
    all_task5_auc_ci_upper_train.append(task5_auc_ci_upper_train)
    all_task6_auc_ci_lower_train.append(task6_auc_ci_lower_train)
    all_task6_auc_ci_upper_train.append(task6_auc_ci_upper_train)
    all_task7_auc_ci_lower_train.append(task7_auc_ci_lower_train)
    all_task7_auc_ci_upper_train.append(task7_auc_ci_upper_train)
    all_task8_auc_ci_lower_train.append(task8_auc_ci_lower_train)
    all_task8_auc_ci_upper_train.append(task8_auc_ci_upper_train)
    all_task9_auc_ci_lower_train.append(task9_auc_ci_lower_train)
    all_task9_auc_ci_upper_train.append(task9_auc_ci_upper_train)

    # 验证模型
    model.eval()
    with torch.no_grad():
        all_task1_preds_val = []
        all_task2_preds_val = []
        all_task3_preds_val = []
        all_task4_preds_val = []
        all_task5_preds_val = []
        all_task6_preds_val = []
        all_task7_preds_val = []
        all_task8_preds_val = []
        all_task9_preds_val = []
        all_labels_val = []
        running_loss_val = 0

        for x1, x2, x3, x4, labels in val_dataloader:
            x1, x2, x3, x4, labels = x1.to('cuda'), x2.to('cuda'), x3.to('cuda'), x4.to('cuda'), labels.to('cuda')
            task1_output, task2_output, task3_output, task4_output, task5_output, task6_output, task7_output, task8_output, task9_output = model(
                x1, x2, x3, x4)

            # 分别计算各任务的损失
            labels = torch.round(labels).long()
            loss1 = criterion(task1_output, labels[:, 0])
            loss2 = criterion(task2_output, labels[:, 1])
            loss3 = criterion(task3_output, labels[:, 2])
            loss4 = criterion(task4_output, labels[:, 3])
            loss5 = criterion(task5_output, labels[:, 4])
            loss6 = criterion(task6_output, labels[:, 5])
            loss7 = criterion(task7_output, labels[:, 6])
            loss8 = criterion(task8_output, labels[:, 7])
            loss9 = criterion(task9_output, labels[:, 8])

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9

            running_loss_val += loss.item()

            all_task1_preds_val.append(task1_output)
            all_task2_preds_val.append(task2_output)
            all_task3_preds_val.append(task3_output)
            all_task4_preds_val.append(task4_output)
            all_task5_preds_val.append(task5_output)
            all_task6_preds_val.append(task6_output)
            all_task7_preds_val.append(task7_output)
            all_task8_preds_val.append(task8_output)
            all_task9_preds_val.append(task9_output)
            all_labels_val.append(labels)

        epoch_loss_val = running_loss_val / len(val_dataloader.dataset)
        all_task1_preds_val = torch.cat(all_task1_preds_val, dim=0)
        all_task2_preds_val = torch.cat(all_task2_preds_val, dim=0)
        all_task3_preds_val = torch.cat(all_task3_preds_val, dim=0)
        all_task4_preds_val = torch.cat(all_task4_preds_val, dim=0)
        all_task5_preds_val = torch.cat(all_task5_preds_val, dim=0)
        all_task6_preds_val = torch.cat(all_task6_preds_val, dim=0)
        all_task7_preds_val = torch.cat(all_task7_preds_val, dim=0)
        all_task8_preds_val = torch.cat(all_task8_preds_val, dim=0)
        all_task9_preds_val = torch.cat(all_task9_preds_val, dim=0)
        all_labels_val = torch.cat(all_labels_val, dim=0)

        # 计算准确率
        task1_accuracy_val = accuracy_score(all_labels_val[:, 0].cpu(),
                                            torch.argmax(all_task1_preds_val, dim=1).cpu())
        task2_accuracy_val = accuracy_score(all_labels_val[:, 1].cpu(),
                                            torch.argmax(all_task2_preds_val, dim=1).cpu())
        task3_accuracy_val = accuracy_score(all_labels_val[:, 2].cpu(),
                                            torch.argmax(all_task3_preds_val, dim=1).cpu())
        task4_accuracy_val = accuracy_score(all_labels_val[:, 3].cpu(),
                                            torch.argmax(all_task4_preds_val, dim=1).cpu())
        task5_accuracy_val = accuracy_score(all_labels_val[:, 4].cpu(),
                                            torch.argmax(all_task5_preds_val, dim=1).cpu())
        task6_accuracy_val = accuracy_score(all_labels_val[:, 5].cpu(),
                                            torch.argmax(all_task6_preds_val, dim=1).cpu())
        task7_accuracy_val = accuracy_score(all_labels_val[:, 6].cpu(),
                                            torch.argmax(all_task7_preds_val, dim=1).cpu())
        task8_accuracy_val = accuracy_score(all_labels_val[:, 7].cpu(),
                                            torch.argmax(all_task8_preds_val, dim=1).cpu())
        task9_accuracy_val = accuracy_score(all_labels_val[:, 8].cpu(),
                                            torch.argmax(all_task9_preds_val, dim=1).cpu())

        # 计算AUC
        task1_auc_val = roc_auc_score(all_labels_val[:, 0].cpu().detach().numpy(),
                                      F.softmax(all_task1_preds_val, dim=1).cpu().detach().numpy(),
                                      multi_class='ovo')
        task2_auc_val = roc_auc_score(all_labels_val[:, 1].cpu().detach().numpy(),
                                      F.softmax(all_task2_preds_val, dim=1).cpu().detach().numpy(),
                                      multi_class='ovo')
        task3_auc_val = roc_auc_score(all_labels_val[:, 2].cpu().detach().numpy(),
                                      all_task3_preds_val[:, 1].cpu().detach().numpy())
        task4_auc_val = roc_auc_score(all_labels_val[:, 3].cpu().detach().numpy(),
                                      all_task4_preds_val[:, 1].cpu().detach().numpy())
        task5_auc_val = roc_auc_score(all_labels_val[:, 4].cpu().detach().numpy(),
                                      all_task5_preds_val[:, 1].cpu().detach().numpy())
        task6_auc_val = roc_auc_score(all_labels_val[:, 5].cpu().detach().numpy(),
                                      all_task6_preds_val[:, 1].cpu().detach().numpy())
        task7_auc_val = roc_auc_score(all_labels_val[:, 6].cpu().detach().numpy(),
                                      all_task7_preds_val[:, 1].cpu().detach().numpy())
        task8_auc_val = roc_auc_score(all_labels_val[:, 7].cpu().detach().numpy(),
                                      all_task8_preds_val[:, 1].cpu().detach().numpy())
        task9_auc_val = roc_auc_score(all_labels_val[:, 8].cpu().detach().numpy(),
                                      all_task9_preds_val[:, 1].cpu().detach().numpy())

        all_auc_val = task1_auc_val + task2_auc_val + task3_auc_val + task4_auc_val + task5_auc_val + task6_auc_val + task7_auc_val + task8_auc_val + task9_auc_val

        # 计算 F1 分数
        task1_f1_val = f1_score(all_labels_val[:, 0].cpu(), torch.argmax(all_task1_preds_val, dim=1).cpu(),
                                average='macro')
        task2_f1_val = f1_score(all_labels_val[:, 1].cpu(), torch.argmax(all_task2_preds_val, dim=1).cpu(),
                                average='macro')
        task3_f1_val = f1_score(all_labels_val[:, 2].cpu(), torch.argmax(all_task3_preds_val, dim=1).cpu(),
                                average='binary')
        task4_f1_val = f1_score(all_labels_val[:, 3].cpu(), torch.argmax(all_task4_preds_val, dim=1).cpu(),
                                average='binary')
        task5_f1_val = f1_score(all_labels_val[:, 4].cpu(), torch.argmax(all_task5_preds_val, dim=1).cpu(),
                                average='binary')
        task6_f1_val = f1_score(all_labels_val[:, 5].cpu(), torch.argmax(all_task6_preds_val, dim=1).cpu(),
                                average='binary')
        task7_f1_val = f1_score(all_labels_val[:, 6].cpu(), torch.argmax(all_task7_preds_val, dim=1).cpu(),
                                average='binary')
        task8_f1_val = f1_score(all_labels_val[:, 7].cpu(), torch.argmax(all_task8_preds_val, dim=1).cpu(),
                                average='binary')
        task9_f1_val = f1_score(all_labels_val[:, 8].cpu(), torch.argmax(all_task9_preds_val, dim=1).cpu(),
                                average='binary')

        # 计算混淆矩阵以计算敏感度和特异度
        task1_conf_matrix_val = confusion_matrix(all_labels_val[:, 0].cpu(),
                                                 torch.argmax(all_task1_preds_val, dim=1).cpu())
        task2_conf_matrix_val = confusion_matrix(all_labels_val[:, 1].cpu(),
                                                 torch.argmax(all_task2_preds_val, dim=1).cpu())
        task3_conf_matrix_val = confusion_matrix(all_labels_val[:, 2].cpu(),
                                                 torch.argmax(all_task3_preds_val, dim=1).cpu())
        task4_conf_matrix_val = confusion_matrix(all_labels_val[:, 3].cpu(),
                                                 torch.argmax(all_task4_preds_val, dim=1).cpu())
        task5_conf_matrix_val = confusion_matrix(all_labels_val[:, 4].cpu(),
                                                 torch.argmax(all_task5_preds_val, dim=1).cpu())
        task6_conf_matrix_val = confusion_matrix(all_labels_val[:, 5].cpu(),
                                                 torch.argmax(all_task6_preds_val, dim=1).cpu())
        task7_conf_matrix_val = confusion_matrix(all_labels_val[:, 6].cpu(),
                                                 torch.argmax(all_task7_preds_val, dim=1).cpu())
        task8_conf_matrix_val = confusion_matrix(all_labels_val[:, 7].cpu(),
                                                 torch.argmax(all_task8_preds_val, dim=1).cpu())
        task9_conf_matrix_val = confusion_matrix(all_labels_val[:, 8].cpu(),
                                                 torch.argmax(all_task9_preds_val, dim=1).cpu())

        task1_sensitivity_val = np.mean(cal_duo_sensitivity(task1_conf_matrix_val))
        task2_sensitivity_val = np.mean(cal_duo_sensitivity(task2_conf_matrix_val))
        task3_sensitivity_val = task3_conf_matrix_val[1, 1] / (
                task3_conf_matrix_val[1, 1] + task3_conf_matrix_val[1, 0])
        task4_sensitivity_val = task4_conf_matrix_val[1, 1] / (
                task4_conf_matrix_val[1, 1] + task4_conf_matrix_val[1, 0])
        task5_sensitivity_val = task5_conf_matrix_val[1, 1] / (
                task5_conf_matrix_val[1, 1] + task5_conf_matrix_val[1, 0])
        task6_sensitivity_val = task6_conf_matrix_val[1, 1] / (
                task6_conf_matrix_val[1, 1] + task6_conf_matrix_val[1, 0])
        task7_sensitivity_val = task7_conf_matrix_val[1, 1] / (
                task7_conf_matrix_val[1, 1] + task7_conf_matrix_val[1, 0])
        task8_sensitivity_val = task8_conf_matrix_val[1, 1] / (
                task8_conf_matrix_val[1, 1] + task8_conf_matrix_val[1, 0])
        task9_sensitivity_val = task9_conf_matrix_val[1, 1] / (
                task9_conf_matrix_val[1, 1] + task9_conf_matrix_val[1, 0])

        task1_specificity_val = np.mean(cal_duo_specificity(task1_conf_matrix_val))
        task2_specificity_val = np.mean(cal_duo_specificity(task2_conf_matrix_val))
        task3_specificity_val = task3_conf_matrix_val[0, 0] / (
                task3_conf_matrix_val[0, 0] + task3_conf_matrix_val[0, 1])
        task4_specificity_val = task4_conf_matrix_val[0, 0] / (
                task4_conf_matrix_val[0, 0] + task4_conf_matrix_val[0, 1])
        task5_specificity_val = task5_conf_matrix_val[0, 0] / (
                task5_conf_matrix_val[0, 0] + task5_conf_matrix_val[0, 1])
        task6_specificity_val = task6_conf_matrix_val[0, 0] / (
                task6_conf_matrix_val[0, 0] + task6_conf_matrix_val[0, 1])
        task7_specificity_val = task7_conf_matrix_val[0, 0] / (
                task7_conf_matrix_val[0, 0] + task7_conf_matrix_val[0, 1])
        task8_specificity_val = task8_conf_matrix_val[0, 0] / (
                task8_conf_matrix_val[0, 0] + task8_conf_matrix_val[0, 1])
        task9_specificity_val = task9_conf_matrix_val[0, 0] / (
                task9_conf_matrix_val[0, 0] + task9_conf_matrix_val[0, 1])

        # 计算 AUC 的 95% CI
        task1_auc_ci_lower_val, task1_auc_ci_upper_val = bootstrap_auc_ci_duo(
            all_labels_val[:, 0].cpu().detach().numpy(), F.softmax(all_task1_preds_val, dim=1).cpu().detach().numpy())
        task2_auc_ci_lower_val, task2_auc_ci_upper_val = bootstrap_auc_ci_duo(
            all_labels_val[:, 1].cpu().detach().numpy(), F.softmax(all_task2_preds_val, dim=1).cpu().detach().numpy())
        task3_auc_ci_lower_val, task3_auc_ci_upper_val = bootstrap_auc_ci(
            all_labels_val[:, 2].cpu().detach().numpy(),
            all_task3_preds_val[:,
            1].cpu().detach().numpy())
        task4_auc_ci_lower_val, task4_auc_ci_upper_val = bootstrap_auc_ci(
            all_labels_val[:, 3].cpu().detach().numpy(),
            all_task4_preds_val[:,
            1].cpu().detach().numpy())
        task5_auc_ci_lower_val, task5_auc_ci_upper_val = bootstrap_auc_ci(
            all_labels_val[:, 4].cpu().detach().numpy(),
            all_task5_preds_val[:,
            1].cpu().detach().numpy())
        task6_auc_ci_lower_val, task6_auc_ci_upper_val = bootstrap_auc_ci(
            all_labels_val[:, 5].cpu().detach().numpy(),
            all_task6_preds_val[:,
            1].cpu().detach().numpy())
        task7_auc_ci_lower_val, task7_auc_ci_upper_val = bootstrap_auc_ci(
            all_labels_val[:, 6].cpu().detach().numpy(),
            all_task7_preds_val[:,
            1].cpu().detach().numpy())
        task8_auc_ci_lower_val, task8_auc_ci_upper_val = bootstrap_auc_ci(
            all_labels_val[:, 7].cpu().detach().numpy(),
            all_task8_preds_val[:,
            1].cpu().detach().numpy())
        task9_auc_ci_lower_val, task9_auc_ci_upper_val = bootstrap_auc_ci(
            all_labels_val[:, 8].cpu().detach().numpy(),
            all_task9_preds_val[:,
            1].cpu().detach().numpy())

        loss_val.append(epoch_loss_val)
        all_task1_accuracy_val.append(task1_accuracy_val)
        all_task2_accuracy_val.append(task2_accuracy_val)
        all_task3_accuracy_val.append(task3_accuracy_val)
        all_task4_accuracy_val.append(task4_accuracy_val)
        all_task5_accuracy_val.append(task5_accuracy_val)
        all_task6_accuracy_val.append(task6_accuracy_val)
        all_task7_accuracy_val.append(task7_accuracy_val)
        all_task8_accuracy_val.append(task8_accuracy_val)
        all_task9_accuracy_val.append(task9_accuracy_val)

        all_task1_auc_val.append(task1_auc_val)
        all_task2_auc_val.append(task2_auc_val)
        all_task3_auc_val.append(task3_auc_val)
        all_task4_auc_val.append(task4_auc_val)
        all_task5_auc_val.append(task5_auc_val)
        all_task6_auc_val.append(task6_auc_val)
        all_task7_auc_val.append(task7_auc_val)
        all_task8_auc_val.append(task8_auc_val)
        all_task9_auc_val.append(task9_auc_val)

        all_task1_f1_val.append(task1_f1_val)
        all_task2_f1_val.append(task2_f1_val)
        all_task3_f1_val.append(task3_f1_val)
        all_task4_f1_val.append(task4_f1_val)
        all_task5_f1_val.append(task5_f1_val)
        all_task6_f1_val.append(task6_f1_val)
        all_task7_f1_val.append(task7_f1_val)
        all_task8_f1_val.append(task8_f1_val)
        all_task9_f1_val.append(task9_f1_val)

        all_task1_sensitivity_val.append(task1_sensitivity_val)
        all_task2_sensitivity_val.append(task2_sensitivity_val)
        all_task3_sensitivity_val.append(task3_sensitivity_val)
        all_task4_sensitivity_val.append(task4_sensitivity_val)
        all_task5_sensitivity_val.append(task5_sensitivity_val)
        all_task6_sensitivity_val.append(task6_sensitivity_val)
        all_task7_sensitivity_val.append(task7_sensitivity_val)
        all_task8_sensitivity_val.append(task8_sensitivity_val)
        all_task9_sensitivity_val.append(task9_sensitivity_val)

        all_task1_specificity_val.append(task1_specificity_val)
        all_task2_specificity_val.append(task2_specificity_val)
        all_task3_specificity_val.append(task3_specificity_val)
        all_task4_specificity_val.append(task4_specificity_val)
        all_task5_specificity_val.append(task5_specificity_val)
        all_task6_specificity_val.append(task6_specificity_val)
        all_task7_specificity_val.append(task7_specificity_val)
        all_task8_specificity_val.append(task8_specificity_val)
        all_task9_specificity_val.append(task9_specificity_val)

        all_task1_auc_ci_lower_val.append(task1_auc_ci_lower_val)
        all_task1_auc_ci_upper_val.append(task1_auc_ci_upper_val)
        all_task2_auc_ci_lower_val.append(task2_auc_ci_lower_val)
        all_task2_auc_ci_upper_val.append(task2_auc_ci_upper_val)
        all_task3_auc_ci_lower_val.append(task3_auc_ci_lower_val)
        all_task3_auc_ci_upper_val.append(task3_auc_ci_upper_val)
        all_task4_auc_ci_lower_val.append(task4_auc_ci_lower_val)
        all_task4_auc_ci_upper_val.append(task4_auc_ci_upper_val)
        all_task5_auc_ci_lower_val.append(task5_auc_ci_lower_val)
        all_task5_auc_ci_upper_val.append(task5_auc_ci_upper_val)
        all_task6_auc_ci_lower_val.append(task6_auc_ci_lower_val)
        all_task6_auc_ci_upper_val.append(task6_auc_ci_upper_val)
        all_task7_auc_ci_lower_val.append(task7_auc_ci_lower_val)
        all_task7_auc_ci_upper_val.append(task7_auc_ci_upper_val)
        all_task8_auc_ci_lower_val.append(task8_auc_ci_lower_val)
        all_task8_auc_ci_upper_val.append(task8_auc_ci_upper_val)
        all_task9_auc_ci_lower_val.append(task9_auc_ci_lower_val)
        all_task9_auc_ci_upper_val.append(task9_auc_ci_upper_val)

    # 如果验证集的 AUC 更高，则保存模型
    if epoch >= 59 and all_auc_val > best_auc:
        best_epoch = epoch
        best_auc = all_auc_val
        torch.save(model, 'duo_ren_wu_val2.pth')
        print("模型已保存，best_epoch为：{}".format(best_epoch))

    # 在每个epoch结束后测试模型
    model.eval()
    with torch.no_grad():
        all_task1_preds = []
        all_task2_preds = []
        all_task3_preds = []
        all_task4_preds = []
        all_task5_preds = []
        all_task6_preds = []
        all_task7_preds = []
        all_task8_preds = []
        all_task9_preds = []
        all_labels = []
        running_loss_test = 0

        for x1, x2, x3, x4, labels in test_dataloader:
            x1, x2, x3, x4, labels = x1.to('cuda'), x2.to('cuda'), x3.to('cuda'), x4.to('cuda'), labels.to('cuda')
            task1_output, task2_output, task3_output, task4_output, task5_output, task6_output, task7_output, task8_output, task9_output = model(
                x1, x2, x3, x4)

            # 分别计算各任务的损失
            labels = torch.round(labels).long()
            loss1 = criterion(task1_output, labels[:, 0])
            loss2 = criterion(task2_output, labels[:, 1])
            loss3 = criterion(task3_output, labels[:, 2])
            loss4 = criterion(task4_output, labels[:, 3])
            loss5 = criterion(task5_output, labels[:, 4])
            loss6 = criterion(task6_output, labels[:, 5])
            loss7 = criterion(task7_output, labels[:, 6])
            loss8 = criterion(task8_output, labels[:, 7])
            loss9 = criterion(task9_output, labels[:, 8])

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9

            running_loss_test += loss.item()

            all_task1_preds.append(task1_output)
            all_task2_preds.append(task2_output)
            all_task3_preds.append(task3_output)
            all_task4_preds.append(task4_output)
            all_task5_preds.append(task5_output)
            all_task6_preds.append(task6_output)
            all_task7_preds.append(task7_output)
            all_task8_preds.append(task8_output)
            all_task9_preds.append(task9_output)
            all_labels.append(labels)

        epoch_loss_test = running_loss_test / len(test_dataloader.dataset)
        all_task1_preds = torch.cat(all_task1_preds, dim=0)
        all_task2_preds = torch.cat(all_task2_preds, dim=0)
        all_task3_preds = torch.cat(all_task3_preds, dim=0)
        all_task4_preds = torch.cat(all_task4_preds, dim=0)
        all_task5_preds = torch.cat(all_task5_preds, dim=0)
        all_task6_preds = torch.cat(all_task6_preds, dim=0)
        all_task7_preds = torch.cat(all_task7_preds, dim=0)
        all_task8_preds = torch.cat(all_task8_preds, dim=0)
        all_task9_preds = torch.cat(all_task9_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算准确率
        task1_accuracy = accuracy_score(all_labels[:, 0].cpu(), torch.argmax(all_task1_preds, dim=1).cpu())
        task2_accuracy = accuracy_score(all_labels[:, 1].cpu(), torch.argmax(all_task2_preds, dim=1).cpu())
        task3_accuracy = accuracy_score(all_labels[:, 2].cpu(), torch.argmax(all_task3_preds, dim=1).cpu())
        task4_accuracy = accuracy_score(all_labels[:, 3].cpu(), torch.argmax(all_task4_preds, dim=1).cpu())
        task5_accuracy = accuracy_score(all_labels[:, 4].cpu(), torch.argmax(all_task5_preds, dim=1).cpu())
        task6_accuracy = accuracy_score(all_labels[:, 5].cpu(), torch.argmax(all_task6_preds, dim=1).cpu())
        task7_accuracy = accuracy_score(all_labels[:, 6].cpu(), torch.argmax(all_task7_preds, dim=1).cpu())
        task8_accuracy = accuracy_score(all_labels[:, 7].cpu(), torch.argmax(all_task8_preds, dim=1).cpu())
        task9_accuracy = accuracy_score(all_labels[:, 8].cpu(), torch.argmax(all_task9_preds, dim=1).cpu())

        # 计算AUC
        task1_auc = roc_auc_score(all_labels[:, 0].cpu().detach().numpy(),
                                  F.softmax(all_task1_preds, dim=1).cpu().detach().numpy(), multi_class='ovo')
        task2_auc = roc_auc_score(all_labels[:, 1].cpu().detach().numpy(),
                                  F.softmax(all_task2_preds, dim=1).cpu().detach().numpy(), multi_class='ovo')
        task3_auc = roc_auc_score(all_labels[:, 2].cpu().detach().numpy(), all_task3_preds[:, 1].cpu().detach().numpy())
        task4_auc = roc_auc_score(all_labels[:, 3].cpu().detach().numpy(), all_task4_preds[:, 1].cpu().detach().numpy())
        task5_auc = roc_auc_score(all_labels[:, 4].cpu().detach().numpy(), all_task5_preds[:, 1].cpu().detach().numpy())
        task6_auc = roc_auc_score(all_labels[:, 5].cpu().detach().numpy(), all_task6_preds[:, 1].cpu().detach().numpy())
        task7_auc = roc_auc_score(all_labels[:, 6].cpu().detach().numpy(), all_task7_preds[:, 1].cpu().detach().numpy())
        task8_auc = roc_auc_score(all_labels[:, 7].cpu().detach().numpy(), all_task8_preds[:, 1].cpu().detach().numpy())
        task9_auc = roc_auc_score(all_labels[:, 8].cpu().detach().numpy(), all_task9_preds[:, 1].cpu().detach().numpy())

        # 计算 F1 分数
        task1_f1 = f1_score(all_labels[:, 0].cpu(), torch.argmax(all_task1_preds, dim=1).cpu(), average='macro')
        task2_f1 = f1_score(all_labels[:, 1].cpu(), torch.argmax(all_task2_preds, dim=1).cpu(), average='macro')
        task3_f1 = f1_score(all_labels[:, 2].cpu(), torch.argmax(all_task3_preds, dim=1).cpu(), average='binary')
        task4_f1 = f1_score(all_labels[:, 3].cpu(), torch.argmax(all_task4_preds, dim=1).cpu(), average='binary')
        task5_f1 = f1_score(all_labels[:, 4].cpu(), torch.argmax(all_task5_preds, dim=1).cpu(), average='binary')
        task6_f1 = f1_score(all_labels[:, 5].cpu(), torch.argmax(all_task6_preds, dim=1).cpu(), average='binary')
        task7_f1 = f1_score(all_labels[:, 6].cpu(), torch.argmax(all_task7_preds, dim=1).cpu(), average='binary')
        task8_f1 = f1_score(all_labels[:, 7].cpu(), torch.argmax(all_task8_preds, dim=1).cpu(), average='binary')
        task9_f1 = f1_score(all_labels[:, 8].cpu(), torch.argmax(all_task9_preds, dim=1).cpu(), average='binary')

        # 计算混淆矩阵以计算敏感度和特异度
        task1_conf_matrix = confusion_matrix(all_labels[:, 0].cpu(), torch.argmax(all_task1_preds, dim=1).cpu())
        task2_conf_matrix = confusion_matrix(all_labels[:, 1].cpu(), torch.argmax(all_task2_preds, dim=1).cpu())
        task3_conf_matrix = confusion_matrix(all_labels[:, 2].cpu(), torch.argmax(all_task3_preds, dim=1).cpu())
        task4_conf_matrix = confusion_matrix(all_labels[:, 3].cpu(), torch.argmax(all_task4_preds, dim=1).cpu())
        task5_conf_matrix = confusion_matrix(all_labels[:, 4].cpu(), torch.argmax(all_task5_preds, dim=1).cpu())
        task6_conf_matrix = confusion_matrix(all_labels[:, 5].cpu(), torch.argmax(all_task6_preds, dim=1).cpu())
        task7_conf_matrix = confusion_matrix(all_labels[:, 6].cpu(), torch.argmax(all_task7_preds, dim=1).cpu())
        task8_conf_matrix = confusion_matrix(all_labels[:, 7].cpu(), torch.argmax(all_task8_preds, dim=1).cpu())
        task9_conf_matrix = confusion_matrix(all_labels[:, 8].cpu(), torch.argmax(all_task9_preds, dim=1).cpu())

        # print(cal_duo_sensitivity(task1_conf_matrix))
        # print(cal_duo_sensitivity(task2_conf_matrix))
        task1_sensitivity = np.mean(cal_duo_sensitivity(task1_conf_matrix))
        task2_sensitivity = np.mean(cal_duo_sensitivity(task2_conf_matrix))
        task3_sensitivity = task3_conf_matrix[1, 1] / (task3_conf_matrix[1, 1] + task3_conf_matrix[1, 0])
        task4_sensitivity = task4_conf_matrix[1, 1] / (task4_conf_matrix[1, 1] + task4_conf_matrix[1, 0])
        task5_sensitivity = task5_conf_matrix[1, 1] / (task5_conf_matrix[1, 1] + task5_conf_matrix[1, 0])
        task6_sensitivity = task6_conf_matrix[1, 1] / (task6_conf_matrix[1, 1] + task6_conf_matrix[1, 0])
        task7_sensitivity = task7_conf_matrix[1, 1] / (task7_conf_matrix[1, 1] + task7_conf_matrix[1, 0])
        task8_sensitivity = task8_conf_matrix[1, 1] / (task8_conf_matrix[1, 1] + task8_conf_matrix[1, 0])
        task9_sensitivity = task9_conf_matrix[1, 1] / (task9_conf_matrix[1, 1] + task9_conf_matrix[1, 0])

        # print(cal_duo_specificity(task1_conf_matrix))
        # print(cal_duo_specificity(task2_conf_matrix))
        task1_specificity = np.mean(cal_duo_specificity(task1_conf_matrix))
        task2_specificity = np.mean(cal_duo_specificity(task2_conf_matrix))
        task3_specificity = task3_conf_matrix[0, 0] / (task3_conf_matrix[0, 0] + task3_conf_matrix[0, 1])
        task4_specificity = task4_conf_matrix[0, 0] / (task4_conf_matrix[0, 0] + task4_conf_matrix[0, 1])
        task5_specificity = task5_conf_matrix[0, 0] / (task5_conf_matrix[0, 0] + task5_conf_matrix[0, 1])
        task6_specificity = task6_conf_matrix[0, 0] / (task6_conf_matrix[0, 0] + task6_conf_matrix[0, 1])
        task7_specificity = task7_conf_matrix[0, 0] / (task7_conf_matrix[0, 0] + task7_conf_matrix[0, 1])
        task8_specificity = task8_conf_matrix[0, 0] / (task8_conf_matrix[0, 0] + task8_conf_matrix[0, 1])
        task9_specificity = task9_conf_matrix[0, 0] / (task9_conf_matrix[0, 0] + task9_conf_matrix[0, 1])

        # 计算 AUC 的 95% CI
        task1_auc_ci_lower, task1_auc_ci_upper = bootstrap_auc_ci_duo(all_labels[:, 0].cpu().detach().numpy(),
                                                                      F.softmax(all_task1_preds,
                                                                                dim=1).cpu().detach().numpy())
        task2_auc_ci_lower, task2_auc_ci_upper = bootstrap_auc_ci_duo(all_labels[:, 1].cpu().detach().numpy(),
                                                                      F.softmax(all_task2_preds,
                                                                                dim=1).cpu().detach().numpy())
        task3_auc_ci_lower, task3_auc_ci_upper = bootstrap_auc_ci(all_labels[:, 2].cpu().detach().numpy(),
                                                                  all_task3_preds[:, 1].cpu().detach().numpy())
        task4_auc_ci_lower, task4_auc_ci_upper = bootstrap_auc_ci(all_labels[:, 3].cpu().detach().numpy(),
                                                                  all_task4_preds[:, 1].cpu().detach().numpy())
        task5_auc_ci_lower, task5_auc_ci_upper = bootstrap_auc_ci(all_labels[:, 4].cpu().detach().numpy(),
                                                                  all_task5_preds[:, 1].cpu().detach().numpy())
        task6_auc_ci_lower, task6_auc_ci_upper = bootstrap_auc_ci(all_labels[:, 5].cpu().detach().numpy(),
                                                                  all_task6_preds[:, 1].cpu().detach().numpy())
        task7_auc_ci_lower, task7_auc_ci_upper = bootstrap_auc_ci(all_labels[:, 6].cpu().detach().numpy(),
                                                                  all_task7_preds[:, 1].cpu().detach().numpy())
        task8_auc_ci_lower, task8_auc_ci_upper = bootstrap_auc_ci(all_labels[:, 7].cpu().detach().numpy(),
                                                                  all_task8_preds[:, 1].cpu().detach().numpy())
        task9_auc_ci_lower, task9_auc_ci_upper = bootstrap_auc_ci(all_labels[:, 8].cpu().detach().numpy(),
                                                                  all_task9_preds[:, 1].cpu().detach().numpy())

        loss_test.append(epoch_loss_test)
        all_task1_accuracy_test.append(task1_accuracy)
        all_task2_accuracy_test.append(task2_accuracy)
        all_task3_accuracy_test.append(task3_accuracy)
        all_task4_accuracy_test.append(task4_accuracy)
        all_task5_accuracy_test.append(task5_accuracy)
        all_task6_accuracy_test.append(task6_accuracy)
        all_task7_accuracy_test.append(task7_accuracy)
        all_task8_accuracy_test.append(task8_accuracy)
        all_task9_accuracy_test.append(task9_accuracy)

        all_task1_auc_test.append(task1_auc)
        all_task2_auc_test.append(task2_auc)
        all_task3_auc_test.append(task3_auc)
        all_task4_auc_test.append(task4_auc)
        all_task5_auc_test.append(task5_auc)
        all_task6_auc_test.append(task6_auc)
        all_task7_auc_test.append(task7_auc)
        all_task8_auc_test.append(task8_auc)
        all_task9_auc_test.append(task9_auc)

        all_task1_f1_test.append(task1_f1)
        all_task2_f1_test.append(task2_f1)
        all_task3_f1_test.append(task3_f1)
        all_task4_f1_test.append(task4_f1)
        all_task5_f1_test.append(task5_f1)
        all_task6_f1_test.append(task6_f1)
        all_task7_f1_test.append(task7_f1)
        all_task8_f1_test.append(task8_f1)
        all_task9_f1_test.append(task9_f1)

        all_task1_sensitivity_test.append(task1_sensitivity)
        all_task2_sensitivity_test.append(task2_sensitivity)
        all_task3_sensitivity_test.append(task3_sensitivity)
        all_task4_sensitivity_test.append(task4_sensitivity)
        all_task5_sensitivity_test.append(task5_sensitivity)
        all_task6_sensitivity_test.append(task6_sensitivity)
        all_task7_sensitivity_test.append(task7_sensitivity)
        all_task8_sensitivity_test.append(task8_sensitivity)
        all_task9_sensitivity_test.append(task9_sensitivity)

        all_task1_specificity_test.append(task1_specificity)
        all_task2_specificity_test.append(task2_specificity)
        all_task3_specificity_test.append(task3_specificity)
        all_task4_specificity_test.append(task4_specificity)
        all_task5_specificity_test.append(task5_specificity)
        all_task6_specificity_test.append(task6_specificity)
        all_task7_specificity_test.append(task7_specificity)
        all_task8_specificity_test.append(task8_specificity)
        all_task9_specificity_test.append(task9_specificity)

        all_task1_auc_ci_lower_test.append(task1_auc_ci_lower)
        all_task1_auc_ci_upper_test.append(task1_auc_ci_upper)
        all_task2_auc_ci_lower_test.append(task2_auc_ci_lower)
        all_task2_auc_ci_upper_test.append(task2_auc_ci_upper)
        all_task3_auc_ci_lower_test.append(task3_auc_ci_lower)
        all_task3_auc_ci_upper_test.append(task3_auc_ci_upper)
        all_task4_auc_ci_lower_test.append(task4_auc_ci_lower)
        all_task4_auc_ci_upper_test.append(task4_auc_ci_upper)
        all_task5_auc_ci_lower_test.append(task5_auc_ci_lower)
        all_task5_auc_ci_upper_test.append(task5_auc_ci_upper)
        all_task6_auc_ci_lower_test.append(task6_auc_ci_lower)
        all_task6_auc_ci_upper_test.append(task6_auc_ci_upper)
        all_task7_auc_ci_lower_test.append(task7_auc_ci_lower)
        all_task7_auc_ci_upper_test.append(task7_auc_ci_upper)
        all_task8_auc_ci_lower_test.append(task8_auc_ci_lower)
        all_task8_auc_ci_upper_test.append(task8_auc_ci_upper)
        all_task9_auc_ci_lower_test.append(task9_auc_ci_lower)
        all_task9_auc_ci_upper_test.append(task9_auc_ci_upper)

        print(f"Epoch {epoch + 1} - \n"
              f"train loss: {epoch_loss_train:.4f}, \n"
              f"train Task1 Accuracy: {task1_accuracy_train:.4f}, \n"
              f"train Task2 Accuracy: {task2_accuracy_train:.4f}, \n"
              f"train Task3 Accuracy: {task3_accuracy_train:.4f}, \n"
              f"train Task4 Accuracy: {task4_accuracy_train:.4f}, \n"
              f"train Task5 Accuracy: {task5_accuracy_train:.4f}, \n"
              f"train Task6 Accuracy: {task6_accuracy_train:.4f}, \n"
              f"train Task7 Accuracy: {task7_accuracy_train:.4f}, \n"
              f"train Task8 Accuracy: {task8_accuracy_train:.4f}, \n"
              f"train Task9 Accuracy: {task9_accuracy_train:.4f}, \n"
              f"train Task1 AUC: {task1_auc_train:.4f}, \n"
              f"train Task2 AUC: {task2_auc_train:.4f}, \n"
              f"train Task3 AUC: {task3_auc_train:.4f}, \n"
              f"train Task4 AUC: {task4_auc_train:.4f}, \n"
              f"train Task5 AUC: {task5_auc_train:.4f}, \n"
              f"train Task6 AUC: {task6_auc_train:.4f}, \n"
              f"train Task7 AUC: {task7_auc_train:.4f}, \n"
              f"train Task8 AUC: {task8_auc_train:.4f}, \n"
              f"train Task9 AUC: {task9_auc_train:.4f}, \n"
              f"train Task1 F1 score: {task1_f1_train:.4f}, \n"
              f"train Task2 F1 score: {task2_f1_train:.4f}, \n"
              f"train Task3 F1 score: {task3_f1_train:.4f}, \n"
              f"train Task4 F1 score: {task4_f1_train:.4f}, \n"
              f"train Task5 F1 score: {task5_f1_train:.4f}, \n"
              f"train Task6 F1 score: {task6_f1_train:.4f}, \n"
              f"train Task7 F1 score: {task7_f1_train:.4f}, \n"
              f"train Task8 F1 score: {task8_f1_train:.4f}, \n"
              f"train Task9 F1 score: {task9_f1_train:.4f}, \n"
              f"train Task1 Sensitivity: {task1_sensitivity_train:.4f}, \n"
              f"train Task2 Sensitivity: {task2_sensitivity_train:.4f}, \n"
              f"train Task3 Sensitivity: {task3_sensitivity_train:.4f}, \n"
              f"train Task4 Sensitivity: {task4_sensitivity_train:.4f}, \n"
              f"train Task5 Sensitivity: {task5_sensitivity_train:.4f}, \n"
              f"train Task6 Sensitivity: {task6_sensitivity_train:.4f}, \n"
              f"train Task7 Sensitivity: {task7_sensitivity_train:.4f}, \n"
              f"train Task8 Sensitivity: {task8_sensitivity_train:.4f}, \n"
              f"train Task9 Sensitivity: {task9_sensitivity_train:.4f}, \n"
              f"train Task1 Specificity: {task1_specificity_train:.4f}, \n"
              f"train Task2 Specificity: {task2_specificity_train:.4f}, \n"
              f"train Task3 Specificity: {task3_specificity_train:.4f}, \n"
              f"train Task4 Specificity: {task4_specificity_train:.4f}, \n"
              f"train Task5 Specificity: {task5_specificity_train:.4f}, \n"
              f"train Task6 Specificity: {task6_specificity_train:.4f}, \n"
              f"train Task7 Specificity: {task7_specificity_train:.4f}, \n"
              f"train Task8 Specificity: {task8_specificity_train:.4f}, \n"
              f"train Task9 Specificity: {task9_specificity_train:.4f}, \n"
              f"train Task1 AUC_95%_CI: [{task1_auc_ci_lower_train:.4f}-{task1_auc_ci_upper_train:.4f}], \n"
              f"train Task2 AUC_95%_CI: [{task2_auc_ci_lower_train:.4f}-{task2_auc_ci_upper_train:.4f}], \n"
              f"train Task3 AUC_95%_CI: [{task3_auc_ci_lower_train:.4f}-{task3_auc_ci_upper_train:.4f}], \n"
              f"train Task4 AUC_95%_CI: [{task4_auc_ci_lower_train:.4f}-{task4_auc_ci_upper_train:.4f}], \n"
              f"train Task5 AUC_95%_CI: [{task5_auc_ci_lower_train:.4f}-{task5_auc_ci_upper_train:.4f}], \n"
              f"train Task6 AUC_95%_CI: [{task6_auc_ci_lower_train:.4f}-{task6_auc_ci_upper_train:.4f}], \n"
              f"train Task7 AUC_95%_CI: [{task7_auc_ci_lower_train:.4f}-{task7_auc_ci_upper_train:.4f}], \n"
              f"train Task8 AUC_95%_CI: [{task8_auc_ci_lower_train:.4f}-{task8_auc_ci_upper_train:.4f}], \n"
              f"train Task9 AUC_95%_CI: [{task9_auc_ci_lower_train:.4f}-{task9_auc_ci_upper_train:.4f}] \n"
              f"val loss: {epoch_loss_val:.4f}, \n"
              f"val Task1 Accuracy: {task1_accuracy_val:.4f}, \n"
              f"val Task2 Accuracy: {task2_accuracy_val:.4f}, \n"
              f"val Task3 Accuracy: {task3_accuracy_val:.4f}, \n"
              f"val Task4 Accuracy: {task4_accuracy_val:.4f}, \n"
              f"val Task5 Accuracy: {task5_accuracy_val:.4f}, \n"
              f"val Task6 Accuracy: {task6_accuracy_val:.4f}, \n"
              f"val Task7 Accuracy: {task7_accuracy_val:.4f}, \n"
              f"val Task8 Accuracy: {task8_accuracy_val:.4f}, \n"
              f"val Task9 Accuracy: {task9_accuracy_val:.4f}, \n"
              f"val Task1 AUC: {task1_auc_val:.4f}, \n"
              f"val Task2 AUC: {task2_auc_val:.4f}, \n"
              f"val Task3 AUC: {task3_auc_val:.4f}, \n"
              f"val Task4 AUC: {task4_auc_val:.4f}, \n"
              f"val Task5 AUC: {task5_auc_val:.4f}, \n"
              f"val Task6 AUC: {task6_auc_val:.4f}, \n"
              f"val Task7 AUC: {task7_auc_val:.4f}, \n"
              f"val Task8 AUC: {task8_auc_val:.4f}, \n"
              f"val Task9 AUC: {task9_auc_val:.4f}, \n"
              f"val Task1 F1 score: {task1_f1_val:.4f}, \n"
              f"val Task2 F1 score: {task2_f1_val:.4f}, \n"
              f"val Task3 F1 score: {task3_f1_val:.4f}, \n"
              f"val Task4 F1 score: {task4_f1_val:.4f}, \n"
              f"val Task5 F1 score: {task5_f1_val:.4f}, \n"
              f"val Task6 F1 score: {task6_f1_val:.4f}, \n"
              f"val Task7 F1 score: {task7_f1_val:.4f}, \n"
              f"val Task8 F1 score: {task8_f1_val:.4f}, \n"
              f"val Task9 F1 score: {task9_f1_val:.4f}, \n"
              f"val Task1 Sensitivity: {task1_sensitivity_val:.4f}, \n"
              f"val Task2 Sensitivity: {task2_sensitivity_val:.4f}, \n"
              f"val Task3 Sensitivity: {task3_sensitivity_val:.4f}, \n"
              f"val Task4 Sensitivity: {task4_sensitivity_val:.4f}, \n"
              f"val Task5 Sensitivity: {task5_sensitivity_val:.4f}, \n"
              f"val Task6 Sensitivity: {task6_sensitivity_val:.4f}, \n"
              f"val Task7 Sensitivity: {task7_sensitivity_val:.4f}, \n"
              f"val Task8 Sensitivity: {task8_sensitivity_val:.4f}, \n"
              f"val Task9 Sensitivity: {task9_sensitivity_val:.4f}, \n"
              f"val Task1 Specificity: {task1_specificity_val:.4f}, \n"
              f"val Task2 Specificity: {task2_specificity_val:.4f}, \n"
              f"val Task3 Specificity: {task3_specificity_val:.4f}, \n"
              f"val Task4 Specificity: {task4_specificity_val:.4f}, \n"
              f"val Task5 Specificity: {task5_specificity_val:.4f}, \n"
              f"val Task6 Specificity: {task6_specificity_val:.4f}, \n"
              f"val Task7 Specificity: {task7_specificity_val:.4f}, \n"
              f"val Task8 Specificity: {task8_specificity_val:.4f}, \n"
              f"val Task9 Specificity: {task9_specificity_val:.4f}, \n"
              f"val Task1 AUC_95%_CI: [{task1_auc_ci_lower_val:.4f}-{task1_auc_ci_upper_val:.4f}], \n"
              f"val Task2 AUC_95%_CI: [{task2_auc_ci_lower_val:.4f}-{task2_auc_ci_upper_val:.4f}], \n"
              f"val Task3 AUC_95%_CI: [{task3_auc_ci_lower_val:.4f}-{task3_auc_ci_upper_val:.4f}], \n"
              f"val Task4 AUC_95%_CI: [{task4_auc_ci_lower_val:.4f}-{task4_auc_ci_upper_val:.4f}], \n"
              f"val Task5 AUC_95%_CI: [{task5_auc_ci_lower_val:.4f}-{task5_auc_ci_upper_val:.4f}], \n"
              f"val Task6 AUC_95%_CI: [{task6_auc_ci_lower_val:.4f}-{task6_auc_ci_upper_val:.4f}], \n"
              f"val Task7 AUC_95%_CI: [{task7_auc_ci_lower_val:.4f}-{task7_auc_ci_upper_val:.4f}], \n"
              f"val Task8 AUC_95%_CI: [{task8_auc_ci_lower_val:.4f}-{task8_auc_ci_upper_val:.4f}], \n"
              f"val Task9 AUC_95%_CI: [{task9_auc_ci_lower_val:.4f}-{task9_auc_ci_upper_val:.4f}] \n"
              f"test loss: {epoch_loss_test:.4f}, \n"
              f"test Task1 Accuracy: {task1_accuracy:.4f}, \n"
              f"test Task2 Accuracy: {task2_accuracy:.4f}, \n"
              f"test Task3 Accuracy: {task3_accuracy:.4f}, \n"
              f"test Task4 Accuracy: {task4_accuracy:.4f}, \n"
              f"test Task5 Accuracy: {task5_accuracy:.4f}, \n"
              f"test Task6 Accuracy: {task6_accuracy:.4f}, \n"
              f"test Task7 Accuracy: {task7_accuracy:.4f}, \n"
              f"test Task8 Accuracy: {task8_accuracy:.4f}, \n"
              f"test Task9 Accuracy: {task9_accuracy:.4f}, \n"
              f"test Task1 AUC: {task1_auc:.4f}, \n"
              f"test Task2 AUC: {task2_auc:.4f}, \n"
              f"test Task3 AUC: {task3_auc:.4f}, \n"
              f"test Task4 AUC: {task4_auc:.4f}, \n"
              f"test Task5 AUC: {task5_auc:.4f}, \n"
              f"test Task6 AUC: {task6_auc:.4f}, \n"
              f"test Task7 AUC: {task7_auc:.4f}, \n"
              f"test Task8 AUC: {task8_auc:.4f}, \n"
              f"test Task9 AUC: {task9_auc:.4f}, \n"
              f"test Task1 F1 score: {task1_f1:.4f}, \n"
              f"test Task2 F1 score: {task2_f1:.4f}, \n"
              f"test Task3 F1 score: {task3_f1:.4f}, \n"
              f"test Task4 F1 score: {task4_f1:.4f}, \n"
              f"test Task5 F1 score: {task5_f1:.4f}, \n"
              f"test Task6 F1 score: {task6_f1:.4f}, \n"
              f"test Task7 F1 score: {task7_f1:.4f}, \n"
              f"test Task8 F1 score: {task8_f1:.4f}, \n"
              f"test Task9 F1 score: {task9_f1:.4f}, \n"
              f"test Task1 Sensitivity: {task1_sensitivity:.4f}, \n"
              f"test Task2 Sensitivity: {task2_sensitivity:.4f}, \n"
              f"test Task3 Sensitivity: {task3_sensitivity:.4f}, \n"
              f"test Task4 Sensitivity: {task4_sensitivity:.4f}, \n"
              f"test Task5 Sensitivity: {task5_sensitivity:.4f}, \n"
              f"test Task6 Sensitivity: {task6_sensitivity:.4f}, \n"
              f"test Task7 Sensitivity: {task7_sensitivity:.4f}, \n"
              f"test Task8 Sensitivity: {task8_sensitivity:.4f}, \n"
              f"test Task9 Sensitivity: {task9_sensitivity:.4f}, \n"
              f"test Task1 Specificity: {task1_specificity:.4f}, \n"
              f"test Task2 Specificity: {task2_specificity:.4f}, \n"
              f"test Task3 Specificity: {task3_specificity:.4f}, \n"
              f"test Task4 Specificity: {task4_specificity:.4f}, \n"
              f"test Task5 Specificity: {task5_specificity:.4f}, \n"
              f"test Task6 Specificity: {task6_specificity:.4f}, \n"
              f"test Task7 Specificity: {task7_specificity:.4f}, \n"
              f"test Task8 Specificity: {task8_specificity:.4f}, \n"
              f"test Task9 Specificity: {task9_specificity:.4f}, \n"
              f"test Task1 AUC_95%_CI: [{task1_auc_ci_lower:.4f}-{task1_auc_ci_upper:.4f}], \n"
              f"test Task2 AUC_95%_CI: [{task2_auc_ci_lower:.4f}-{task2_auc_ci_upper:.4f}], \n"
              f"test Task3 AUC_95%_CI: [{task3_auc_ci_lower:.4f}-{task3_auc_ci_upper:.4f}], \n"
              f"test Task4 AUC_95%_CI: [{task4_auc_ci_lower:.4f}-{task4_auc_ci_upper:.4f}], \n"
              f"test Task5 AUC_95%_CI: [{task5_auc_ci_lower:.4f}-{task5_auc_ci_upper:.4f}], \n"
              f"test Task6 AUC_95%_CI: [{task6_auc_ci_lower:.4f}-{task6_auc_ci_upper:.4f}], \n"
              f"test Task7 AUC_95%_CI: [{task7_auc_ci_lower:.4f}-{task7_auc_ci_upper:.4f}], \n"
              f"test Task8 AUC_95%_CI: [{task8_auc_ci_lower:.4f}-{task8_auc_ci_upper:.4f}], \n"
              f"test Task9 AUC_95%_CI: [{task9_auc_ci_lower:.4f}-{task9_auc_ci_upper:.4f}]")

# 可以在测试集上评估模型的性能并输出准确率和AUC。

print(f"Best Epoch {best_epoch + 1} - \n"
      f"train loss: {loss_train[best_epoch]:.4f}, \n"
      f"train Task1 Accuracy: {all_task1_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task2 Accuracy: {all_task2_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task3 Accuracy: {all_task3_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task4 Accuracy: {all_task4_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task5 Accuracy: {all_task5_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task6 Accuracy: {all_task6_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task7 Accuracy: {all_task7_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task8 Accuracy: {all_task8_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task9 Accuracy: {all_task9_accuracy_train[best_epoch]:.4f}, \n"
      f"train Task1 AUC: {all_task1_auc_train[best_epoch]:.4f}, \n"
      f"train Task2 AUC: {all_task2_auc_train[best_epoch]:.4f}, \n"
      f"train Task3 AUC: {all_task3_auc_train[best_epoch]:.4f}, \n"
      f"train Task4 AUC: {all_task4_auc_train[best_epoch]:.4f}, \n"
      f"train Task5 AUC: {all_task5_auc_train[best_epoch]:.4f}, \n"
      f"train Task6 AUC: {all_task6_auc_train[best_epoch]:.4f}, \n"
      f"train Task7 AUC: {all_task7_auc_train[best_epoch]:.4f}, \n"
      f"train Task8 AUC: {all_task8_auc_train[best_epoch]:.4f}, \n"
      f"train Task9 AUC: {all_task9_auc_train[best_epoch]:.4f}, \n"
      f"train Task1 F1 score: {all_task1_f1_train[best_epoch]:.4f}, \n"
      f"train Task2 F1 score: {all_task2_f1_train[best_epoch]:.4f}, \n"
      f"train Task3 F1 score: {all_task3_f1_train[best_epoch]:.4f}, \n"
      f"train Task4 F1 score: {all_task4_f1_train[best_epoch]:.4f}, \n"
      f"train Task5 F1 score: {all_task5_f1_train[best_epoch]:.4f}, \n"
      f"train Task6 F1 score: {all_task6_f1_train[best_epoch]:.4f}, \n"
      f"train Task7 F1 score: {all_task7_f1_train[best_epoch]:.4f}, \n"
      f"train Task8 F1 score: {all_task8_f1_train[best_epoch]:.4f}, \n"
      f"train Task9 F1 score: {all_task9_f1_train[best_epoch]:.4f}, \n"
      f"train Task1 Sensitivity: {all_task1_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task2 Sensitivity: {all_task2_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task3 Sensitivity: {all_task3_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task4 Sensitivity: {all_task4_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task5 Sensitivity: {all_task5_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task6 Sensitivity: {all_task6_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task7 Sensitivity: {all_task7_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task8 Sensitivity: {all_task8_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task9 Sensitivity: {all_task9_sensitivity_train[best_epoch]:.4f}, \n"
      f"train Task1 Specificity: {all_task1_specificity_train[best_epoch]:.4f}, \n"
      f"train Task2 Specificity: {all_task2_specificity_train[best_epoch]:.4f}, \n"
      f"train Task3 Specificity: {all_task3_specificity_train[best_epoch]:.4f}, \n"
      f"train Task4 Specificity: {all_task4_specificity_train[best_epoch]:.4f}, \n"
      f"train Task5 Specificity: {all_task5_specificity_train[best_epoch]:.4f}, \n"
      f"train Task6 Specificity: {all_task6_specificity_train[best_epoch]:.4f}, \n"
      f"train Task7 Specificity: {all_task7_specificity_train[best_epoch]:.4f}, \n"
      f"train Task8 Specificity: {all_task8_specificity_train[best_epoch]:.4f}, \n"
      f"train Task9 Specificity: {all_task9_specificity_train[best_epoch]:.4f}, \n"
      f"train Task1 AUC_95%_CI: [{all_task1_auc_ci_lower_train[best_epoch]:.4f}-{all_task1_auc_ci_upper_train[best_epoch]:.4f}], \n"
      f"train Task2 AUC_95%_CI: [{all_task2_auc_ci_lower_train[best_epoch]:.4f}-{all_task2_auc_ci_upper_train[best_epoch]:.4f}], \n"
      f"train Task3 AUC_95%_CI: [{all_task3_auc_ci_lower_train[best_epoch]:.4f}-{all_task3_auc_ci_upper_train[best_epoch]:.4f}], \n"
      f"train Task4 AUC_95%_CI: [{all_task4_auc_ci_lower_train[best_epoch]:.4f}-{all_task4_auc_ci_upper_train[best_epoch]:.4f}], \n"
      f"train Task5 AUC_95%_CI: [{all_task5_auc_ci_lower_train[best_epoch]:.4f}-{all_task5_auc_ci_upper_train[best_epoch]:.4f}], \n"
      f"train Task6 AUC_95%_CI: [{all_task6_auc_ci_lower_train[best_epoch]:.4f}-{all_task6_auc_ci_upper_train[best_epoch]:.4f}], \n"
      f"train Task7 AUC_95%_CI: [{all_task7_auc_ci_lower_train[best_epoch]:.4f}-{all_task7_auc_ci_upper_train[best_epoch]:.4f}], \n"
      f"train Task8 AUC_95%_CI: [{all_task8_auc_ci_lower_train[best_epoch]:.4f}-{all_task8_auc_ci_upper_train[best_epoch]:.4f}], \n"
      f"train Task9 AUC_95%_CI: [{all_task9_auc_ci_lower_train[best_epoch]:.4f}-{all_task9_auc_ci_upper_train[best_epoch]:.4f}] \n"
      f"val loss: {loss_val[best_epoch]:.4f}, \n"
      f"val Task1 Accuracy: {all_task1_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task2 Accuracy: {all_task2_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task3 Accuracy: {all_task3_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task4 Accuracy: {all_task4_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task5 Accuracy: {all_task5_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task6 Accuracy: {all_task6_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task7 Accuracy: {all_task7_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task8 Accuracy: {all_task8_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task9 Accuracy: {all_task9_accuracy_val[best_epoch]:.4f}, \n"
      f"val Task1 AUC: {all_task1_auc_val[best_epoch]:.4f}, \n"
      f"val Task2 AUC: {all_task2_auc_val[best_epoch]:.4f}, \n"
      f"val Task3 AUC: {all_task3_auc_val[best_epoch]:.4f}, \n"
      f"val Task4 AUC: {all_task4_auc_val[best_epoch]:.4f}, \n"
      f"val Task5 AUC: {all_task5_auc_val[best_epoch]:.4f}, \n"
      f"val Task6 AUC: {all_task6_auc_val[best_epoch]:.4f}, \n"
      f"val Task7 AUC: {all_task7_auc_val[best_epoch]:.4f}, \n"
      f"val Task8 AUC: {all_task8_auc_val[best_epoch]:.4f}, \n"
      f"val Task9 AUC: {all_task9_auc_val[best_epoch]:.4f}, \n"
      f"val Task1 F1 score: {all_task1_f1_val[best_epoch]:.4f}, \n"
      f"val Task2 F1 score: {all_task2_f1_val[best_epoch]:.4f}, \n"
      f"val Task3 F1 score: {all_task3_f1_val[best_epoch]:.4f}, \n"
      f"val Task4 F1 score: {all_task4_f1_val[best_epoch]:.4f}, \n"
      f"val Task5 F1 score: {all_task5_f1_val[best_epoch]:.4f}, \n"
      f"val Task6 F1 score: {all_task6_f1_val[best_epoch]:.4f}, \n"
      f"val Task7 F1 score: {all_task7_f1_val[best_epoch]:.4f}, \n"
      f"val Task8 F1 score: {all_task8_f1_val[best_epoch]:.4f}, \n"
      f"val Task9 F1 score: {all_task9_f1_val[best_epoch]:.4f}, \n"
      f"val Task1 Sensitivity: {all_task1_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task2 Sensitivity: {all_task2_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task3 Sensitivity: {all_task3_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task4 Sensitivity: {all_task4_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task5 Sensitivity: {all_task5_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task6 Sensitivity: {all_task6_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task7 Sensitivity: {all_task7_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task8 Sensitivity: {all_task8_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task9 Sensitivity: {all_task9_sensitivity_val[best_epoch]:.4f}, \n"
      f"val Task1 Specificity: {all_task1_specificity_val[best_epoch]:.4f}, \n"
      f"val Task2 Specificity: {all_task2_specificity_val[best_epoch]:.4f}, \n"
      f"val Task3 Specificity: {all_task3_specificity_val[best_epoch]:.4f}, \n"
      f"val Task4 Specificity: {all_task4_specificity_val[best_epoch]:.4f}, \n"
      f"val Task5 Specificity: {all_task5_specificity_val[best_epoch]:.4f}, \n"
      f"val Task6 Specificity: {all_task6_specificity_val[best_epoch]:.4f}, \n"
      f"val Task7 Specificity: {all_task7_specificity_val[best_epoch]:.4f}, \n"
      f"val Task8 Specificity: {all_task8_specificity_val[best_epoch]:.4f}, \n"
      f"val Task9 Specificity: {all_task9_specificity_val[best_epoch]:.4f}, \n"
      f"val Task1 AUC_95%_CI: [{all_task1_auc_ci_lower_val[best_epoch]:.4f}-{all_task1_auc_ci_upper_val[best_epoch]:.4f}], \n"
      f"val Task2 AUC_95%_CI: [{all_task2_auc_ci_lower_val[best_epoch]:.4f}-{all_task2_auc_ci_upper_val[best_epoch]:.4f}], \n"
      f"val Task3 AUC_95%_CI: [{all_task3_auc_ci_lower_val[best_epoch]:.4f}-{all_task3_auc_ci_upper_val[best_epoch]:.4f}], \n"
      f"val Task4 AUC_95%_CI: [{all_task4_auc_ci_lower_val[best_epoch]:.4f}-{all_task4_auc_ci_upper_val[best_epoch]:.4f}], \n"
      f"val Task5 AUC_95%_CI: [{all_task5_auc_ci_lower_val[best_epoch]:.4f}-{all_task5_auc_ci_upper_val[best_epoch]:.4f}], \n"
      f"val Task6 AUC_95%_CI: [{all_task6_auc_ci_lower_val[best_epoch]:.4f}-{all_task6_auc_ci_upper_val[best_epoch]:.4f}], \n"
      f"val Task7 AUC_95%_CI: [{all_task7_auc_ci_lower_val[best_epoch]:.4f}-{all_task7_auc_ci_upper_val[best_epoch]:.4f}], \n"
      f"val Task8 AUC_95%_CI: [{all_task8_auc_ci_lower_val[best_epoch]:.4f}-{all_task8_auc_ci_upper_val[best_epoch]:.4f}], \n"
      f"val Task9 AUC_95%_CI: [{all_task9_auc_ci_lower_val[best_epoch]:.4f}-{all_task9_auc_ci_upper_val[best_epoch]:.4f}] \n"
      f"test loss: {loss_test[best_epoch]:.4f}, \n"
      f"test Task1 Accuracy: {all_task1_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task2 Accuracy: {all_task2_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task3 Accuracy: {all_task3_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task4 Accuracy: {all_task4_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task5 Accuracy: {all_task5_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task6 Accuracy: {all_task6_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task7 Accuracy: {all_task7_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task8 Accuracy: {all_task8_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task9 Accuracy: {all_task9_accuracy_test[best_epoch]:.4f}, \n"
      f"test Task1 AUC: {all_task1_auc_test[best_epoch]:.4f}, \n"
      f"test Task2 AUC: {all_task2_auc_test[best_epoch]:.4f}, \n"
      f"test Task3 AUC: {all_task3_auc_test[best_epoch]:.4f}, \n"
      f"test Task4 AUC: {all_task4_auc_test[best_epoch]:.4f}, \n"
      f"test Task5 AUC: {all_task5_auc_test[best_epoch]:.4f}, \n"
      f"test Task6 AUC: {all_task6_auc_test[best_epoch]:.4f}, \n"
      f"test Task7 AUC: {all_task7_auc_test[best_epoch]:.4f}, \n"
      f"test Task8 AUC: {all_task8_auc_test[best_epoch]:.4f}, \n"
      f"test Task9 AUC: {all_task9_auc_test[best_epoch]:.4f}, \n"
      f"test Task1 F1 score: {all_task1_f1_test[best_epoch]:.4f}, \n"
      f"test Task2 F1 score: {all_task2_f1_test[best_epoch]:.4f}, \n"
      f"test Task3 F1 score: {all_task3_f1_test[best_epoch]:.4f}, \n"
      f"test Task4 F1 score: {all_task4_f1_test[best_epoch]:.4f}, \n"
      f"test Task5 F1 score: {all_task5_f1_test[best_epoch]:.4f}, \n"
      f"test Task6 F1 score: {all_task6_f1_test[best_epoch]:.4f}, \n"
      f"test Task7 F1 score: {all_task7_f1_test[best_epoch]:.4f}, \n"
      f"test Task8 F1 score: {all_task8_f1_test[best_epoch]:.4f}, \n"
      f"test Task9 F1 score: {all_task9_f1_test[best_epoch]:.4f}, \n"
      f"test Task1 Sensitivity: {all_task1_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task2 Sensitivity: {all_task2_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task3 Sensitivity: {all_task3_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task4 Sensitivity: {all_task4_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task5 Sensitivity: {all_task5_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task6 Sensitivity: {all_task6_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task7 Sensitivity: {all_task7_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task8 Sensitivity: {all_task8_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task9 Sensitivity: {all_task9_sensitivity_test[best_epoch]:.4f}, \n"
      f"test Task1 Specificity: {all_task1_specificity_test[best_epoch]:.4f}, \n"
      f"test Task2 Specificity: {all_task2_specificity_test[best_epoch]:.4f}, \n"
      f"test Task3 Specificity: {all_task3_specificity_test[best_epoch]:.4f}, \n"
      f"test Task4 Specificity: {all_task4_specificity_test[best_epoch]:.4f}, \n"
      f"test Task5 Specificity: {all_task5_specificity_test[best_epoch]:.4f}, \n"
      f"test Task6 Specificity: {all_task6_specificity_test[best_epoch]:.4f}, \n"
      f"test Task7 Specificity: {all_task7_specificity_test[best_epoch]:.4f}, \n"
      f"test Task8 Specificity: {all_task8_specificity_test[best_epoch]:.4f}, \n"
      f"test Task9 Specificity: {all_task9_specificity_test[best_epoch]:.4f}, \n"
      f"test Task1 AUC_95%_CI: [{all_task1_auc_ci_lower_test[best_epoch]:.4f}-{all_task1_auc_ci_upper_test[best_epoch]:.4f}], \n"
      f"test Task2 AUC_95%_CI: [{all_task2_auc_ci_lower_test[best_epoch]:.4f}-{all_task2_auc_ci_upper_test[best_epoch]:.4f}], \n"
      f"test Task3 AUC_95%_CI: [{all_task3_auc_ci_lower_test[best_epoch]:.4f}-{all_task3_auc_ci_upper_test[best_epoch]:.4f}], \n"
      f"test Task4 AUC_95%_CI: [{all_task4_auc_ci_lower_test[best_epoch]:.4f}-{all_task4_auc_ci_upper_test[best_epoch]:.4f}], \n"
      f"test Task5 AUC_95%_CI: [{all_task5_auc_ci_lower_test[best_epoch]:.4f}-{all_task5_auc_ci_upper_test[best_epoch]:.4f}], \n"
      f"test Task6 AUC_95%_CI: [{all_task6_auc_ci_lower_test[best_epoch]:.4f}-{all_task6_auc_ci_upper_test[best_epoch]:.4f}], \n"
      f"test Task7 AUC_95%_CI: [{all_task7_auc_ci_lower_test[best_epoch]:.4f}-{all_task7_auc_ci_upper_test[best_epoch]:.4f}], \n"
      f"test Task8 AUC_95%_CI: [{all_task8_auc_ci_lower_test[best_epoch]:.4f}-{all_task8_auc_ci_upper_test[best_epoch]:.4f}], \n"
      f"test Task9 AUC_95%_CI: [{all_task9_auc_ci_lower_test[best_epoch]:.4f}-{all_task9_auc_ci_upper_test[best_epoch]:.4f}] \n")

# 创建一个字典来存储所有的数据
data = {
    "best_epoch": [best_epoch] * num_epochs,  # 重复 epochs 次
    "best_auc": [best_auc] * num_epochs,  # 重复 epochs 次
    "loss_train": loss_train,
    "all_task1_accuracy_train": all_task1_accuracy_train,
    "all_task2_accuracy_train": all_task2_accuracy_train,
    "all_task3_accuracy_train": all_task3_accuracy_train,
    "all_task4_accuracy_train": all_task4_accuracy_train,
    "all_task5_accuracy_train": all_task5_accuracy_train,
    "all_task6_accuracy_train": all_task6_accuracy_train,
    "all_task7_accuracy_train": all_task7_accuracy_train,
    "all_task8_accuracy_train": all_task8_accuracy_train,
    "all_task9_accuracy_train": all_task9_accuracy_train,
    "all_task1_auc_train": all_task1_auc_train,
    "all_task2_auc_train": all_task2_auc_train,
    "all_task3_auc_train": all_task3_auc_train,
    "all_task4_auc_train": all_task4_auc_train,
    "all_task5_auc_train": all_task5_auc_train,
    "all_task6_auc_train": all_task6_auc_train,
    "all_task7_auc_train": all_task7_auc_train,
    "all_task8_auc_train": all_task8_auc_train,
    "all_task9_auc_train": all_task9_auc_train,
    "all_task1_f1_train": all_task1_f1_train,
    "all_task2_f1_train": all_task2_f1_train,
    "all_task3_f1_train": all_task3_f1_train,
    "all_task4_f1_train": all_task4_f1_train,
    "all_task5_f1_train": all_task5_f1_train,
    "all_task6_f1_train": all_task6_f1_train,
    "all_task7_f1_train": all_task7_f1_train,
    "all_task8_f1_train": all_task8_f1_train,
    "all_task9_f1_train": all_task9_f1_train,
    "all_task1_sensitivity_train": all_task1_sensitivity_train,
    "all_task2_sensitivity_train": all_task2_sensitivity_train,
    "all_task3_sensitivity_train": all_task3_sensitivity_train,
    "all_task4_sensitivity_train": all_task4_sensitivity_train,
    "all_task5_sensitivity_train": all_task5_sensitivity_train,
    "all_task6_sensitivity_train": all_task6_sensitivity_train,
    "all_task7_sensitivity_train": all_task7_sensitivity_train,
    "all_task8_sensitivity_train": all_task8_sensitivity_train,
    "all_task9_sensitivity_train": all_task9_sensitivity_train,
    "all_task1_specificity_train": all_task1_specificity_train,
    "all_task2_specificity_train": all_task2_specificity_train,
    "all_task3_specificity_train": all_task3_specificity_train,
    "all_task4_specificity_train": all_task4_specificity_train,
    "all_task5_specificity_train": all_task5_specificity_train,
    "all_task6_specificity_train": all_task6_specificity_train,
    "all_task7_specificity_train": all_task7_specificity_train,
    "all_task8_specificity_train": all_task8_specificity_train,
    "all_task9_specificity_train": all_task9_specificity_train,
    "all_task1_auc_ci_lower_train": all_task1_auc_ci_lower_train,
    "all_task1_auc_ci_upper_train": all_task1_auc_ci_upper_train,
    "all_task2_auc_ci_lower_train": all_task2_auc_ci_lower_train,
    "all_task2_auc_ci_upper_train": all_task2_auc_ci_upper_train,
    "all_task3_auc_ci_lower_train": all_task3_auc_ci_lower_train,
    "all_task3_auc_ci_upper_train": all_task3_auc_ci_upper_train,
    "all_task4_auc_ci_lower_train": all_task4_auc_ci_lower_train,
    "all_task4_auc_ci_upper_train": all_task4_auc_ci_upper_train,
    "all_task5_auc_ci_lower_train": all_task5_auc_ci_lower_train,
    "all_task5_auc_ci_upper_train": all_task5_auc_ci_upper_train,
    "all_task6_auc_ci_lower_train": all_task6_auc_ci_lower_train,
    "all_task6_auc_ci_upper_train": all_task6_auc_ci_upper_train,
    "all_task7_auc_ci_lower_train": all_task7_auc_ci_lower_train,
    "all_task7_auc_ci_upper_train": all_task7_auc_ci_upper_train,
    "all_task8_auc_ci_lower_train": all_task8_auc_ci_lower_train,
    "all_task8_auc_ci_upper_train": all_task8_auc_ci_upper_train,
    "all_task9_auc_ci_lower_train": all_task9_auc_ci_lower_train,
    "all_task9_auc_ci_upper_train": all_task9_auc_ci_upper_train,
    "loss_val": loss_val,
    "all_task1_accuracy_val": all_task1_accuracy_val,
    "all_task2_accuracy_val": all_task2_accuracy_val,
    "all_task3_accuracy_val": all_task3_accuracy_val,
    "all_task4_accuracy_val": all_task4_accuracy_val,
    "all_task5_accuracy_val": all_task5_accuracy_val,
    "all_task6_accuracy_val": all_task6_accuracy_val,
    "all_task7_accuracy_val": all_task7_accuracy_val,
    "all_task8_accuracy_val": all_task8_accuracy_val,
    "all_task9_accuracy_val": all_task9_accuracy_val,
    "all_task1_auc_val": all_task1_auc_val,
    "all_task2_auc_val": all_task2_auc_val,
    "all_task3_auc_val": all_task3_auc_val,
    "all_task4_auc_val": all_task4_auc_val,
    "all_task5_auc_val": all_task5_auc_val,
    "all_task6_auc_val": all_task6_auc_val,
    "all_task7_auc_val": all_task7_auc_val,
    "all_task8_auc_val": all_task8_auc_val,
    "all_task9_auc_val": all_task9_auc_val,
    "all_task1_f1_val": all_task1_f1_val,
    "all_task2_f1_val": all_task2_f1_val,
    "all_task3_f1_val": all_task3_f1_val,
    "all_task4_f1_val": all_task4_f1_val,
    "all_task5_f1_val": all_task5_f1_val,
    "all_task6_f1_val": all_task6_f1_val,
    "all_task7_f1_val": all_task7_f1_val,
    "all_task8_f1_val": all_task8_f1_val,
    "all_task9_f1_val": all_task9_f1_val,
    "all_task1_sensitivity_val": all_task1_sensitivity_val,
    "all_task2_sensitivity_val": all_task2_sensitivity_val,
    "all_task3_sensitivity_val": all_task3_sensitivity_val,
    "all_task4_sensitivity_val": all_task4_sensitivity_val,
    "all_task5_sensitivity_val": all_task5_sensitivity_val,
    "all_task6_sensitivity_val": all_task6_sensitivity_val,
    "all_task7_sensitivity_val": all_task7_sensitivity_val,
    "all_task8_sensitivity_val": all_task8_sensitivity_val,
    "all_task9_sensitivity_val": all_task9_sensitivity_val,
    "all_task1_specificity_val": all_task1_specificity_val,
    "all_task2_specificity_val": all_task2_specificity_val,
    "all_task3_specificity_val": all_task3_specificity_val,
    "all_task4_specificity_val": all_task4_specificity_val,
    "all_task5_specificity_val": all_task5_specificity_val,
    "all_task6_specificity_val": all_task6_specificity_val,
    "all_task7_specificity_val": all_task7_specificity_val,
    "all_task8_specificity_val": all_task8_specificity_val,
    "all_task9_specificity_val": all_task9_specificity_val,
    "all_task1_auc_ci_lower_val": all_task1_auc_ci_lower_val,
    "all_task1_auc_ci_upper_val": all_task1_auc_ci_upper_val,
    "all_task2_auc_ci_lower_val": all_task2_auc_ci_lower_val,
    "all_task2_auc_ci_upper_val": all_task2_auc_ci_upper_val,
    "all_task3_auc_ci_lower_val": all_task3_auc_ci_lower_val,
    "all_task3_auc_ci_upper_val": all_task3_auc_ci_upper_val,
    "all_task4_auc_ci_lower_val": all_task4_auc_ci_lower_val,
    "all_task4_auc_ci_upper_val": all_task4_auc_ci_upper_val,
    "all_task5_auc_ci_lower_val": all_task5_auc_ci_lower_val,
    "all_task5_auc_ci_upper_val": all_task5_auc_ci_upper_val,
    "all_task6_auc_ci_lower_val": all_task6_auc_ci_lower_val,
    "all_task6_auc_ci_upper_val": all_task6_auc_ci_upper_val,
    "all_task7_auc_ci_lower_val": all_task7_auc_ci_lower_val,
    "all_task7_auc_ci_upper_val": all_task7_auc_ci_upper_val,
    "all_task8_auc_ci_lower_val": all_task8_auc_ci_lower_val,
    "all_task8_auc_ci_upper_val": all_task8_auc_ci_upper_val,
    "all_task9_auc_ci_lower_val": all_task9_auc_ci_lower_val,
    "all_task9_auc_ci_upper_val": all_task9_auc_ci_upper_val,
    "loss_test": loss_test,
    "all_task1_accuracy_test": all_task1_accuracy_test,
    "all_task2_accuracy_test": all_task2_accuracy_test,
    "all_task3_accuracy_test": all_task3_accuracy_test,
    "all_task4_accuracy_test": all_task4_accuracy_test,
    "all_task5_accuracy_test": all_task5_accuracy_test,
    "all_task6_accuracy_test": all_task6_accuracy_test,
    "all_task7_accuracy_test": all_task7_accuracy_test,
    "all_task8_accuracy_test": all_task8_accuracy_test,
    "all_task9_accuracy_test": all_task9_accuracy_test,
    "all_task1_auc_test": all_task1_auc_test,
    "all_task2_auc_test": all_task2_auc_test,
    "all_task3_auc_test": all_task3_auc_test,
    "all_task4_auc_test": all_task4_auc_test,
    "all_task5_auc_test": all_task5_auc_test,
    "all_task6_auc_test": all_task6_auc_test,
    "all_task7_auc_test": all_task7_auc_test,
    "all_task8_auc_test": all_task8_auc_test,
    "all_task9_auc_test": all_task9_auc_test,
    "all_task1_f1_test": all_task1_f1_test,
    "all_task2_f1_test": all_task2_f1_test,
    "all_task3_f1_test": all_task3_f1_test,
    "all_task4_f1_test": all_task4_f1_test,
    "all_task5_f1_test": all_task5_f1_test,
    "all_task6_f1_test": all_task6_f1_test,
    "all_task7_f1_test": all_task7_f1_test,
    "all_task8_f1_test": all_task8_f1_test,
    "all_task9_f1_test": all_task9_f1_test,
    "all_task1_sensitivity_test": all_task1_sensitivity_test,
    "all_task2_sensitivity_test": all_task2_sensitivity_test,
    "all_task3_sensitivity_test": all_task3_sensitivity_test,
    "all_task4_sensitivity_test": all_task4_sensitivity_test,
    "all_task5_sensitivity_test": all_task5_sensitivity_test,
    "all_task6_sensitivity_test": all_task6_sensitivity_test,
    "all_task7_sensitivity_test": all_task7_sensitivity_test,
    "all_task8_sensitivity_test": all_task8_sensitivity_test,
    "all_task9_sensitivity_test": all_task9_sensitivity_test,
    "all_task1_specificity_test": all_task1_specificity_test,
    "all_task2_specificity_test": all_task2_specificity_test,
    "all_task3_specificity_test": all_task3_specificity_test,
    "all_task4_specificity_test": all_task4_specificity_test,
    "all_task5_specificity_test": all_task5_specificity_test,
    "all_task6_specificity_test": all_task6_specificity_test,
    "all_task7_specificity_test": all_task7_specificity_test,
    "all_task8_specificity_test": all_task8_specificity_test,
    "all_task9_specificity_test": all_task9_specificity_test,
    "all_task1_auc_ci_lower_test": all_task1_auc_ci_lower_test,
    "all_task1_auc_ci_upper_test": all_task1_auc_ci_upper_test,
    "all_task2_auc_ci_lower_test": all_task2_auc_ci_lower_test,
    "all_task2_auc_ci_upper_test": all_task2_auc_ci_upper_test,
    "all_task3_auc_ci_lower_test": all_task3_auc_ci_lower_test,
    "all_task3_auc_ci_upper_test": all_task3_auc_ci_upper_test,
    "all_task4_auc_ci_lower_test": all_task4_auc_ci_lower_test,
    "all_task4_auc_ci_upper_test": all_task4_auc_ci_upper_test,
    "all_task5_auc_ci_lower_test": all_task5_auc_ci_lower_test,
    "all_task5_auc_ci_upper_test": all_task5_auc_ci_upper_test,
    "all_task6_auc_ci_lower_test": all_task6_auc_ci_lower_test,
    "all_task6_auc_ci_upper_test": all_task6_auc_ci_upper_test,
    "all_task7_auc_ci_lower_test": all_task7_auc_ci_lower_test,
    "all_task7_auc_ci_upper_test": all_task7_auc_ci_upper_test,
    "all_task8_auc_ci_lower_test": all_task8_auc_ci_lower_test,
    "all_task8_auc_ci_upper_test": all_task8_auc_ci_upper_test,
    "all_task9_auc_ci_lower_test": all_task9_auc_ci_lower_test,
    "all_task9_auc_ci_upper_test": all_task9_auc_ci_upper_test
}

# 将字典转换为 DataFrame
df_save = pd.DataFrame(data)

# 保存到 CSV 文件
df_save.to_csv('duo_ren_wu_val2.csv', index=False)