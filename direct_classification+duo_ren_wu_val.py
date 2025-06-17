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
        img1 = self.transform(read_image(folder_path, img_name))
        img2 = self.transform(read_image(folder_path2, img_name2))
        img3 = self.transform(read_image(folder_path3, img_name3))
        img4 = self.transform(read_image(folder_path4, img_name4))
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

# 多任务网络模型
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

    def forward(self, x1,x2,x3,x4):
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


model_duo = torch.load('duo_ren_wu_val2.pth')


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

    def forward(self, x1,x2,x3,x4):
        feature1 = self.vgg1(x1)
        feature2 = self.vgg2(x2)
        feature3 = self.vgg3(x3)
        feature4 = self.vgg4(x4)
        merged_feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)  # 在通道维度上拼接特征
        # print(merged_feature.shape)
        merged_feature = merged_feature.view(merged_feature.size(0), -1)
        # print(merged_feature.shape)
        output = self.classifier(merged_feature)
        return output


model = FusionNet_and_Duo_ren_wu_Net(vgg1, vgg2, vgg3, vgg4)


# 存储共享特征提取层的初始参数
shared_initial_params = {}

# 迁移net1中相同结构的参数到net2
for name, param in model_duo.named_parameters():
    if name in model.state_dict():
        model.state_dict()[name].copy_(param.data)
        # 保存初始参数
        shared_initial_params[name] = param.detach().clone()


#优化器
optim=torch.optim.Adam(model.parameters(),lr=0.0001)
# sche_lr = lr_scheduler.StepLR(optim,step_size=10,gamma=0.95)
# sche_lr = lr_scheduler.CosineAnnealingLR(optim,T_max=2)


# 正则化系数
alpha = 0.01


# 定义自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self, alpha, shared_initial_params):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.shared_initial_params = shared_initial_params

    def forward(self, outputs, targets, model):
        # 计算任务损失
        task_loss = nn.CrossEntropyLoss()(outputs, targets)

        # 计算正则化项
        reg_loss = 0
        for param, initial_param in zip(model.shared_feature_extractor.parameters(), self.shared_initial_params):
            reg_loss += torch.norm(param - initial_param, p=2) ** 2

        # 总损失
        total_loss = task_loss + self.alpha * reg_loss
        return total_loss


# 使用自定义损失函数
criterion = CustomLoss(alpha=alpha, shared_initial_params=shared_initial_params)


# #损失函数
# criterion = nn.CrossEntropyLoss()



#利用GPU训练
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



#训练函数
def fit(epoch, model, trainloader, valloader, testloader, best_epoch, best_auc):
    all_preds_train = []
    all_labels_train = []
    running_loss_train = 0
    model.train()
    for x1, x2, x3, x4, label in tqdm(trainloader):
        x1, x2, x3, x4, label = x1.to('cuda'), x2.to('cuda'), x3.to('cuda'), x4.to('cuda'), label.to('cuda')
        optim.zero_grad()
        output = model(x1, x2, x3, x4)
        loss = criterion(output, label)
        loss.backward()
        optim.step()


        with torch.no_grad():
            running_loss_train += loss.item()
            all_preds_train.append(output)
            all_labels_train.append(label)

    # sche_lr.step()
    epoch_train_loss = running_loss_train / len(train_dataloader.dataset)
    all_preds_train = torch.cat(all_preds_train, dim=0)
    all_labels_train = torch.cat(all_labels_train, dim=0)
    # 计算准确率
    epoch_train_acc = accuracy_score(all_labels_train.cpu(), torch.argmax(all_preds_train, dim=1).cpu())

    #画图用
    epoch_train_FPR, epoch_train_TPR, epoch_train_P = roc_curve(all_labels_train.cpu().detach().numpy(), all_preds_train[:, 1].cpu().detach().numpy())

    # 计算 F1 分数
    epoch_train_f1 = f1_score(all_labels_train.cpu(), torch.argmax(all_preds_train, dim=1).cpu())

    # 计算混淆矩阵以计算敏感度和特异度
    epoch_train_conf_matrix = confusion_matrix(all_labels_train.cpu(), torch.argmax(all_preds_train, dim=1).cpu())
    epoch_train_sensitivity = epoch_train_conf_matrix[1, 1] / (epoch_train_conf_matrix[1, 1] + epoch_train_conf_matrix[1, 0])
    epoch_train_specificity = epoch_train_conf_matrix[0, 0] / (epoch_train_conf_matrix[0, 0] + epoch_train_conf_matrix[0, 1])

    # 计算 AUC 和 AUC 的 95% CI
    epoch_train_auc = roc_auc_score(all_labels_train.cpu().detach().numpy(),all_preds_train[:, 1].cpu().detach().numpy())
    epoch_train_auc_ci_lower, epoch_train_auc_ci_upper = bootstrap_auc_ci(all_labels_train.cpu().detach().numpy(),all_preds_train[:, 1].cpu().detach().numpy())


    model.eval()
    with torch.no_grad():
        all_preds_val = []
        all_labels_val = []
        running_loss_val = 0

        for x1, x2, x3, x4, label in tqdm(valloader):
            x1, x2, x3, x4, label = x1.to('cuda'), x2.to('cuda'), x3.to('cuda'), x4.to('cuda'), label.to('cuda')
            output = model(x1, x2, x3, x4)
            loss = criterion(output, label)
            running_loss_val += loss.item()

            all_preds_val.append(output)
            all_labels_val.append(label)

    epoch_val_loss = running_loss_val / len(val_dataloader.dataset)
    all_preds_val = torch.cat(all_preds_val, dim=0)
    all_labels_val = torch.cat(all_labels_val, dim=0)
    # 计算准确率
    epoch_val_acc = accuracy_score(all_labels_val.cpu(), torch.argmax(all_preds_val, dim=1).cpu())

    # 画图用
    epoch_val_FPR, epoch_val_TPR, epoch_val_P = roc_curve(all_labels_val.cpu().detach().numpy(),
                                                          all_preds_val[:, 1].cpu().detach().numpy())

    # 计算 F1 分数
    epoch_val_f1 = f1_score(all_labels_val.cpu(), torch.argmax(all_preds_val, dim=1).cpu())

    # 计算混淆矩阵以计算敏感度和特异度
    epoch_val_conf_matrix = confusion_matrix(all_labels_val.cpu(), torch.argmax(all_preds_val, dim=1).cpu())
    epoch_val_sensitivity = epoch_val_conf_matrix[1, 1] / (
            epoch_val_conf_matrix[1, 1] + epoch_val_conf_matrix[1, 0])
    epoch_val_specificity = epoch_val_conf_matrix[0, 0] / (
            epoch_val_conf_matrix[0, 0] + epoch_val_conf_matrix[0, 1])

    # 计算 AUC 和 AUC 的 95% CI
    epoch_val_auc = roc_auc_score(all_labels_val.cpu().detach().numpy(), all_preds_val[:, 1].cpu().detach().numpy())
    epoch_val_auc_ci_lower, epoch_val_auc_ci_upper = bootstrap_auc_ci(all_labels_val.cpu().detach().numpy(),
                                                                      all_preds_val[:, 1].cpu().detach().numpy())

    # 如果验证集的 AUC 更高，则保存模型
    if epoch >= 5 and epoch_val_auc > best_auc:
        best_epoch = epoch
        best_auc = epoch_val_auc
        torch.save(model, 'direct_classification+duo_ren_wu_val.pth')
        print("模型已保存，best_epoch为：{}".format(best_epoch))



    model.eval()
    with torch.no_grad():
        all_preds_test = []
        all_labels_test = []
        running_loss_test = 0

        for x1, x2, x3, x4, label in tqdm(testloader):
            x1, x2, x3, x4, label = x1.to('cuda'), x2.to('cuda'), x3.to('cuda'), x4.to('cuda'), label.to('cuda')
            output = model(x1, x2, x3, x4)
            loss = criterion(output, label)
            running_loss_test += loss.item()

            all_preds_test.append(output)
            all_labels_test.append(label)

    epoch_test_loss = running_loss_test / len(test_dataloader.dataset)
    all_preds_test = torch.cat(all_preds_test, dim=0)
    all_labels_test = torch.cat(all_labels_test, dim=0)
    # 计算准确率
    epoch_test_acc = accuracy_score(all_labels_test.cpu(), torch.argmax(all_preds_test, dim=1).cpu())

    # 画图用
    epoch_test_FPR, epoch_test_TPR, epoch_test_P = roc_curve(all_labels_test.cpu().detach().numpy(),all_preds_test[:, 1].cpu().detach().numpy())

    # 计算 F1 分数
    epoch_test_f1 = f1_score(all_labels_test.cpu(), torch.argmax(all_preds_test, dim=1).cpu())

    # 计算混淆矩阵以计算敏感度和特异度
    epoch_test_conf_matrix = confusion_matrix(all_labels_test.cpu(), torch.argmax(all_preds_test, dim=1).cpu())
    epoch_test_sensitivity = epoch_test_conf_matrix[1, 1] / (epoch_test_conf_matrix[1, 1] + epoch_test_conf_matrix[1, 0])
    epoch_test_specificity = epoch_test_conf_matrix[0, 0] / (epoch_test_conf_matrix[0, 0] + epoch_test_conf_matrix[0, 1])

    # 计算 AUC 和 AUC 的 95% CI
    epoch_test_auc = roc_auc_score(all_labels_test.cpu().detach().numpy(), all_preds_test[:, 1].cpu().detach().numpy())
    epoch_test_auc_ci_lower, epoch_test_auc_ci_upper = bootstrap_auc_ci(all_labels_test.cpu().detach().numpy(),all_preds_test[:, 1].cpu().detach().numpy())



    print('epoch: ', epoch,
          'train_loss： ', round(epoch_train_loss, 3),
          'train_accuracy:', round(epoch_train_acc, 3),
          'train_auc:', round(epoch_train_auc, 3),
          'train_F1_score:', round(epoch_train_f1, 3),
          'train_sensitivity:', round(epoch_train_sensitivity, 3),
          'train_specificity:', round(epoch_train_specificity, 3),
          'train_AUC_95%_CI:', '[', round(epoch_train_auc_ci_lower, 3), '-', round(epoch_train_auc_ci_upper, 3), ']',
          'val_loss： ', round(epoch_val_loss, 3),
          'val_accuracy:', round(epoch_val_acc, 3),
          'val_auc:', round(epoch_val_auc, 3),
          'val_F1_score:', round(epoch_val_f1, 3),
          'val_sensitivity:', round(epoch_val_sensitivity, 3),
          'val_specificity:', round(epoch_val_specificity, 3),
          'val_AUC_95%_CI:', '[', round(epoch_val_auc_ci_lower, 3), '-', round(epoch_val_auc_ci_upper, 3), ']',
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3),
          'test_auc:', round(epoch_test_auc, 3),
          'test_F1_score:', round(epoch_test_f1, 3),
          'test_sensitivity:', round(epoch_test_sensitivity, 3),
          'test_specificity:', round(epoch_test_specificity, 3),
          'test_AUC_95%_CI:', '[', round(epoch_test_auc_ci_lower, 3), '-', round(epoch_test_auc_ci_upper, 3), ']'
          )

    return epoch_train_loss, epoch_train_acc, epoch_train_auc, epoch_train_FPR, epoch_train_TPR, epoch_train_P, epoch_train_f1, epoch_train_sensitivity, epoch_train_specificity, epoch_train_auc_ci_lower, epoch_train_auc_ci_upper, \
        epoch_val_loss, epoch_val_acc, epoch_val_auc, epoch_val_FPR, epoch_val_TPR, epoch_val_P, epoch_val_f1, epoch_val_sensitivity, epoch_val_specificity, epoch_val_auc_ci_lower, epoch_val_auc_ci_upper, \
        epoch_test_loss, epoch_test_acc, epoch_test_auc, epoch_test_FPR, epoch_test_TPR, epoch_test_P, epoch_test_f1, epoch_test_sensitivity, epoch_test_specificity, epoch_test_auc_ci_lower, epoch_test_auc_ci_upper, \
        best_epoch, best_auc


# 开始训练
epochs = 60

best_epoch = 0
best_auc = 0

train_loss = []
train_acc = []
train_auc = []
train_FPR = []
train_TPR = []
train_P = []
train_f1 = []
train_sensitivity = []
train_specificity = []
train_auc_ci_lower = []
train_auc_ci_upper = []

val_loss = []
val_acc = []
val_auc = []
val_FPR = []
val_TPR = []
val_P = []
val_f1 = []
val_sensitivity = []
val_specificity = []
val_auc_ci_lower = []
val_auc_ci_upper = []

test_loss = []
test_acc = []
test_auc = []
test_FPR = []
test_TPR = []
test_P = []
test_f1 = []
test_sensitivity = []
test_specificity = []
test_auc_ci_lower = []
test_auc_ci_upper = []

for epoch in range(epochs):
    epoch_train_loss, epoch_train_acc, epoch_train_auc, epoch_train_FPR, epoch_train_TPR, epoch_train_P, epoch_train_f1, epoch_train_sensitivity, epoch_train_specificity, epoch_train_auc_ci_lower, epoch_train_auc_ci_upper, \
        epoch_val_loss, epoch_val_acc, epoch_val_auc, epoch_val_FPR, epoch_val_TPR, epoch_val_P, epoch_val_f1, epoch_val_sensitivity, epoch_val_specificity, epoch_val_auc_ci_lower, epoch_val_auc_ci_upper, \
        epoch_test_loss, epoch_test_acc, epoch_test_auc, epoch_test_FPR, epoch_test_TPR, epoch_test_P, epoch_test_f1, epoch_test_sensitivity, epoch_test_specificity, epoch_test_auc_ci_lower, epoch_test_auc_ci_upper, \
        best_epoch, best_auc = fit(epoch, model, train_dataloader, val_dataloader, test_dataloader, best_epoch,
                                   best_auc)

    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    train_auc.append(epoch_train_auc)
    train_FPR.append(epoch_train_FPR)
    train_TPR.append(epoch_train_TPR)
    train_P.append(epoch_train_P)
    train_f1.append(epoch_train_f1)
    train_sensitivity.append(epoch_train_sensitivity)
    train_specificity.append(epoch_train_specificity)
    train_auc_ci_lower.append(epoch_train_auc_ci_lower)
    train_auc_ci_upper.append(epoch_train_auc_ci_upper)

    val_loss.append(epoch_val_loss)
    val_acc.append(epoch_val_acc)
    val_auc.append(epoch_val_auc)
    val_FPR.append(epoch_val_FPR)
    val_TPR.append(epoch_val_TPR)
    val_P.append(epoch_val_P)
    val_f1.append(epoch_val_f1)
    val_sensitivity.append(epoch_val_sensitivity)
    val_specificity.append(epoch_val_specificity)
    val_auc_ci_lower.append(epoch_val_auc_ci_lower)
    val_auc_ci_upper.append(epoch_val_auc_ci_upper)

    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
    test_auc.append(epoch_test_auc)
    test_FPR.append(epoch_test_FPR)
    test_TPR.append(epoch_test_TPR)
    test_P.append(epoch_test_P)
    test_f1.append(epoch_test_f1)
    test_sensitivity.append(epoch_test_sensitivity)
    test_specificity.append(epoch_test_specificity)
    test_auc_ci_lower.append(epoch_test_auc_ci_lower)
    test_auc_ci_upper.append(epoch_test_auc_ci_upper)

print('best_epoch: ', best_epoch,
      'train_loss： ', round(train_loss[best_epoch], 3),
      'train_accuracy:', round(train_acc[best_epoch], 3),
      'train_auc:', round(train_auc[best_epoch], 3),
      'train_F1_score:', round(train_f1[best_epoch], 3),
      'train_sensitivity:', round(train_sensitivity[best_epoch], 3),
      'train_specificity:', round(train_specificity[best_epoch], 3),
      'train_AUC_95%_CI:', '[', round(train_auc_ci_lower[best_epoch], 3), '-', round(train_auc_ci_upper[best_epoch], 3),
      ']',
      'val_loss： ', round(val_loss[best_epoch], 3),
      'val_accuracy:', round(val_acc[best_epoch], 3),
      'val_auc:', round(val_auc[best_epoch], 3),
      'val_F1_score:', round(val_f1[best_epoch], 3),
      'val_sensitivity:', round(val_sensitivity[best_epoch], 3),
      'val_specificity:', round(val_specificity[best_epoch], 3),
      'val_AUC_95%_CI:', '[', round(val_auc_ci_lower[best_epoch], 3), '-', round(val_auc_ci_upper[best_epoch], 3), ']',
      'test_loss： ', round(test_loss[best_epoch], 3),
      'test_accuracy:', round(test_acc[best_epoch], 3),
      'test_auc:', round(test_auc[best_epoch], 3),
      'test_F1_score:', round(test_f1[best_epoch], 3),
      'test_sensitivity:', round(test_sensitivity[best_epoch], 3),
      'test_specificity:', round(test_specificity[best_epoch], 3),
      'test_AUC_95%_CI:', '[', round(test_auc_ci_lower[best_epoch], 3), '-', round(test_auc_ci_upper[best_epoch], 3),
      ']'
      )


# 创建一个字典来存储所有的数据
data = {
    "epochs": list(range(1, epochs + 1)),  # 使用每个 epoch 的编号
    "best_epoch": [best_epoch] * epochs,  # 重复 epochs 次
    "best_auc": [best_auc] * epochs,  # 重复 epochs 次
    "train_loss": train_loss,
    "train_acc": train_acc,
    "train_auc": train_auc,
    "train_FPR": train_FPR,
    "train_TPR": train_TPR,
    "train_P": train_P,
    "train_f1": train_f1,
    "train_sensitivity": train_sensitivity,
    "train_specificity": train_specificity,
    "train_auc_ci_lower": train_auc_ci_lower,
    "train_auc_ci_upper": train_auc_ci_upper,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "val_auc": val_auc,
    "val_FPR": val_FPR,
    "val_TPR": val_TPR,
    "val_P": val_P,
    "val_f1": val_f1,
    "val_sensitivity": val_sensitivity,
    "val_specificity": val_specificity,
    "val_auc_ci_lower": val_auc_ci_lower,
    "val_auc_ci_upper": val_auc_ci_upper,
    "test_loss": test_loss,
    "test_acc": test_acc,
    "test_auc": test_auc,
    "test_FPR": test_FPR,
    "test_TPR": test_TPR,
    "test_P": test_P,
    "test_f1": test_f1,
    "test_sensitivity": test_sensitivity,
    "test_specificity": test_specificity,
    "test_auc_ci_lower": test_auc_ci_lower,
    "test_auc_ci_upper": test_auc_ci_upper
}

# 将字典转换为 DataFrame
df_save = pd.DataFrame(data)

# 保存到 CSV 文件
df_save.to_csv('direct_classification+duo_ren_wu_val.csv', index=False)


#画图
plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1, epochs+1), test_loss, label='test_loss')
plt.legend()
plt.savefig("direct_classification+duo_ren_wu_val_loss.png")
plt.close()


plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1, epochs+1), test_acc, label='test_acc')
plt.legend()
plt.savefig("direct_classification+duo_ren_wu_val_acc.png")
plt.close()

plt.plot(range(1, epochs+1), train_auc, label='train_auc')
plt.plot(range(1, epochs+1), test_auc, label='test_auc')
plt.legend()
plt.savefig("direct_classification+duo_ren_wu_val_auc.png")
plt.close()


#训练集的ROC曲线
#寻找最优阈值
yuedeng=[train_TPR[-1][i]-train_FPR[-1][i] for i in range(len(train_P[-1]))]
zuiyouyuzhi=train_P[-1][yuedeng.index(max(yuedeng))]

#绘制ROC曲线
plt.plot(train_FPR[-1],train_TPR[-1],'b*-',label='ROC曲线，最优阈值='+str('%.3f'%zuiyouyuzhi)+'，AUC='+str(round(train_auc[-1],3)))
plt.plot([0,1],[0,1],'r--',label='45°参考线')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig("direct_classification+duo_ren_wu_val_last_train_roc.png")
plt.close()


#测试集的ROC曲线
#寻找最优阈值
yuedeng=[test_TPR[-1][i]-test_FPR[-1][i] for i in range(len(test_P[-1]))]
zuiyouyuzhi=test_P[-1][yuedeng.index(max(yuedeng))]

#绘制ROC曲线
plt.plot(test_FPR[-1],test_TPR[-1],'b*-',label='ROC曲线，最优阈值='+str('%.3f'%zuiyouyuzhi)+'，AUC='+str(round(test_auc[-1],3)))
plt.plot([0,1],[0,1],'r--',label='45°参考线')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig("direct_classification+duo_ren_wu_val_last_test_roc.png")
plt.close()


#验证集最好时的测试集的ROC曲线
#寻找最优阈值
yuedeng=[test_TPR[best_epoch][i]-test_FPR[best_epoch][i] for i in range(len(test_P[best_epoch]))]
zuiyouyuzhi=test_P[best_epoch][yuedeng.index(max(yuedeng))]

#绘制ROC曲线
plt.plot(test_FPR[best_epoch],test_TPR[best_epoch],'b*-',label='ROC曲线，最优阈值='+str('%.3f'%zuiyouyuzhi)+'，AUC='+str(round(test_auc[best_epoch],3)))
plt.plot([0,1],[0,1],'r--',label='45°参考线')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig("direct_classification+duo_ren_wu_val_best_test_roc.png")
plt.close()