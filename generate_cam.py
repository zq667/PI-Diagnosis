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
from Transformer import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor


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

print(len(img_paths_health_test))
print(len(img_paths_sick_test))

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

    def forward(self, x1,x2,x3,x4):
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




# 加载预训练好的模型并加载到GPU
model1 = torch.load('direct_classification_val.pth')
model2 = torch.load('direct_classification+duo_ren_wu_val.pth')

model_list = [model2]




class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al.
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''

    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers

        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_backward_hook(self.backward_hook)

        self.activations = []
        self.grads = []

    def forward_hook(self, module, input, output):
        self.activations.append(output[0])

    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())

    def calculate_cam(self, x1,x2,x3,x4):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)  # Module.to() is in-place method
            x1,x2,x3,x4 = x1.to(device),x2.to(device),x3.to(device),x4.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()

        # forward
        y_hat = self.model(x1,x2,x3,x4)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)

        # backward
        model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()

        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()

        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()
        return cam

    @staticmethod
    def show_cam_on_image(image, cam, name, shitu_num, model_num):
        # image: [H,W,C]
        h, w = image.shape[:2]

        cam = cv2.resize(cam, (h, w))
        cam = cam / cam.max()
        heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)  # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        image = image / image.max()
        heatmap = heatmap / heatmap.max()

        result = 0.4 * heatmap + 0.6 * image
        result = result / result.max()

        # 确保保存路径存在
        save_dir = 'relitu3'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 构建保存文件名
        name, extension = os.path.splitext(name)
        save_name = f"{name}_{model_num}_{shitu_num}.png"
        save_path = os.path.join(save_dir, save_name)

        plt.figure()
        plt.imshow((result * 255).astype(np.uint8))
        plt.colorbar(shrink=0.8)
        plt.tight_layout()

        plt.savefig(save_path)  # 先保存再显示
        # plt.show()

        print(f"Saved CAM image at: {save_path}")

    @staticmethod
    def preprocess_image(name1, name2, name3, name4, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        preprocessing = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
        img1 = read_image(folder_path, name1)
        img2 = read_image(folder_path2, name2)
        img3 = read_image(folder_path3, name3)
        img4 = read_image(folder_path4, name4)
        return preprocessing(img1.copy()).unsqueeze(0),preprocessing(img2.copy()).unsqueeze(0),\
            preprocessing(img3.copy()).unsqueeze(0),preprocessing(img4.copy()).unsqueeze(0)


if __name__ == '__main__':
    for num in range(len(model_list)):
        model = model_list[num]
        for index in range(len(img_paths_train)):
            # print(torch.__version__)
            # print(img_paths_train[index])
            # print(img_paths_train2[index])
            # print(img_paths_train3[index])
            # print(img_paths_train4[index])
            # print(getlabel(getname(img_paths_train[index])))
            # print(len(img_paths_train))
            # for i in img_paths_train:
            #     print(getlabel(getname(i)))
            image1 = cv2.imread(os.path.join(folder_path, img_paths_train[index]))  # (224,224,3)
            image2 = cv2.imread(os.path.join(folder_path2, img_paths_train2[index]))
            image3 = cv2.imread(os.path.join(folder_path3, img_paths_train3[index]))
            image4 = cv2.imread(os.path.join(folder_path4, img_paths_train4[index]))
            x1,x2,x3,x4 = GradCAM.preprocess_image(img_paths_train[index],img_paths_train2[index],img_paths_train3[index],img_paths_train4[index])
            # print(model.vgg1[0][28])
            grad_cam1 = GradCAM(model, model.vgg1[0][28], 224)
            cam1 = grad_cam1.calculate_cam(x1, x2, x3, x4)
            GradCAM.show_cam_on_image(image1, cam1, img_paths_train[index], 1, num)

            grad_cam2 = GradCAM(model, model.vgg2[0][28], 224)
            cam2 = grad_cam2.calculate_cam(x1, x2, x3, x4)
            GradCAM.show_cam_on_image(image2, cam2, img_paths_train2[index], 2, num)

            grad_cam3 = GradCAM(model, model.vgg3[0][28], 224)
            cam3 = grad_cam3.calculate_cam(x1, x2, x3, x4)
            GradCAM.show_cam_on_image(image3, cam3, img_paths_train3[index], 3, num)

            grad_cam4 = GradCAM(model, model.vgg4[0][28], 224)
            cam4 = grad_cam4.calculate_cam(x1, x2, x3, x4)
            GradCAM.show_cam_on_image(image4, cam4, img_paths_train4[index], 4, num)

