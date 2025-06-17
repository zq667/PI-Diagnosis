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

matplotlib.rc("font", family='Noto Sans CJK JP')

# biaoqian = 'zj_biaoqian.xlsx'
# biaoqian2 = 'biaoqian_hq.xlsx'
# biaoqian3 = 'biaoqian_gs.xlsx'
# biaoqian4 = 'biaoqian_yb.xlsx'


file_arg = [['zj_biaoqian.xlsx','zj_binguzuida_processed','zj_jinggujiejie_processed','zj_ruangufugai_processed','zj_shizhuangwei_processed'],
     ['biaoqian_hq.xlsx','hq_binguzuida_processed','hq_jinggujiejie_processed','hq_ruangufugai_processed','hq_shizhuangwei_processed'],
     ['biaoqian_gs.xlsx','gs_binguzuida_processed','gs_jinggujiejie_processed','gs_ruangufugai_processed','gs_shizhuangwei_processed'],
     ['biaoqian_yb.xlsx','yb_binguzuida_processed','yb_jinggujiejie_processed','yb_ruangufugai_processed','yb_shizhuangwei_processed']]

for filename in file_arg:
    print(filename[0])
    print('-' * 30)
    # 读取excel表
    df = pd.read_excel(filename[0], header=None)
    column_data = df.iloc[:, 0].copy()
    for i in range(len(column_data)):
        column_data[i] = unicodedata.normalize("NFKD", column_data[i])
    df.iloc[:, 0] = column_data

    # 文件路径的列表
    folder_path = filename[1]
    folder_path2 = filename[2]
    folder_path3 = filename[3]
    folder_path4 = filename[4]
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
    def xulie_liter(img_paths, img_paths2, img_paths3, img_paths4, name):
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
            if a == b == c == d == name:
                p1.append(img_paths[i])
                p2.append(img_paths2[i])
                p3.append(img_paths3[i])
                p4.append(img_paths4[i])
        return p1, p2, p3, p4


    # img_paths,img_paths2,img_paths3,img_paths4 = xulie_liter(img_paths,img_paths2,img_paths3,img_paths4,'PD-FS')

    # 千万注意要修改列数
    # 通过文件名获取标签
    def getlabel(name):
        goal = df.loc[df[0] == name]
        label = goal.iloc[0, 1]
        return int(label)


    # 分隔文件路径的列表
    img_paths_health = []
    img_paths_sick = []

    for i in img_paths:
        if getlabel(getname(i)) == 0:
            img_paths_health.append(i)
        if getlabel(getname(i)) == 1:
            img_paths_sick.append(i)

    print(len(img_paths_health))
    print(len(img_paths_sick))

    # if filename[0] == 'biaoqian_hq.xlsx':
    #     img_paths_health = img_paths_health[:-55]
    #     print(len(img_paths_health))
    #     print(len(img_paths_sick))
    #
    # if filename[0] == 'biaoqian_gs.xlsx':
    #     img_paths_health = img_paths_health[:-69]
    #     print(len(img_paths_health))
    #     print(len(img_paths_sick))
    #
    # if filename[0] == 'biaoqian_yb.xlsx':
    #     img_paths_health = img_paths_health[:-29]
    #     print(len(img_paths_health))
    #     print(len(img_paths_sick))
    #
    # img_paths = img_paths_health + img_paths_sick


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


    img_paths_test2 = add_elements(img_paths, img_paths2)
    img_paths_test3 = add_elements(img_paths, img_paths3)
    img_paths_test4 = add_elements(img_paths, img_paths4)


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
                scaled_image = np.array(
                    Image.fromarray(scaled_image).resize((int(new_image.shape[1] * scaling_factor[0]),
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
            label = getlabel(getname(img_name))
            return img1, img2, img3, img4, label

        def __len__(self):
            return len(self.img_path)


    test_dataset = MyData(img_paths, img_paths_test2, img_paths_test3, img_paths_test4, test_transform)

    # length 长度
    test_data_size = len(test_dataset)
    # 如果train_data_size=10, 训练数据集的长度为：10
    print("外部测试数据集的长度为：{}".format(test_data_size))

    # 利用 DataLoader 来加载数据集
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


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
            print(merged_feature.shape)

            merged_feature = merged_feature.view(merged_feature.size(0), -1)
            # print(merged_feature.shape)

            output = self.classifier(merged_feature)
            return output


    class FusionNet_Transformer(nn.Module):
        def __init__(self, vgg1, vgg2, vgg3, vgg4):
            super(FusionNet_Transformer, self).__init__()
            self.vgg1 = vgg1
            self.vgg2 = vgg2
            self.vgg3 = vgg3
            self.vgg4 = vgg4

            self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=7)
            self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=7)
            self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=7)
            self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=7)

            # self.max_pool1 = nn.MaxPool2d(kernel_size=(7, 7))
            # self.max_pool2 = nn.MaxPool2d(kernel_size=(7, 7))
            # self.max_pool3 = nn.MaxPool2d(kernel_size=(7, 7))
            # self.max_pool4 = nn.MaxPool2d(kernel_size=(7, 7))

            self.transformer = Transformer(get_transformer_config(256))

            self.classifier = nn.Sequential(
                nn.Linear(1024, 2)
            )

        def forward(self, x1, x2, x3, x4):
            feature1 = self.vgg1(x1)
            feature2 = self.vgg2(x2)
            feature3 = self.vgg3(x3)
            feature4 = self.vgg4(x4)
            # print(feature1.shape)
            feature1 = self.conv1(feature1)
            feature1 = feature1.view(feature1.size(0), 1, -1)
            # feature1 = feature1[:, :, :256]

            feature2 = self.conv2(feature2)
            feature2 = feature2.view(feature2.size(0), 1, -1)
            # feature2 = feature2[:, :, :256]

            feature3 = self.conv3(feature3)
            feature3 = feature3.view(feature3.size(0), 1, -1)
            # feature3 = feature3[:, :, :256]

            feature4 = self.conv4(feature4)
            feature4 = feature4.view(feature4.size(0), 1, -1)
            # feature4 = feature4[:, :, :256]

            merged_feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)  # 在通道维度上拼接特征
            # print(merged_feature.shape)
            merged_feature = self.transformer(merged_feature)
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


    class FusionNet_Transformer_and_Duo_ren_wu_Net(nn.Module):
        def __init__(self, vgg1, vgg2, vgg3, vgg4):
            super(FusionNet_Transformer_and_Duo_ren_wu_Net, self).__init__()
            self.vgg1 = vgg1
            self.vgg2 = vgg2
            self.vgg3 = vgg3
            self.vgg4 = vgg4

            self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=7)
            self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=7)
            self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=7)
            self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=7, stride=7)

            # self.max_pool1 = nn.MaxPool2d(kernel_size=(7, 7))
            # self.max_pool2 = nn.MaxPool2d(kernel_size=(7, 7))
            # self.max_pool3 = nn.MaxPool2d(kernel_size=(7, 7))
            # self.max_pool4 = nn.MaxPool2d(kernel_size=(7, 7))

            self.transformer = Transformer(get_transformer_config(256))

            self.classifier = nn.Sequential(
                nn.Linear(1024, 2)
            )

        def forward(self, x1, x2, x3, x4):
            feature1 = self.vgg1(x1)
            feature2 = self.vgg2(x2)
            feature3 = self.vgg3(x3)
            feature4 = self.vgg4(x4)
            # print(feature1.shape)
            feature1 = self.conv1(feature1)
            feature1 = feature1.view(feature1.size(0), 1, -1)
            # feature1 = feature1[:, :, :256]

            feature2 = self.conv2(feature2)
            feature2 = feature2.view(feature2.size(0), 1, -1)
            # feature2 = feature2[:, :, :256]

            feature3 = self.conv3(feature3)
            feature3 = feature3.view(feature3.size(0), 1, -1)
            # feature3 = feature3[:, :, :256]

            feature4 = self.conv4(feature4)
            feature4 = feature4.view(feature4.size(0), 1, -1)
            # feature4 = feature4[:, :, :256]

            merged_feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)  # 在通道维度上拼接特征
            # print(merged_feature.shape)
            merged_feature = self.transformer(merged_feature)
            # print(merged_feature.shape)
            merged_feature = merged_feature.view(merged_feature.size(0), -1)
            # print(merged_feature.shape)

            output = self.classifier(merged_feature)
            return output


    # a = 'direct_classification.pth'
    # b = 'direct_classification+duo_ren_wu.pth'
    # a = 'direct_classification_val.pth'
    # b = 'direct_classification+duo_ren_wu_val.pth'
    # a = 'direct_classification_transformer_val.pth'
    # b = 'direct_classification_transformer+duo_ren_wu_val.pth'    记得要改网络结构的名称
    model1 = torch.load('direct_classification_val.pth')
    model2 = torch.load('direct_classification+duo_ren_wu_val.pth')

    model_list = [model1,model2]


    # 定义通用的 bootstrap 函数，用于计算指标的 95% 置信区间
    def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, random_state=None):
        rng = np.random.default_rng(random_state)
        metrics = []
        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_true), len(y_true), replace=True)
            bootstrap_true = y_true[indices]
            bootstrap_pred = y_pred[indices]
            metric = metric_fn(bootstrap_true, bootstrap_pred)
            metrics.append(metric)
        metrics = np.array(metrics)
        lower_ci = np.percentile(metrics, 2.5)
        upper_ci = np.percentile(metrics, 97.5)
        return lower_ci, upper_ci


    # 定义敏感度和特异度的计算函数
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


    # 模型评估代码
    for model in model_list:
        # 损失函数
        criterion = nn.CrossEntropyLoss()

        # 利用 GPU
        if torch.cuda.is_available():
            model.to('cuda')


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


        thresholds = np.arange(0.1, 1.0, 0.1)

        model.eval()
        with torch.no_grad():
            all_preds_test = []
            all_labels_test = []
            running_loss_test = 0

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

            # 计算准确率和置信区间
            accuracy = accuracy_score(y_true, y_pred)
            accuracy_ci = bootstrap_metric_ci(y_true, y_pred, accuracy_score)

            # 计算 F1 分数和置信区间
            f1 = f1_score(y_true, y_pred)
            f1_ci = bootstrap_metric_ci(y_true, y_pred, f1_score)

            # 计算敏感度和置信区间
            sensitivity = sensitivity_score(y_true, y_pred)
            sensitivity_ci = bootstrap_metric_ci(y_true, y_pred, sensitivity_score)

            # 计算特异度和置信区间
            specificity = specificity_score(y_true, y_pred)
            specificity_ci = bootstrap_metric_ci(y_true, y_pred, specificity_score)

            # 计算 AUC 和置信区间
            auc = roc_auc_score(y_true, all_preds_test[:, 1].cpu().numpy())
            auc_ci_lower, auc_ci_upper = bootstrap_auc_ci(
                y_true, all_preds_test[:, 1].cpu().numpy()
            )

            # # 各阈值的计算
            # results = {}
            # for threshold in thresholds:
            #     preds = (all_preds_test[:, 1] >= threshold).long()
            #     accuracy_th = accuracy_score(y_true, preds.cpu().numpy())
            #     f1_th = f1_score(y_true, preds.cpu().numpy())
            #     sensitivity_th = sensitivity_score(y_true, preds.cpu().numpy())
            #     specificity_th = specificity_score(y_true, preds.cpu().numpy())
            #
            #     # 计算阈值下的置信区间
            #     accuracy_ci_th = bootstrap_metric_ci(y_true, preds.cpu().numpy(), accuracy_score)
            #     f1_ci_th = bootstrap_metric_ci(y_true, preds.cpu().numpy(), f1_score)
            #     sensitivity_ci_th = bootstrap_metric_ci(y_true, preds.cpu().numpy(), sensitivity_score)
            #     specificity_ci_th = bootstrap_metric_ci(y_true, preds.cpu().numpy(), specificity_score)
            #
            #     results[threshold] = {
            #         'accuracy_th': accuracy_th,
            #         'f1_score_th': f1_th,
            #         'sensitivity_th': sensitivity_th,
            #         'specificity_th': specificity_th,
            #         'accuracy_ci_th': accuracy_ci_th,
            #         'f1_ci_th': f1_ci_th,
            #         'sensitivity_ci_th': sensitivity_ci_th,
            #         'specificity_ci_th': specificity_ci_th,
            #     }

            # 输出总体指标
            print(f'Test Loss: {round(epoch_loss_test, 3)}')
            print(f'Accuracy: {round(accuracy, 3)} (95% CI: [{round(accuracy_ci[0], 3)} - {round(accuracy_ci[1], 3)}])')
            print(f'F1 Score: {round(f1, 3)} (95% CI: [{round(f1_ci[0], 3)} - {round(f1_ci[1], 3)}])')
            print(
                f'Sensitivity: {round(sensitivity, 3)} (95% CI: [{round(sensitivity_ci[0], 3)} - {round(sensitivity_ci[1], 3)}])')
            print(
                f'Specificity: {round(specificity, 3)} (95% CI: [{round(specificity_ci[0], 3)} - {round(specificity_ci[1], 3)}])')
            print(f'AUC: {round(auc, 3)} (95% CI: [{round(auc_ci_lower, 3)} - {round(auc_ci_upper, 3)}])')
            print('-' * 50)

            # # 输出各阈值结果
            # for threshold, metrics in results.items():
            #     print(f'Threshold: {threshold:.2f}')
            #     print(
            #         f"Accuracy_th: {metrics['accuracy_th']:.4f} (95% CI: [{metrics['accuracy_ci_th'][0]:.4f} - {metrics['accuracy_ci_th'][1]:.4f}])")
            #     print(
            #         f"F1 Score_th: {metrics['f1_score_th']:.4f} (95% CI: [{metrics['f1_ci_th'][0]:.4f} - {metrics['f1_ci_th'][1]:.4f}])")
            #     print(
            #         f"Sensitivity_th: {metrics['sensitivity_th']:.4f} (95% CI: [{metrics['sensitivity_ci_th'][0]:.4f} - {metrics['sensitivity_ci_th'][1]:.4f}])")
            #     print(
            #         f"Specificity_th: {metrics['specificity_th']:.4f} (95% CI: [{metrics['specificity_ci_th'][0]:.4f} - {metrics['specificity_ci_th'][1]:.4f}])")
            #     print('-' * 30)


