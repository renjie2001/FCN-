import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
import numpy as np
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import random
from torchvision.utils import save_image
from torch.nn import functional as F

path = r'test/horse021.jpg'  # 测试图片路径
model = r'checkpoints/fcn_model_165.pth'  # 模型路径
crop_size = 160
num_classes = 2


# 将标记图（每个像素值代该位置像素点的类别）转换为onehot编码
def onehot(data, n):
    buf = np.zeros(data.shape + (n,))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk - 1] = 1
    return buf


# 利用torchvision提供的transform，定义原始图片的预处理步骤（转换为tensor和标准化处理）
def dataprocess():
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return preprocess


# 利用torch提供的Dataset类，定义数据集
class CreatDataset(Dataset):
    def __init__(self, preprocess=None):
        self.preprocess = preprocess

    def __len__(self):
        return len(os.listdir('./data'))  # 导入数据目录

    def __getitem__(self, idx):  # 导入数据
        img_name = os.listdir('./data')[idx]
        raw_img = cv2.imread('./data/' + img_name)
        raw_img = cv2.resize(raw_img, (160, 160))  # 固定图像大小
        mask_img = cv2.imread('./data_msk/' + img_name, 0)
        mask_img = cv2.resize(mask_img, (160, 160))  # 固定图像大小
        mask_img = mask_img / 255  # 均一化
        mask_img = mask_img.astype('uint8')  # 转unit8格式
        mask_img = onehot(mask_img, 2)  # 转为独热码
        mask_img = mask_img.transpose(2, 0, 1)  # 将通道数提前
        mask_img = torch.FloatTensor(mask_img)
        # print(mask_img.shape)
        if self.preprocess:
            raw_img = self.preprocess(raw_img)  # 图像预处理

        return raw_img, mask_img


# FCN模型构建
# in_channels：输入通道数
# classes：语义分割的分类数
class FCNNet(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.conv_1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.act_1_1 = nn.ReLU(True)

        self.conv_1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.act_1_2 = nn.ReLU(True)

        self.pool_1_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/2

        self.conv_2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.act_2_1 = nn.ReLU(True)

        self.conv_2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.act_2_2 = nn.ReLU(True)
        self.pool_2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/4

        self.conv_3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.act_3_1 = nn.ReLU(True)
        self.conv_3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.act_3_2 = nn.ReLU(True)
        self.pool_3_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/8

        self.conv_4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.act_4_1 = nn.ReLU(True)
        self.conv_4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.act_4_2 = nn.ReLU(True)
        self.pool_4_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/16

        self.conv_5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.act_5_1 = nn.ReLU(True)
        self.conv_5_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.act_5_2 = nn.ReLU(True)
        self.pool_5_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/32

        self.de_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                           output_padding=1, bias=False)  # 1/16
        self.BN1 = nn.BatchNorm2d(128)

        self.de_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                           output_padding=1, bias=False)  # 1/8
        self.BN2 = nn.BatchNorm2d(64)

        self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                           output_padding=1, bias=False)  # 1/4
        self.BN3 = nn.BatchNorm2d(32)

        self.de_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1,
                                           output_padding=1, bias=False)  # 1/2
        self.BN4 = nn.BatchNorm2d(16)

        self.de_conv5 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1,
                                           output_padding=1, bias=False)  # 1
        self.BN5 = nn.BatchNorm2d(16)

        self.act = nn.ReLU(True)
        self.output = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=3, padding=1)

    def forward(self, x):  # x：[B C H W]
        x = self.act_1_1(self.conv_1_1(x))  # x: [B 16 H W]
        x = self.act_1_2(self.conv_1_2(x))  # x: [B 16 H W]
        x = self.pool_1_1(x)  # x: [B 16 H/2 W/2]

        x = self.act_2_1(self.conv_2_1(x))  # x: [B 32 H/2 W/2]
        x = self.act_2_2(self.conv_2_2(x))  # x: [B 32 H/2 W/2]
        x = self.pool_2_1(x)  # x: [B 32 H/4 W/4]

        x = self.act_3_1(self.conv_3_1(x))  # x: [B 64 H/4 W/4]
        x = self.act_3_2(self.conv_3_2(x))  # x: [B 64 H/4 W/4]
        x = self.pool_3_1(x)  # x: [B 64 H/8 W/8]

        x = self.act_4_1(self.conv_4_1(x))  # x: [B 128 H/8 W/8]
        x = self.act_4_2(self.conv_4_2(x))  # x: [B 128 H/8 W/8]
        x = self.pool_4_1(x)  # x: [B 128 H/16 W/16]

        x_ = x  # 用于反卷积

        x = self.act_5_1(self.conv_5_1(x))  # x: [B 256 H/16 W/16]
        x = self.act_5_2(self.conv_5_2(x))  # x: [B 256 H/16 W/16]
        x = self.pool_5_1(x)  # x: [B 256 H/32 W/32]

        x = self.act(self.de_conv1(x))  # x: [B 128 H/16 W/16]
        x = self.BN1(x + x_)  # 转置卷积     # x: [B 128 H/16 W/16]

        x = self.act(self.de_conv2(x))  # x: [B 64 H/8 W/8]
        x = self.BN2(x)  # 转置卷积          # x: [B 64 H/8 W/8]

        x = self.act(self.de_conv3(x))  # x: [B 32 H/4 W/4]
        x = self.BN3(x)  # 转置卷积          # x: [B 32 H/4 W/4]

        x = self.act(self.de_conv4(x))  # x: [B 16 H/2 W/2]
        x = self.BN4(x)  # 转置卷积          # x: [B 16 H/2 W/2]

        x = self.act(self.de_conv5(x))  # x: [B 16 H W]
        x = self.BN5(x)  # 转置卷积          # x: [B 16 H W]

        out = self.output(x)  # x: [B 3 H W]

        return out


# <---------------------------------------------->
# 下面开始训练网络

# 在训练网络前定义函数用于计算Acc 和 mIou
# 计算混淆矩阵
def cal_confusion_matrix(label_true, label_pred, classes):
    mask = (label_true >= 0) & (label_true < classes)
    matrix = np.bincount(
        classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=classes ** 2).reshape(classes, classes)
    return matrix


# 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, classes):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    matrix = np.zeros((classes, classes))
    for lt, lp in zip(label_trues, label_preds):
        matrix += cal_confusion_matrix(lt.flatten(), lp.flatten(), classes)
    acc = np.diag(matrix).sum() / matrix.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(matrix) / matrix.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(matrix) / (
                matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
        )
    mean_iu = np.nanmean(iu)
    freq = matrix.sum(axis=1) / matrix.sum()
    return acc, acc_cls, mean_iu


def train(epo_num=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cpu和gpu的选择
    fcn_model = FCNNet(in_channels=3, classes=2)  # 调用模型
    fcn_model = fcn_model.to(device)

    # 这里只有两类，采用二分类常用的损失函数BCE
    criterion = nn.BCELoss().to(device)
    # 随机梯度下降优化，学习率0.0001，惯性分数0.7
    optimizer = optim.SGD(fcn_model.parameters(), lr=0.0005, momentum=0.7)

    # 记录训练过程相关指标
    all_train_iter_loss = []
    all_val_iter_loss = []
    val_Acc = []
    val_mIou = []
    # start timing
    prev_time = datetime.now()

    for epo in range(epo_num):
        # 训练
        train_loss = 0
        fcn_model.train()
        for index, (data, data_msk) in enumerate(train_dataloader):
            data = data.to(device)
            data_msk = data_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(data)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, data_msk)
            loss.backward()  # 需要计算导数，则调用backward
            iter_loss = loss.item()  # .item()返回一个具体的值，一般用于loss和acc
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            data_msk_np = data_msk.cpu().detach().numpy().copy()
            data_msk_np = np.argmin(data_msk_np, axis=1)

            # 每15个bacth，输出一次训练过程的数据
            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))

        # 验证
        val_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (data, data_msk) in enumerate(val_dataloader):
                data = data.to(device)
                data_msk = data_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(data)
                output = torch.sigmoid(output)  # ([4, 2, 160, 160])
                loss = criterion(output, data_msk)
                iter_loss = loss.item()
                all_val_iter_loss.append(iter_loss)
                val_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()
                output_np = np.argmin(output_np, axis=1)
                data_msk_np = data_msk.cpu().detach().numpy().copy()
                data_msk_np = np.argmin(data_msk_np, axis=1)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('<---------------------------------------------------->')
        print('epoch: %f' % epo)
        print('epoch train loss = %f, epoch val loss = %f, %s'
              % (train_loss / len(train_dataloader), val_loss / len(val_dataloader), time_str))

        acc, acc_cls, mean_iu = label_accuracy_score(data_msk_np, output_np, 2)
        val_Acc.append(acc)
        val_mIou.append(mean_iu)

        print('Acc = %f, mIou = %f' % (acc, mean_iu))
        # 每5个epoch存储一次模型
        if np.mod(epo, 5) == 0:
            # 只存储模型参数
            # torch.save(fcn_model.state_dict(), 'checkpoints/fcn_model_{}.pth'.format(epo))
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pth'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pth'.format(epo))

    # 绘制训练过程数据
    plt.figure()
    plt.subplot(221)
    plt.title('train_loss')
    plt.plot(all_train_iter_loss)
    plt.xlabel('batch')
    plt.subplot(222)
    plt.title('val_loss')
    plt.plot(all_val_iter_loss)
    plt.xlabel('batch')
    plt.subplot(223)
    plt.title('val_Acc')
    plt.plot(val_Acc)
    plt.xlabel('epoch')
    plt.subplot(224)
    plt.title('val_mIou')
    plt.plot(val_mIou)
    plt.xlabel('epoch')
    plt.show()


full_to_train = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3,
                 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                 26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}
train_to_full = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26,
                 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 19: 0}
full_to_colour = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70), 12: (102, 102, 156),
                  13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0), 21: (107, 142, 35),
                  22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142),
                  27: (0, 0, 70), 28: (0, 60, 100), 31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}


# 模型测试
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(model)
    net = net.to(device)
    net.eval()
    input = Image.open(path)
    w, h = input.size
    x1, y1 = random.randint(0, w - crop_size), random.randint(0, h - crop_size)
    input = input.crop((x1, y1, x1 + crop_size, y1 + crop_size))

    w, h = input.size
    input = torch.ByteTensor(torch.ByteStorage.from_buffer(input.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(
        255)

    input[0].add_(-0.485).div_(0.229)
    input[1].add_(-0.456).div_(0.224)
    input[2].add_(-0.406).div_(0.225)
    save_image(input, r'./test1.png')
    input = input.to(device)
    input = input.unsqueeze(0)
    output = F.log_softmax(net(input))
    b, _, h, w = output.size()
    pred = output.permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, h, w)
    pred = pred.data.cpu()
    pred_remapped = pred.clone()

    for k, v in train_to_full.items():
        pred_remapped[pred == k] = v
    pred = pred_remapped
    pred_colour = torch.zeros(b, 3, h, w)
    for k, v in full_to_colour.items():
        pred_r = torch.zeros(b, 1, h, w)
        pred = pred.reshape(1, 1, h, -1)
        pred_r[(pred == k)] = v[0]
        pred_g = torch.zeros(b, 1, h, w)
        pred_g[(pred == k)] = v[1]
        pred_b = torch.zeros(b, 1, h, w)
        pred_b[(pred == k)] = v[2]
        pred_colour.add_(torch.cat((pred_r, pred_g, pred_b), 1))
    print(pred_colour[0].float())
    print('-----------------')
    pred = pred_colour[0].float().div(255)
    save_image(pred, r'./test2.png')


if __name__ == "__main__":
    # 实例化数据集
    preprocess = dataprocess()
    data = CreatDataset(preprocess)

    train_size = int(0.85 * len(data))  # 85%的样本用于训练
    val_size = len(data) - train_size  # 15%的样本用于验证
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # 利用DataLoader生成一个分batch获取数据的可迭代对象
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

    # train(epo_num=200) #模型训练
    test()  # 模型测试
