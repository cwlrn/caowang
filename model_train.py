import copy
import time
import pandas as pd
import torch
import torch.utils.data as data
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import VGG16  # 修改模型导入路径
import os
from datetime import datetime

# 修改参数
train_data_path = "E:\\pycharm\\cw\\data\\train"  # 训练集的地址
save_model_param_path = "E:\\pycharm\\cw\\VGG\\best_model.pth"  # 保存模型参数的地址
image_size = (224, 224)  # 输入模型的尺寸
split_train_rate, split_val_rate = 0.9, 0.1  # 划分训练集/验证集比例（在训练集细分）
num_epochs = 3  # 训练的轮次，默认为2次
batch_size = 32  # 每批取出的数量
num_workers = 0  # 数据加载的子进程数量，num_workers > 0 表示数据加载将在多个子进程中并行进行，这可以加速数据加载过程。设置为 0 表示数据将在主进程中加载。
learn_rate = 0.001  # 优化器学习率
mean = [0.8268, 0.7396, 0.5916]  # 均值
std = [0.0529, 0.0758, 0.1411]  # 方差（标准差）


def train_val_data_process():
    # 加载数据集
    # train_data = datasets.FashionMNIST("./data_fashionmnist",
    #                                    train=True,
    #                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]),
    #                                    download=True)
    # 预处理操作（变为张量-尺寸固定-归一化）
    pre_train = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(image_size),
                                    transforms.Normalize(mean=mean, std=std)])
    # 加载数据集
    train_data = datasets.ImageFolder(train_data_path, transform=pre_train)
    # 划分训练集80%，验证集20%
    train_data, val_data = data.random_split(train_data, [round(split_train_rate * len(train_data)),
                                                          round(split_val_rate * len(train_data))])
    # 随机取32个训练集图片，32个验证集图片
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 返回训练集/验证集的数据生成器
    return train_dataloader, val_dataloader


# 定义训练过程
def train_model_process(model, train_dataloader, val_dataloader, num_epoch=2):
    # 判断在那种设配训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型放入训练设备中
    model = model.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    # 优化器，初始学习率0.001
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    # 加载模型，做增量训练
    if os.path.exists(save_model_param_path):
        model.load_state_dict(torch.load(save_model_param_path))
        print("模型加载成功，开始增量训练！！！")
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 最高准确度
    best_acc = 0.0
    # 训练集，验证集损失/准确度列表
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    # 当前时间
    since = time.time()

    # 开始训练
    for epoch in range(1, num_epoch + 1):
        # 打印当前轮次
        print("epoch: {}/{}".format(epoch, num_epoch))
        print("_" * 10)
        # 训练集/验证集 损失值/准确度
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        # 训练集/验证集 样本数量
        train_num = 0
        val_num = 0
        # 对每一mini-batch 训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征/标签放入到训练设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)
            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值得作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_correct加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量,获取第一个维度数如torch.Size([64, 1, 28, 28])
            train_num += b_x.size(0)
        # 开始这一轮结束后的，验证集测试
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征/标签放入到训练设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            """
            output==> tensor([[ -4.7900,  -2.5546,  -2.6225,  -0.5127,  -1.2331,   6.5889,  -2.8269,
           6.9545,   0.9755,   2.6147],
        [  2.1169,   6.6870,  -1.9209,   5.8406,   1.0788,  -6.3721,  -0.3663,
          -2.3425,  -4.0790,  -6.3385],...])
            """
            # 获取预测值
            output = model(b_x)
            # pre_lab: tensor([7, 1, 0, 0, 2, 2, 3, 9, 3, 2, 3, 3, 9, 8, 2, 2, 0, 3, 5, 2, 2, 0, 9, 4,5, 3, 9, 2, 0, 3, 4, 7])
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # print("______________")
            # print("output:", output)
            # print("pre_lab:", pre_lab)
            # print("type_pre_lab:", type(pre_lab))
            # print("______________")
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_y.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_y.size(0)
        # 计算并保存每一次迭代的loss和准确率（训练集和验证集）
        train_loss_all.append(train_loss / train_num)
        # train_acc_all.append(train_corrects.double().item() / train_num)
        train_acc_all.append(train_corrects.item() / train_num)

        val_loss_all.append(val_loss / val_num)
        # val_acc_all.append(val_corrects.double().item() / val_num)
        val_acc_all.append(val_corrects.item() / val_num)

        # 打印每一次迭代信息
        print("第{}轮-train loss:{:.4f} train acc:{:.4f}".format(epoch, train_acc_all[-1], train_acc_all[-1]))
        print("第{}轮---val loss:{:.4f}   val acc:{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重
        if val_acc_all[-1] > best_acc:
            # 保存当前的最高准备度
            best_acc = val_acc_all[-1]
            # 保存最高准确度下的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())
    # 训练耗费时间
    time_use = time.time() - since
    print("训练和验证耗费时间:{:.0f}小时{:.0f}分钟{:.0f}秒".format(time_use // 3600, (time_use % 3600) // 60,
                                                                   time_use % 60))

    # 选择最优参数
    # 保存最高准确率下的模型参数
    torch.save(best_model_wts, save_model_param_path)
    # 获取当前时间
    now = datetime.now()
    print("{}年{}月{}日 {:.0f}小时{:.0f}分钟{:.0f}秒".format(now.year, now.month, now.day, now.hour, now.minute,
                                                             now.second))
    print("最优模型参数保存成功！！！")

    # 用表格形式呈现
    train_process = pd.DataFrame(data={
        "epoch": range(num_epoch),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all})
    return train_process


# 可视化数据
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model = VGG16()  # 修改模型
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epoch=num_epochs)
    matplot_acc_loss(train_process)
