import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(1)

from utils import Net, load_data2, DEVICE, LR, BATCH_SIZE, CLASS_NUM, confusion_matrix, plot_confusion_matrix, Evaluate, calculate_roc

EPOCH = 200

train_loader = load_data2("dataset1_int.csv")
test_loader = load_data2("dataset3_int.csv")

cnn = Net().to(DEVICE)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=BATCH_SIZE, gamma=0.1)


def train(num_epochs, _model, _device, _train_loader, _optimizer, _lr_scheduler):
    _model.train()  # 设置模型为训练模式
    _lr_scheduler.step()  # 设置学习率调度器开始准备更新
    for step, (images, labels) in enumerate(train_loader):
        samples = images.to(_device)
        labels = labels.to(_device)
        output = cnn(samples)
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 30 == 0:
            print("Epoch:{}, step:{}, loss:{:.6f}".format(num_epochs, step, loss.item()))


def test(_test_loader, _model, _device, epoch):
    _model.eval()  # 设置模型进入预测模式 evaluation
    conf_matrix = torch.zeros(CLASS_NUM, CLASS_NUM)
    loss, correct = 0, 0
    score_list = []     # 存储预测得分
    label_list = []     # 存储真实标签

    # 不计算梯度，节约显存
    with torch.no_grad():
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            test_output = cnn(data)
            loss += loss_function(test_output, target).item()  # 添加损失值
            pred = test_output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # .cpu()将参数迁移到cpu
            
            conf_matrix = confusion_matrix(test_output, target, conf_matrix)
            conf_matrix = conf_matrix.cpu()

            score_list.extend(test_output.detach().cpu().numpy())
            label_list.extend(target.cpu().numpy())

        loss /= len(_test_loader.dataset)

        print('\nAverage loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            loss, correct, len(_test_loader.dataset),
            100. * correct / len(_test_loader.dataset)))

        if epoch == EPOCH - 1:
            conf_matrix = np.array(conf_matrix.cpu())
            corrects = conf_matrix.diagonal(offset=0)
            per_kinds = conf_matrix.sum(axis=1)
            
            # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)),1829))
            # print(conf_matrix)
            # print("每种类别总个数：",per_kinds)
            # print("每种类别预测正确的个数：",corrects)
            # print("每种类别的识别准确率为：{0}".format([rate*100 for rate in corrects/per_kinds]))
            plot_confusion_matrix(conf_matrix, "cm.png")
            Evaluate(conf_matrix)
            
            y_score = np.array(score_list)
            y_test = np.array(label_list)
            calculate_roc(y_test, y_score)

        
if __name__ == '__main__':
    for epoch in range(EPOCH):
        train(epoch, cnn, DEVICE, train_loader, optimizer, exp_lr_scheduler)
        test(test_loader, cnn, DEVICE, epoch)
    # torch.save(cnn, "CNN.pth")

    