import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas
from tqdm import tqdm

from scipy import interp
from itertools import cycle


EPOCH = 1
LR = 1e-2
TEST_SIZE = 0.3
BATCH_SIZE = 32
CLASS_NUM = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # (32, 1, 79)
            nn.Conv1d(1, 16, 5),  # (32, 64, 77)
            nn.Tanh(),
            nn.MaxPool1d(3, 3)  # (32, 64, 25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 5),  # (32, 32, 23)
            nn.Tanh(),
            nn.MaxPool1d(3, 3)  # (32, 32, 7)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out = nn.Linear(128, CLASS_NUM)
 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(32, 1, 79)  # 32条数据 1通道 79特征
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.out(x)
        return x


class csvDataset(Dataset):
    def __init__(self, X, y):
        sc = StandardScaler()
        X = sc.fit_transform(X)
        y = np.array(y)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data2(file_name):
    data = pandas.read_csv(file_name)
    label = data.iloc[:, -1]
    data.drop("tagTPL", axis=1, inplace=True)
    data.drop("id", axis=1, inplace=True)
    dataset = csvDataset(X=data, y=label)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


def load_data(file_name):  
    data = pandas.read_csv(file_name)
    label = data.iloc[:, -1]
    data.drop("tagTPL", axis=1, inplace=True)
    data.drop("id", axis=1, inplace=True)
    # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0, stratify=label)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=TEST_SIZE, random_state=0)

    train_dataset = csvDataset(X=X_train, y=y_train)
    test_dataset = csvDataset(X=X_test, y=y_test)
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    conf_matrix = torch.zeros(CLASS_NUM, CLASS_NUM)
    score_list = []     # 存储预测得分
    label_list = []     # 存储真实标签
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            conf_matrix = confusion_matrix(outputs, labels, conf_matrix)
            conf_matrix = conf_matrix.cpu()

            score_list.extend(outputs.detach().cpu().numpy())
            label_list.extend(labels.cpu().numpy())
    conf_matrix = np.array(conf_matrix.cpu())
    corrects = conf_matrix.diagonal(offset=0)
    per_kinds = conf_matrix.sum(axis=1)
    
    # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)),11298))
    # print(conf_matrix)
    # print("每种类别总个数：",per_kinds)
    # print("每种类别预测正确的个数：",corrects)
    # print("每种类别的识别准确率为：{0}".format([rate*100 for rate in corrects/per_kinds]))
    plot_confusion_matrix(conf_matrix, "cm.png")
    Evaluate(conf_matrix)
    
    y_score = np.array(score_list)
    y_test = np.array(label_list)
    calculate_roc(y_test, y_score)
            
    return loss / len(testloader.dataset), correct / total 

# def test(net, testloader):
#     """Validate the model on the test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, total, loss = 0, 0, 0.0
#     with torch.no_grad():
#         for images, labels in tqdm(testloader):
#             outputs = net(images.to(DEVICE))
#             labels = labels.to(DEVICE)
#             loss += criterion(outputs, labels).item()
#             total += labels.size(0)
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     return loss / len(testloader.dataset), correct / total


def confusion_matrix(preds, labels, conf_matrix):
     preds = torch.argmax(preds, 1)
     for p, t in zip(preds, labels):
         conf_matrix[p, t] += 1
     return conf_matrix


def plot_confusion_matrix(cm, savename):
    # classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
    classes = [0, 1, 2, 3, 4]
    plt.figure(figsize=(20, 20))
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = int(cm[y_val][x_val])
        # plt.text(x_val, y_val, c, fontsize=10, va='center', ha='center')
        plt.text(x_val, y_val, c, fontsize=30, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=35, pad=40)

    cbar = plt.colorbar(shrink=0.8)
    cbar.ax.tick_params(labelsize="30")
    # cbar.ax.tick_params(labelsize="15")

    indices = np.array(range(len(classes)))
    # plt.xticks(indices, classes, size=15)
    # plt.yticks(indices, classes, size=15)
    plt.xticks(indices, classes, size=30)
    plt.yticks(indices, classes, size=30)
    plt.tick_params(width=4, length=12)

    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["top"].set_linewidth(4)

    plt.xlabel("Predicted Value", fontsize=30, labelpad=40)
    plt.ylabel("True Value", fontsize=30, labelpad=40)

    plt.savefig(savename, bbox_inches="tight", pad_inches=0.0)


def Evaluate(Cmatrixs):
    """for Precision & Recall"""
    Cmatrixs = torch.tensor(Cmatrixs)
    # classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
    classes = [0, 1, 2, 3, 4]
    n_classes = 5
    Prec, Rec = torch.zeros(n_classes+1), torch.zeros(n_classes+1)

    sum_cmt_row = torch.sum(Cmatrixs,dim=1)#行的和
    sum_cmt_col = torch.sum(Cmatrixs,dim=0)#列的和
    print("----------------------------------------")
    for i in range(n_classes):
        TP = Cmatrixs[i,i]
        FN = sum_cmt_row[i] - TP
        FP = sum_cmt_col[i] - TP
        # TN = torch.sum(Cmatrixs) - sum_cmt_row[i] - FP
        if TP+FP == 0:
            Prec[i] = 0
        else:
            Prec[i] = TP / (TP + FP)
        if TP+FN ==0:
            Rec[i] = 0
        else:
            Rec[i]  = TP / (TP + FN)
    Prec[-1] = torch.mean(Prec[0:-1])
    Rec[-1] = torch.mean(Rec[0:-1])
    print("ALL".ljust(10," "),"Presion={},     Recall={}".format(Prec[-1], Rec[-1]))
    print("F1-Score: ", (2*Prec[-1]*Rec[-1])/(Prec[-1]+Rec[-1]))
    print("----------------------------------------")


def calculate_roc(y_test, y_score):
    # y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(20, 15))
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["top"].set_linewidth(4)
    plt.tick_params(width=4, length=12, labelsize=30)

    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=8)
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=8)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=8,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=30, labelpad=40)
    plt.ylabel('True Positive Rate', fontsize=30, labelpad=40)
    plt.title('Some extension of Receiver operating characteristic to multi-class', fontsize=35, pad=40)
    plt.legend(loc="lower right", fontsize=30)
    plt.savefig("roc.png", bbox_inches="tight", pad_inches=0.0)