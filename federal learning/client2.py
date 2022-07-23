import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import torch
import flwr as fl

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(1)

from utils import Net, load_data2, train, test, DEVICE, LR, BATCH_SIZE, CLASS_NUM, confusion_matrix, plot_confusion_matrix, Evaluate, calculate_roc, EPOCH

net = Net().to(DEVICE)
trainloader = load_data2("dataset2_int.csv")
testloader = load_data2("dataset3_int.csv")
# trainloader, testloader = load_data(file_name1)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=EPOCH)
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client("[::]:9999", client=FlowerClient())