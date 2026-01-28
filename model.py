import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import matplotlib.pyplot as plt
from IPython import display
import time
import cv2
import timm

BASEDIR = Path(__file__).resolve().parent

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Visualizer:
    def __init__(self, num_epochs, xlabel='epoch', title_ls='loss', title_acc='accuracy', figsize=(10, 4)):
        self.fig, self.axes = plt.subplots(1, 2, figsize=figsize) # 创建1行2列的子图
        self.xlabel = xlabel
        self.num_epochs = num_epochs
        self.titles = [title_ls, title_acc]

        self.train_x = []
        self.valid_x = []

        self.y_train_ls = []
        self.y_valid_ls = []
        self.y_train_acc = []
        self.y_valid_acc = [] 
    
    def add(self, x, y_train_ls, y_valid_ls, y_train_acc, y_valid_acc):
        self.train_x.append(x)
        self.y_train_ls.append(y_train_ls)
        self.y_train_acc.append(y_train_acc)

        # 只有传入不是None时才添加Valid数据
        if y_valid_ls is not None:
            self.valid_x.append(x)
            self.y_valid_ls.append(y_valid_ls)
            self.y_valid_acc.append(y_valid_acc)

        # 清除之前的图像
        self.axes[0].cla()
        self.axes[1].cla()

        # 绘制左图
        self.axes[0].plot(self.train_x, self.y_train_ls, label='train loss')
        if self.valid_x:
            self.axes[0].plot(self.valid_x, self.y_valid_ls, label='valid loss')

        # 绘制右图
        self.axes[1].plot(self.train_x, self.y_train_acc, label='train accuracy')
        if self.valid_x:
            self.axes[1].plot(self.valid_x, self.y_valid_acc, label='valid accuracy')

        # 设置布局
        for i, ax in enumerate(self.axes):
            ax.set_xlabel(self.xlabel)
            ax.set_xlim(1, self.num_epochs)
            ax.set_title(self.titles[i])
            ax.legend()
        self.axes[1].set_ylim(0, 1)

        # 动态刷新
        display.clear_output(wait=True)
        display.display(self.fig)
    
    def close(self):
        plt.close(self.fig)

class LeafDataset(Dataset):
    def __init__(self, csv_file, dataset_dir, transform=None, train=False, valid=False, final_train=False):
        self.data_csv = pd.read_csv(csv_file)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.train = train
        self.valid = valid
        self.final_train = final_train
        if self.train:
            # 将字符串标签映射为整数
            self.label_mapping = {label: idx for idx, label in enumerate(self.data_csv['label'].unique())}
            self.data_csv['label'] = self.data_csv['label'].map(self.label_mapping)
            # 前80%作为训练集，后20%作为验证集
            train_size = int(0.8 * len(self.data_csv))
            if self.valid:
                # 验证集取后20%的数据
                self.data_csv = self.data_csv.iloc[train_size:].reset_index(drop=True)
            elif not self.final_train:
                # 训练集取前80%的数据
                self.data_csv = self.data_csv.iloc[:train_size].reset_index(drop=True)

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.data_csv.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.train:
            label = self.data_csv.iloc[idx, 1]
            return image, label

        return image

    def get_label_mapping(self): # 整数和字符串标签的映射字典
        if self.train or self.final_train:
            return self.label_mapping
        else:
            raise ValueError("Label mapping is only available for training dataset.")

def load_data_leaves(batch_size, transform_train, transform_test, dataset_dir):
    train_dataset = LeafDataset(csv_file=dataset_dir/'train.csv',
                            dataset_dir=dataset_dir,
                            transform=transform_train, train=True, valid=False)
    valid_dataset = LeafDataset(csv_file=dataset_dir/'train.csv',
                            dataset_dir=dataset_dir,
                            transform=transform_test, train=True, valid=True)
    final_train_dataset = LeafDataset(csv_file=dataset_dir/'train.csv',
                            dataset_dir=dataset_dir,
                            transform=transform_train, train=True, final_train=True)
    test_dataset = LeafDataset(csv_file=dataset_dir/'test.csv',
                            dataset_dir=dataset_dir,
                            transform=transform_test, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4)
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4)
    idx_mapping = {v: k for k, v in train_dataset.get_label_mapping().items()} # 用于测试时将预测的整数标签转换回字符串标签
    
    return train_loader, valid_loader, final_train_loader, (test_loader, idx_mapping, test_dataset.data_csv)

def finetune_resnet_init_fc():
    """将resnet50最后一个fc层初始化"""
    finetune_net = timm.create_model('resnet50', pretrained=True) # 使用预训练模型
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 176) # 通道数2048->176
    
    nn.init.kaiming_normal_(finetune_net.fc.weight) # 只初始化fc层
    nn.init.zeros_(finetune_net.fc.bias)
    return finetune_net, ["fc.weight", "fc.bias"] # 这些参数是1倍学习率，其他参数0.1倍学习率

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def validate_gpu(net, valid_loader, devices):
    net.eval()
    metric = Accumulator(3) # 验证损失总和、正确预测数、样本总数
    loss = nn.CrossEntropyLoss()
    for i, (X, y) in enumerate(valid_loader):
        X, y = X.to(devices[0]), y.to(devices[0])
        y_hat = net(X)
        l = loss(y_hat, y)
        metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
    
    return metric[0] / metric[2], metric[1] / metric[2]

def train(train_loader, valid_loader, net, head_names, lr, num_epochs, weight_decay, finetune=False):    
    devices = [try_gpu(i) for i in range(torch.cuda.device_count())]
    net.to(devices[0]) # 将模型搬到主GPU上

    if finetune:
        base_params, head_params = [], []
        for name, param in net.named_parameters():
            if name in head_names:
                head_params.append(param)
            else:
                base_params.append(param)
        optimizer = torch.optim.Adam([{'params':base_params, 'lr': lr*0.1}, # 顶部层以外都采用0.1倍的学习率
                                      {'params':head_params, 'lr': lr}],
                                      weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # 数据并行，准备将小批量划分到各个GPU中并行训练，反向传播后自动将各GPU中的梯度相加并分发
    net = nn.DataParallel(net, device_ids=devices)

    # cosine退火学习率，共num_epochs个迭代步数，每个周期更新一次，从余弦的峰顶降到谷底，最低学习率为1e-6
    sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss = nn.CrossEntropyLoss()
    
    viz = Visualizer(num_epochs)

    timer, num_batches = Timer(), len(train_loader)
    for epoch in range(num_epochs):
        metric = Accumulator(3) # 训练损失总和、训练准确数总和、样本总数
        net.train()
        for i, (X, y) in enumerate(train_loader):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0]) # 将数据移动到主GPU上
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] /metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                viz.add(epoch + (i + 1) / num_batches, train_l, None, train_acc, None)
        sheduler.step() # 更新学习率
        if valid_loader is not None:
            valid_l, valid_acc = validate_gpu(net, valid_loader, devices)
            viz.add(epoch + 1, train_l, valid_l, train_acc, valid_acc)
    
    print(f'train loss {train_l:.3f}, train acc {train_acc:.3f}')
    if valid_loader is not None:
        print(f'valid loss {valid_l:.3f}, valid acc {valid_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec')

def final_train_and_predict(models, final_train_loader, test_loader, idx_mapping, img_paths):
    devices = [try_gpu(i) for i in range(torch.cuda.device_count())]

    for i in range(len(models)):
        train(final_train_loader, None, models[i][0], models[i][1],
               lr, num_epochs, weight_decay, finetune=True)
    
    for i in range(len(models)):
        models[i][0].eval()

    predictions = []
    with torch.no_grad():
        for X in test_loader:
            X = X.to(devices[0])
            # 模型输出形状（批量大小，分类数）
            all_probs = [torch.softmax(model[0](X), dim=1) for model in models]
            # stack之后形状为（模型数，批量大小，分类数）
            # mean之后形状回到（批量大小，分类数）
            avg_probs = torch.stack(all_probs).mean(dim=0)
            preds = torch.argmax(avg_probs, dim=1)
            predictions.extend(preds.cpu().numpy())

    predictions = pd.Series(predictions).map(idx_mapping) # 整数标签转换回字符串标签

    submit = pd.concat([img_paths['image'], predictions.rename('label')], axis=1)
    submit.to_csv(BASEDIR/'submission.csv', index=False)
    print("Submission file 'submission.csv' created.")

if __name__ == "__main__":

    batch_size, lr, num_epochs, weight_decay = 64, 0.001, 150, 1e-5

    transform_train = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomResizedCrop((224, 224), scale=(0.6, 1.0)),
        A.Rotate(),
        A.RandomBrightnessContrast(),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]), # 使用 ImageNet 数据集的均值和标准差
        ToTensorV2()
    ])
    transform_test = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_loader, valid_loader, final_train_loader, test_data = load_data_leaves(batch_size, 
                                                transform_train, transform_test,
                                                dataset_dir=BASEDIR/'dataset')

    # net = finetune_resnet_init_fc()
    # train(train_loader, valid_loader, net, head_names, lr, num_epochs, weight_decay, 
    #       finetune=True)

    models = [finetune_resnet_init_fc() for _ in range(2)] # 2个不同初始化的模型做模型集成

    final_train_and_predict(models, final_train_loader, *test_data)