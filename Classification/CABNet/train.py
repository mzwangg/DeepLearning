import os
import sys
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from MobileNetV3 import mobilenet_v3_large
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


# 对于一些参数的设置
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH = 70
BATCH_SIZE = 16
LR = 0.001

#focal loss损失函数
class focal_loss(nn.Module):
    def __init__(self, alpha, gamma=2, num_classes=5):
        super(focal_loss, self).__init__()
        assert len(alpha) == num_classes

        #alpha为权重参数向量，gamma为(1-p)的次方数
        self.alpha = torch.Tensor(alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)

        #用preds的softmax作为各类别的概率，因softmax之后值的和为1
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))

        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        loss = loss.mean()
        return loss


def main():
    # 对数据的预处理
    data_transform = {'train': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 读取数据并预处理
    image_datasets = {x: datasets.ImageFolder(root='./DDR/' + x,
                                              transform=data_transform[x]) for x in ['train', 'valid']}

    # 读取完数据后，对数据进行装载
    # num_worker表示加载图片的进程数，通过计算选择适合的num_worker数
    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
    dataloader = {'train': DataLoader(dataset=image_datasets['train'],
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=nw),
                  'valid': DataLoader(dataset=image_datasets['valid'],
                                      batch_size=BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=nw)}

    # 输出提示信息
    print("Using device", device)
    print('Using {} dataloader workers every process'.format(nw))
    print("using {} images for training, {} images for validation.".format(
        len(image_datasets['train']),
        len(image_datasets['valid'])))

    # 加载预训练权重
    # 因为原始权值是在类别数为1000的条件下训练的，所以要删除最后一个全连接层权重
    net = mobilenet_v3_large(5)
    weights = torch.load('mobilenet_v3_large.pth', map_location=device)
    for key in list(weights.keys()):
        if 'classifier' in key:
            del (weights[key])
    net.load_state_dict(weights, strict=False)
    net.to(device)

    loss_func = focal_loss(torch.tensor([9861/2992,9861/1638,9861/2238,9861/613,9861/2370]))
    optimizer = optim.Adam(net.parameters(), lr=LR)

    # 如果损失在两轮训练之后仍没下降，则将学习率变为原来的0.8倍
    # 防止模型在最好值的附近来回波动而不能到达最好值
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2)

    max_acc = 0  # 记录最大正确率，以找到最好模型
    kappa = 0  # 记录最大正确率模型的kappa值
    best_stata_dict = {}  # 存放最好模型的参数

    # 为了最后的绘图存储数据
    train_arr = {'loss': [], 'acc': []}
    valid_arr = {'loss': [], 'acc': []}

    # 开始训练
    for epoch in range(EPOCH):
        # 训练
        print('epoch {}/{}'.format(epoch + 1, EPOCH))
        print("LR={}".format(optimizer.param_groups[0]['lr']))
        train_loss = 0
        train_acc = 0

        # 为计算kappa值存储数据
        label_arr = {'true': [], 'pred': []}

        # 使用tqdm显示训练进度
        trainbar = tqdm(dataloader['train'], file=sys.stdout)

        # 启用BatchNormalization 和 Dropout
        net.train()

        for step, (X, Y) in enumerate(trainbar, 1):
            # 将数据放在GPU上训练
            X = X.to(device)
            Y = Y.to(device)

            # 向前传播
            out = net(X)
            loss = loss_func(out, Y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = out.max(1)
            train_acc += (pred == Y).sum().item()
            trainbar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format \
                (epoch + 1, EPOCH, train_loss, train_acc / BATCH_SIZE / step)

        train_arr['loss'].append(train_loss if train_loss < 100 else 100)
        train_arr['acc'].append(train_acc / len(image_datasets['train']))

        # 验证
        # 在验证时不计算梯度，减少计算量，加快训练速度
        with torch.no_grad():
            valid_loss = 0
            valid_acc = 0

            # 不启用BatchNormalization 和 Dropout
            net.eval()

            validbar = tqdm(dataloader['valid'], file=sys.stdout)
            for step, (X, Y) in enumerate(validbar, 1):
                # 将数据放在GPU上验证
                X = X.to(device)
                Y = Y.to(device)

                out = net(X)
                loss = loss_func(out, Y)

                valid_loss += loss.item()
                _, pred = out.max(1)
                valid_acc += (pred == Y).sum().item()

                label_arr['pred'] += list(pred.to('cpu'))
                label_arr['true'] += list(Y.to('cpu'))

                validbar.desc = "valid epoch[{}/{}] loss:{:.3f} acc:{:.3f} kappa:{:.3f}".format \
                    (epoch + 1, EPOCH, valid_loss, valid_acc / BATCH_SIZE / step,
                     cohen_kappa_score(label_arr['true'], label_arr['pred']))

        #更新学习率
        scheduler.step(epoch)

        # 为防止最开始的几轮训练顺势==损失过大导致整体损失图像不好观察，所以将其最大值设为100
        valid_arr['loss'].append(valid_loss if valid_loss < 100 else 100)
        valid_arr['acc'].append(valid_acc / len(image_datasets['valid']))

        # 记录最好模型
        if max_acc < valid_acc / len(image_datasets['valid']):
            best_stata_dict = net.state_dict()
            max_acc = valid_acc / len(image_datasets['valid'])
            kappa = cohen_kappa_score(label_arr['true'], label_arr['pred'])

    #绘图
    for phase in ['loss', 'acc']:
        plt.figure()
        plt.grid()
        plt.title(phase)
        plt.plot(train_arr[phase], label='train_' + phase)
        plt.plot(valid_arr[phase], label='valid_' + phase)
        plt.legend()

        #保存图片
        plt.savefig('./img/LR={}_'.format(LR) + phase + '_spp.jpg')

    #保存模型
    torch.save(best_stata_dict, "./model/LR={}_ACC={:.3f}_kappa={:.3f}_spp".format(LR, max_acc, kappa) + '.pth',
               _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()

# weight=torch.tensor([9851/2992,9851/1638,9851/2238,9851/613,9851/2370]).to(device)
#         loss_func = nn.CrossEntropyLoss(weight=weight)
