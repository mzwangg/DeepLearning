import os
import sys
import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import datasets
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from MobileNetV3 import mobilenet_v3_large
from torch.utils.data import DataLoader
from tqdm import tqdm

device=("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
#在model里放入所有要预测的模型的名称，则会一次进行测试
model=['LR=0.001_ACC=0.879_spp=0.792_ssp.pth'
       'LR=0.001_ACC=0.861_DDRenhance=0.762_ssp.pth']

def main():
    #处理要和验证集相一致
    data_transform = transforms.Compose(
        [transforms.Resize([224,224]),
        transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder(root='./DDR/test',
                                          transform=data_transform)

    # 读取完数据后，对数据进行装载
    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
    dataloader = DataLoader(dataset=image_datasets,batch_size=BATCH_SIZE,num_workers=nw)

    loss_fun=nn.CrossEntropyLoss()

    for model_name in model:
        #载入训练好的模型
        net = mobilenet_v3_large(5)
        net.load_state_dict(torch.load("./model/"+model_name))
        net.to(device)

        acc = 0
        loss = 0
        net.eval()
        label_arr = {'true': [], 'pred': []}
        print("model:",model_name)

        #不更新梯度，减少计算量
        with torch.no_grad():
            bar = tqdm(dataloader, file=sys.stdout)
            for step,(X, Y) in enumerate(bar,1):
                X=X.to(device)
                Y=Y.to(device)
                out = net(X)
                _, pred = out.max(1)
                acc += (pred == Y).sum().item()
                loss+=loss_fun(out,Y).item()
                label_arr['pred'] += list(pred.to('cpu'))
                label_arr['true'] += list(Y.to('cpu'))
                bar.desc = "loss:{:.3f} acc:{:.3f}".format(loss,acc / BATCH_SIZE / step)
            print("loss={} acc={}".format(loss,acc/len(image_datasets)))

            #绘制混淆矩阵
            plt.figure(figsize=(6,5))
            sns.set()
            C2 = confusion_matrix(label_arr['true'], label_arr['pred'], labels=[0, 1, 2, 3, 4],normalize='true')
            ax = sns.heatmap(C2, annot=True,cmap='PuBu')
            ax.set_title(model_name[:9]+"_acc={:.3f}_kappa={:.3f}".format(acc/len(image_datasets),
                                                                  cohen_kappa_score(label_arr['true'], label_arr['pred'])))  # 标题
            ax.set_xlabel('predict')  # x 轴
            ax.set_ylabel('true')  # y 轴

            #保存混淆矩阵
            plt.savefig('./img/consudion_matrix/LR=0.001_focal_new.jpg')
    plt.show()

if __name__ == '__main__':
    main()