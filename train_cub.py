import os
import torch
import torch.nn as nn
import model_cub as model
import argparse
import Metric
from LoadData import data_loader
import pdb

if not os.path.exists('./model'):
    os.mkdir('model/')

# train_list = 'mask_image.txt'
train_list = './CUB_datalist/train_list.txt'
test_list = './CUB_datalist/test_list.txt'
is_cuda = torch.cuda.is_available()
LR = 0.0005
EPOCH = 100

#
parser = argparse.ArgumentParser(description='CAM')
parser.add_argument("--img_dir", type=str, default='./CUB_200_2011/images',
                    help='Directory of training images')
parser.add_argument("--train_list", type=str,
                    default=train_list)
parser.add_argument("--test_list", type=str,
                    default=test_list)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--crop_size", type=int, default=224)
parser.add_argument("--dataset", type=str, default='cub')
parser.add_argument("--num_classes", type=int, default=200)
parser.add_argument("--lr", type=float, default=LR)
parser.add_argument("--decay_points", type=str, default='none')
parser.add_argument("--tencrop", type=str, default='False')
parser.add_argument("--epoch", type=int, default=EPOCH)
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--phase", type=str, default='train')

device = torch.device("cuda" if is_cuda else "cpu")

args = parser.parse_args()

cnn = model.get_net(num_classes=args.num_classes).to(device)
# cnn = model_cub.load_net().to(device)

criterion = nn.CrossEntropyLoss().to(device)

# params = list(cnn.parameters())
params = list(cnn.classifier.parameters())
# params = list(cnn.attn.parameters()) + list(cnn.classifier.parameters())
optimizer = torch.optim.Adam(params, lr=LR)
params1 = list(cnn.conv.parameters())
optimizer1 = torch.optim.Adam(params1, lr=LR/10)

max_acc = 0

print("START TRAINING")
train_loader, test_loader = data_loader(args, test_path=True)

for epoch in range(args.epoch):
    new_lr1 = Metric.poly_lr_scheduler(optimizer, LR, epoch, lr_decay_iter=1, max_iter = args.epoch)
    new_lr2 = Metric.poly_lr_scheduler(optimizer1, LR/10, epoch, lr_decay_iter=1, max_iter = args.epoch)
    # print(optimizer.param_groups[0]['lr'])
    print(optimizer.param_groups[0]['lr'], optimizer1.param_groups[0]['lr'])

    cnn.train()
    epoch_loss = 0
    correct = 0
    total = 0
    acc_sum = 0
    a=0
    for i, dat in enumerate(train_loader):
        img_path, images, labels = dat
        images, labels = images.to(device), labels.to(device)
        # pdb.set_trace()
        labels = labels.long()
        optimizer.zero_grad()
        optimizer1.zero_grad()
        outputs, _ = cnn(images)

        correct += (torch.max(outputs, 1)[1].view(labels.size()).data == labels.data).sum()
        total += train_loader.batch_size
        train_acc = 100. * correct/total
        acc_sum += train_acc
        a += 1
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer1.step()

        if (i+1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, training accuracy: %.3f'
                  % (epoch+1, args.epoch, i+1, len(train_loader), loss.item(), train_acc))

    avg_epoch_loss = epoch_loss / len(train_loader)
    acc_avg = acc_sum/a
    print("Epoch: %d, Avg Loss: %.4f, Avg acc : %.3f" % (epoch+1, avg_epoch_loss, acc_avg))

    a=0
    correct = 0
    total = 0
    acc_sum = 0
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            cnn.eval()
            # for i, dat in enumerate(test_loader):
            #     img_path, images, labels = dat
            #     images, labels = images.to(device), labels.to(device)
            #     labels = labels.long()
            #     outputs, _ = cnn(images)
            #
            #     correct = (torch.max(outputs, 1)[1].view(labels.size()).data == labels.data).sum()
            #     total += images.size(0)
            #     acc_sum += correct
            #
            # eval_acc_avg = float(acc_sum) / float(total) *100
            # print("Epoch: %d, Avg acc : %.3f" % (epoch + 1, eval_acc_avg))
            #
            # # if eval_acc_avg > max_acc:
            # print("Renew model")
            # max_acc = eval_acc_avg
            torch.save(cnn.state_dict(), 'model/vgg_cnn_cub_{}.pth'.format(epoch+1))
            print("----------------------------------")