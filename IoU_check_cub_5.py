import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import model_cub as model
# import model as model
# import model_cub_dense as model
import argparse
import Metric
from LoadData import data_loader
from bounding_box import connected_component

def returnCAM_(feature_conv, weight_softmax, class_idx, reverse_idx, h_x, j, threshold, thr, function=None):
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    # print(bz, nc, h, w)
    output_cam = []
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam1 = weight_softmax[reverse_idx].dot(feature_conv.reshape((nc, h * w)))

    if function == 't1b0':
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

    elif function == 't1b1':
        cam = cam - cam1[-1]
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

    elif function == 'linear':
        h_x = np.linspace(1, -1, 200)
        cam = np.matmul(h_x, cam1)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

    elif function == 'quadratic':
        h_x1 = np.linspace(1, 0, thr)
        h_x1 = h_x1 * h_x1
        h_x2 = np.linspace(0, -1, 200-thr)
        h_x2 = h_x2 * h_x2
        h_x2 = - h_x2
        h_x = np.concatenate([h_x1, h_x2])
        cam = np.matmul(h_x, cam1)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

    else:
        print('please select combinational function')
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

    output_cam.append(cv2.resize(cam_img, size_upsample))
    cam = cv2.resize(cam, size_upsample)

    bounding_box_thr = np.amax(cam) * threshold
    cutting_image = np.where(cam > bounding_box_thr, cam, 0)
    [slice_y, slice_x] = connected_component(cutting_image)
    bounding_box = [slice_x[0], slice_y[0], slice_x[1], slice_y[1]]
    return output_cam, bounding_box

if __name__ == '__main__':
    train_list = './CUB_datalist/train_list.txt'
    test_list = './CUB_datalist/test_list.txt'

    parser = argparse.ArgumentParser(description='CAM')
    parser.add_argument("--img_dir", type=str, default='./CUB_200_2011/images',
                        help='Directory of training images')
    parser.add_argument("--train_list", type=str,
                        default=train_list)
    parser.add_argument("--test_list", type=str,
                        default=test_list)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='cub')
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--phase", default='test', type=str)
    parser.add_argument("--function", default='quadratic', type=str, help='select CCAM function t1b0, t1b1, linear, quadratic')
    parser.add_argument("--model", default='model/pretrained_cnn_cub_max_pool_no_relu_100.pth', type=str)

    args = parser.parse_args()
    args.batch_size = 1
    train_loader, test_loader = data_loader(args, test_path=True)

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    # net = model.load_net_google(args.num_classes).to(device)
    net = model.load_net(num_classes=args.num_classes, model_name=args.model).to(device)

    index = torch.LongTensor().to(device)
    index1 = torch.LongTensor().to(device)
    h_x = torch.FloatTensor().to(device)
    with torch.no_grad():
        for i, dat in enumerate(train_loader):
            net.eval()
            size_upsample = (224, 224)
            img_path, image_tensor, image_level = dat

            image_tensor, image_level = image_tensor.to(device), image_level.to(device)
            logit, _ = net(image_tensor)
            logit1, idx = logit.data.sort(1, True)

            index = torch.cat((index, idx[:, :]), 0)
            h_x = torch.cat((h_x, logit1[:, :].data), 0)

    print(len(index))

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    cam_thr_list = [0.0, 0.1, 0.12, 0.14]
    # thr = [100, 0, 20, 40, 60, 80, 120, 140, 160, 180, 200]
    thr = [100]
    for aa, thr in enumerate(thr):
    # for step, threshold in enumerate(cam_thr_list):
        threshold = 0.12
        params = list(net.parameters())
        net.train(False)
        net.eval()
        with torch.no_grad():
            top1 = Metric.AverageEpochMeter('Top-1 Classification Acc')
            top5 = Metric.AverageEpochMeter('Top-5 Classification Acc')
            GT_loc = Metric.AverageEpochMeter('Top-1 GT-Known Localization Acc')
            top1_loc = Metric.AverageEpochMeter('Top-1 Localization Acc')
            top5_loc = Metric.AverageEpochMeter('Top-5 Localization Acc')
            progress = Metric.ProgressEpochMeter(
                len(test_loader),
                [top1, top5, top1_loc, top5_loc, GT_loc],
                prefix="\nValidation Phase: ")
            gt_file = './CUB_datalist/test_bounding_box.txt'
            f = open(gt_file, 'r')
            net.eval()
            for i, (img_path, image_tensor, image_level) in enumerate(test_loader):
                image_tensor = image_tensor.cuda(non_blocking=True)
                image_level = image_level.cuda(non_blocking=True)
                logit, _ = net(image_tensor)

                feature_map, score = net.get_cam()
                weight_softmax = np.squeeze(params[-2].clone().detach().cpu().numpy())
                feature_map = feature_map.detach().cpu().numpy()

                local_acc5 = 0
                correct5 = 0
                local_acc_temp = 0
                correct_temp = 0
                line = f.readline().split(' ')
                bounding_box_gt1, gt_size = Metric.bounding_box_gt(line)
                for j in range(5):
                    [CAMs, bounding_box] = returnCAM_(feature_map, weight_softmax, [int(index[i, j].item())],
                                                      index[i, :], h_x[i], j, float(threshold), int(thr), args.function)
                    [__, gt_known_bounding_box] = returnCAM_(feature_map, weight_softmax, [int(image_level.item())],
                                                             index[i, :], h_x[i], j, float(threshold), int(thr), args.function)
                    class_match = index[i, j].item() == image_level.item()
                    correct_temp += index[i, j].item() == image_level.item()
                    bounding_box = Metric.bounding_box_resize(bounding_box, gt_size)
                    gt_known_bounding_box = Metric.bounding_box_resize(gt_known_bounding_box, gt_size)
                    if j == 0:
                        local_acc0 = Metric.Localization_acc(bounding_box, bounding_box_gt1, class_match)
                        local_acc_gt = Metric.Localization_acc(gt_known_bounding_box, bounding_box_gt1, class_match=True)
                    local_acc_temp += Metric.Localization_acc(bounding_box, bounding_box_gt1, class_match)
                if local_acc_temp >= 1.0:
                    local_acc5 = 1.0
                top1_loc.update(local_acc0, image_tensor.size(0))
                top5_loc.update(local_acc5, image_tensor.size(0))
                GT_loc.update(local_acc_gt, image_tensor.size(0))

                correct = (index[i, 0].item() == image_level.item())
                if correct_temp >= 1:
                    correct5 = 1.0

                top1.update(correct, image_tensor.size(0))
                top5.update(correct5, image_tensor.size(0))
            f.close()
            print(threshold)
            progress.display(30)
            torch.cuda.empty_cache()

