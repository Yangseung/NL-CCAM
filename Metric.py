import numpy as np
import sys

#boxA, boxB shape -> (B,
#boxA[0] : min x, boxA[1] : min y, boxA[2] : max x, boxA[3] : max y
def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def Localization_acc(boxA, boxB, class_match):
    if class_match == False:
        return 0.0
    iou = IoU(boxA, boxB)
    if iou >= 0.5:
        return 1.0
    else:
        return 0.0

def bounding_box_gt(boxA):
    return [float(boxA[0]), float(boxA[1]), float(boxA[2]), float(boxA[3])], [float(boxA[4]), float(boxA[5])]

def bounding_box_resize(boxA, gt_size):
    real_y = gt_size[0]
    real_x = gt_size[1]
    target_size = 224
    x_scale = real_x / target_size
    y_scale = real_y / target_size
    x = int(np.round((float(boxA[0])) * x_scale))
    y = int(np.round((float(boxA[1])) * y_scale))
    x_max = int(np.round((float(boxA[2])) * x_scale))
    y_max = int(np.round((float(boxA[3])) * y_scale))
    return [x, y, x_max, y_max]

def bounding_box_gt_resize(boxA, gt_size):
    real_y = int(gt_size[0])
    real_x = int(gt_size[1])
    target_size = 224
    x_scale = target_size / real_x
    y_scale = target_size / real_y
    x = int(np.round((float(boxA[0])) * x_scale))
    y = int(np.round((float(boxA[1])) * y_scale))
    x_max = int(np.round((float(boxA[2])) * x_scale))
    y_max = int(np.round((float(boxA[3])) * y_scale))
    return [x, y, x_max, y_max]

def bounding_box_upsize(boxA, H, W):
    real_y = 224
    real_x = 224
    x_scale = H / real_x
    y_scale = W / real_y
    x = int(np.round(float(boxA[0]) * x_scale))
    y = int(np.round(float(boxA[1]) * y_scale))
    x_max = int(np.round(float(boxA[2]) * x_scale))
    y_max = int(np.round(float(boxA[3]) * y_scale))
    return [x, y, x_max, y_max]

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

class SumEpochMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def __str__(self):
        fmtstr = '{name}: {sum' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class AverageEpochMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressEpochMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix]
        entries += [str(meter) for meter in self.meters]
        print('\n'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Logger(object):
    """Log stdout messages"""
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()