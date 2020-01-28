import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import os.path as osp
from torch.utils import model_zoo


def get_in_c_data_loaders(args):
    """Returns data loaders for all ImageNet-C corruptions"""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_loaders_names = \
        {'Brightness': 'brightness',
         'Contrast': 'contrast',
         'Defocus Blur': 'defocus_blur',
         'Elastic Transform': 'elastic_transform',
         'Fog': 'fog',
         'Frost': 'frost',
         'Gaussian Noise': 'gaussian_noise',
         'Glass Blur': 'glass_blur',
         'Impulse Noise': 'impulse_noise',
         'JPEG Compression': 'jpeg_compression',
         'Motion Blur': 'motion_blur',
         'Pixelate': 'pixelate',
         'Shot Noise': 'shot_noise',
         'Snow': 'snow',
         'Zoom Blur': 'zoom_blur'}
    data_loaders = {}
    for name, path in data_loaders_names.items():
        data_loaders[name] = {}
        for severity in range(1, 6):
            dset = datasets.ImageFolder(osp.join(args.imagenetc_path, path, str(severity)),
                                        transforms.Compose([transforms.CenterCrop(224),
                                                            transforms.ToTensor(), normalize, ]))
            data_loaders[name][str(severity)] = torch.utils.data.DataLoader(
                dset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.workers)
    return data_loaders


def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_on_imagenet_c(data_loaders, model, args):
    """Computes model accuracy and mCE on ImageNet-C"""

    print("Performance on ImageNet-C:\n", file=args.file)
    model.eval()
    ce_alexnet = get_ce_alexnet()

    with torch.no_grad():

        top1_in_c = AverageMeter('Acc_IN_C@1', ':6.2f')
        top5_in_c = AverageMeter('Acc_IN_C@5', ':6.2f')
        top1_in_c_wo_noises = AverageMeter('Acc_IN_C_wo_Noises@1', ':6.2f')
        top5_in_c_wo_noises = AverageMeter('Acc_IN_C_wo_Noises@5', ':6.2f')
        mce, counter = 0, 0

        for name, data_loader in data_loaders.items():
            top1_tmp = AverageMeter('Acc_tmp@1', ':6.2f')
            for severity, loader in data_loader.items():
                top1_sev_tmp = AverageMeter('Acc_sev_tmp@1', ':6.2f')
                top5_sev_tmp = AverageMeter('Acc_sev_tmp@5', ':6.2f')

                for data, labels in loader:
                    data, labels = data.cuda(), labels.cuda()
                    logits = model(data)
                    acc1, acc5 = get_accuracy(logits, labels, (1, 5))

                    top1_in_c.update(acc1[0], data.size(0))
                    top5_in_c.update(acc5[0], data.size(0))
                    top1_sev_tmp.update(acc1[0], data.size(0))
                    top5_sev_tmp.update(acc5[0], data.size(0))
                    top1_tmp.update(acc1[0], data.size(0))

                    if name not in ['Gaussian Noise', 'Shot Noise', 'Impulse Noise']:
                        top1_in_c_wo_noises.update(acc1[0], data.size(0))
                        top5_in_c_wo_noises.update(acc5[0], data.size(0))

            # get Corruption Error CE:
            CE = get_mce_from_accuracy(top1_tmp.avg.item(), ce_alexnet[name])
            mce += CE
            counter += 1

            # Logging:
            print("{0}: Top1 accuracy {1:.2f}, Top5 accuracy: {2:.2f}, CE: {3:.2f}\n".format(
                name, top1_tmp.avg.item(), top1_tmp.avg.item(), 100. * CE), file=args.file)

        mce /= counter
        print("Full ImageNet-C: Top1 accuracy {0:.2f}, Top5 accuracy: {1:.2f}, mCE: {2:.2f}\n".format(
            top1_in_c.avg.item(),
            top5_in_c.avg.item(),
            mce * 100.), file=args.file)
        print("ImageNet-C w/o Noises: : Top1 accuracy: Top1 accuracy {0:.2f}, Top5 accuracy: {1:.2f}\n".format(
            top1_in_c_wo_noises.avg.item(),
            top5_in_c_wo_noises.avg.item()), file=args.file)

    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

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


def validate(val_loader, model):
    """Computes accuracy on ImageNet val"""

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.cuda(), target.cuda()

            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return top1.avg, top5.avg


def load_model(model_name):
    """loads robust model specified by modelname"""

    model = models.resnet50(pretrained=True)
   
    model_paths = {
        'ANT-SIN': './Models/ANT_SIN_Model.pth',
        'ANT': './Models/ANT_Model.pth',
        'Speckle': './Models/Speckle_Model.pth',
        'Gauss_mult': './Models/Gauss_mult_Model.pth',
        'Gauss_sigma_0.5': './Models/Gauss_sigma_0.5_Model.pth',
    }
    
    if not model_name == 'clean':
        checkpoint = torch.load(model_paths[model_name])
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().cuda()

    return model


def get_mce_from_accuracy(accuracy, error_AlexNet):
    """Computes mean Corruption Error from accuracy"""

    error = 100. - accuracy
    ce = error / (error_AlexNet * 100.)

    return ce


def get_ce_alexnet():
    """Returns Corruption Error values for AlexNet"""

    ce_alexnet = dict()
    ce_alexnet['Gaussian Noise'] = 0.886428
    ce_alexnet['Shot Noise'] = 0.894468
    ce_alexnet['Impulse Noise'] = 0.922640
    ce_alexnet['Defocus Blur'] = 0.819880
    ce_alexnet['Glass Blur'] = 0.826268
    ce_alexnet['Motion Blur'] = 0.785948
    ce_alexnet['Zoom Blur'] = 0.798360
    ce_alexnet['Snow'] = 0.866816
    ce_alexnet['Frost'] = 0.826572
    ce_alexnet['Fog'] = 0.819324
    ce_alexnet['Brightness'] = 0.564592
    ce_alexnet['Contrast'] = 0.853204
    ce_alexnet['Elastic Transform'] = 0.646056
    ce_alexnet['Pixelate'] = 0.717840
    ce_alexnet['JPEG Compression'] = 0.606500

    return ce_alexnet