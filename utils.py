import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
import os.path as osp


def get_IN_C_data_loaders(args):
    """Returns data loaders for all ImageNet-C corruptions"""
    n_worker=30
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
            dset = datasets.ImageFolder(osp.join(args.imagenetc_path, path,
                                    str(severity)), transforms.Compose([
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
            data_loaders[name][str(severity)] = torch.utils.data.DataLoader(
                dset, batch_size=args.test_batch_size, shuffle=True, num_workers=n_worker)
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
    """Computes the accuracy on ImageNet-C"""
    model.eval()
    args.file.write("\n Performance on ImageNet-C:\n")
    with torch.no_grad():
        
        all_accuracies = []
        all_accuracies_wo_noises = []
        all_accuracies_top5 = []
        all_accuracies_wo_noises_top5 = []
                        
        for name, data_loader in data_loaders.items():
            n_correct, n_correct_top5, n_total = 0, 0, 0
            n_correct_wo_noises, n_correct_wo_noises_top5, n_total_wo_noises = 0, 0, 0
                        
            for severity, loader in data_loader.items(): 
                n_correct_sev, n_total_sev = 0, 0
                for data, labels in loader:
                    data, labels = data.cuda(), labels.cuda()
                    logits = model(data)
                    acc_tmp = get_accuracy(logits, labels, (1, 5))
                    n_correct += acc_tmp[0]
                    n_correct_sev += acc_tmp[0]
                    n_correct_top5 += acc_tmp[1]
                    n_total += float(data.shape[0])
                    n_total_sev += float(data.shape[0])
                    if name not in ['Gaussian Noise', 'Shot Noise', 'Impulse Noise']:
                        n_correct_wo_noises += acc_tmp[0]
                        n_correct_wo_noises_top5 += acc_tmp[1]
                        n_total_wo_noises += float(data.shape[0])
                    break
                        
                args.IN_C_Results[name][int(severity)+1] = n_correct_sev / n_total_sev
            all_accuracies.append(100 * n_correct / n_total)
            all_accuracies_top5.append(100*n_correct_top5 / n_total)
            if name not in ['Gaussian Noise', 'Shot Noise', 'Impulse Noise']:
                all_accuracies_wo_noises.append(100 * n_correct_wo_noises / n_total_wo_noises)
                all_accuracies_wo_noises_top5.append(100*n_correct_wo_noises_top5 / n_total_wo_noises)   
            accuracy = 100 * n_correct / n_total
            accuracy_top5 = 100 * n_correct_top5 / n_total
            args.IN_C_Results[name][0] = accuracy.item()
            args.IN_C_Results[name][1] = accuracy_top5.item()
            
            # Logging:
            print("Top1 accuracy on {0}: {1:.2f}".format(name, accuracy.item()))
            print("Top5 accuracy on {0}: {1:.2f}".format(name, accuracy_top5.item()))
            args.file.write("Top1 accuracy on {0}: {1:.2f}, ".format(name, accuracy.item()))
            args.file.write("Top5 accuracy on {0}: {1:.2f}".format(name, accuracy_top5.item()))
        print("Top1 accuracy on full ImageNet-C: {0:.2f}, ".format(np.mean(all_accuracies)))    
        print("Top5 accuracy on full ImageNet-C: {0:.2f}".format(np.mean(all_accuracies_top5)))    
        print("Top1 accuracy on ImageNet-C w/o Noises: {0:.2f}, ".format(np.mean(all_accuracies_wo_noises))) 
        print("Top5 accuracy on ImageNet-C w/o Noises: {0:.2f}, ".format(np.mean(all_accuracies_wo_noises_top5)))     
        args.file.write("Top1 accuracy on full ImageNet-C: {0:.2f}, ".format(np.mean(all_accuracies)))    
        args.file.write("Top5 accuracy on full ImageNet-C: {0:.2f}\n".format(np.mean(all_accuracies_top5)))    
        args.file.write("Top1 accuracy on ImageNet-C w/o Noises: {0:.2f}, ".format(np.mean(all_accuracies_wo_noises))) 
        args.file.write("Top5 accuracy on ImageNet-C w/o Noises: {0:.2f}\n, ".format(np.mean(all_accuracies_wo_noises_top5)))  
   
        outfile_name_IN_C_accuracy = './Results/' + args.model_name + '_IN_C_Results_resnet_50.npy'
        np.save(outfile_name_IN_C_accuracy, args.IN_C_Results)
        
    return
        
        
def validate(val_loader, model, args):
    """Computes accuracy on ImageNet val"""

    # switch to evaluate mode
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images, target = images.cuda(), target.cuda()

            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            break

    return acc1, acc5


def load_model(model_name):
    """loads robust model specified by modelname"""
    
    model = models.resnet50(pretrained='True')
    
    model_urls = {
            'ANT-SIN': './Models/ANT_SIN_Model_new.pth',
    }
    
    model_dict = torch.load(model_urls[model_name])
    model.load_state_dict(model_dict['model_state_dict'])
    model = model.eval().cuda()
    
    return model
