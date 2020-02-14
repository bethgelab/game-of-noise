import numpy as np
import torch
import torchvision.models as models
from Networks import ZeroOneResNet50_robust


def fix_perturbation_size(x0, delta, epsilon):
    """
    calculates eta such that
        norm(clip(x0 + eta * delta, 0, 1)) == epsilon

    assumes x0 and delta to have a batch dimension
    and epsilon to be a scalar
    """
    n, ch, nx, ny = x0.shape
    assert delta.shape[0] == n

    delta2 = delta.pow(2).flatten(1)
    space = torch.where(delta >= 0, 1 - x0, x0).flatten(1)
    f2 = space.pow(2) / torch.max(delta2, 1e-20 * torch.ones_like(delta2))
    f2_sorted, ks = torch.sort(f2, dim=-1)
    m = torch.cumsum(delta2.gather(dim=-1, index=ks.flip(dims=(1,))), dim=-1).flip(dims=(1,))
    dx = f2_sorted[:, 1:] - f2_sorted[:, :-1]
    dx = torch.cat((f2_sorted[:, :1], dx), dim=-1)
    dy = m * dx
    y = torch.cumsum(dy, dim=-1)
    c = y >= epsilon**2

    # work-around to get first nonzero element in each row
    f = torch.arange(c.shape[-1], 0, -1, device=c.device)
    v, j = torch.max(c.long() * f, dim=-1)

    rows = torch.arange(0, n)

    eta2 = f2_sorted[rows, j] - (y[rows, j] - epsilon**2) / m[rows, j]

    eta2 = torch.where(v == 0, f2_sorted[:, -1], eta2)
    eta = torch.sqrt(eta2)
    eta = eta.reshape((-1,) + (1,) * (len(x0.shape) - 1))

    return torch.clamp(eta * delta + x0, 0, 1).view(n, ch, nx, ny)


def zero_accuracy_perturbation(image, labels, noise, model, n_binary_steps=40):
    """Calculate the l2 distance to the decision boundary; i.e. how much of the 
    given noise do we have to add to achieve a misclassification?"""
    
    model.eval()
    with torch.no_grad():
        save_images = torch.zeros_like(image).flatten(1)
        save_l2s = torch.zeros(image.shape[0], device=image.device).fill_(save_images.shape[1])

        image_f = image.flatten(1)
        noise_f = noise.flatten(1)

        assert (torch.sum(noise_f, dim=1) != 0).any()   # don't divide by 0
        noise_normalized = noise_f / torch.norm(noise_f, p=2, dim=1).view(image.shape[0], 1)

        eta = torch.empty((image.shape[0]), device=image.device).fill_(0)

        # coarse search
        for b in range(0, int(np.round(np.log(784.) / np.log(1.8)))):
            noise_f_clipped = torch.clamp(eta[:, None] * noise_normalized + image_f, 0, 1)
            mask_correct = torch.argmax(model(noise_f_clipped.view(image.shape)), dim=1) == labels
            eta[mask_correct] = 1.8**b

        delta_eta = eta / 2.

        # binary search
        for i in range(n_binary_steps):
            noise_f_clipped = torch.clamp(eta[:, None] * noise_normalized + image_f, 0, 1)
            current_l2 = torch.norm(noise_f_clipped - image_f, p=2, dim=1)

            mask_correct = torch.argmax(model(noise_f_clipped.view(image.shape)), dim=1) == labels
            mask_wrong = ~mask_correct

            eta[mask_wrong] = eta[mask_wrong] - delta_eta[mask_wrong]
            eta[mask_correct] = eta[mask_correct] + delta_eta[mask_correct]
            delta_eta /= 2.

            save_images[mask_wrong, :] = noise_f_clipped[mask_wrong, :]
            save_l2s[mask_wrong] = current_l2[mask_wrong]

        return save_images.view(image.shape), save_l2s


def l2_manual_batch(perturbed, data):
    """Calculates the batch-wise l2 distance"""
    norm = torch.norm(perturbed.flatten(1) - data.flatten(1), p=2, dim=1)
    mse_man = norm / (data.shape[1] * data.shape[2] * data.shape[3])
    return mse_man


def test_classifier_noisy(noise_gen, test_loader, model, args, device):
    """Test the classifier's performance on adversarial noise"""
    model.eval()
    noise_gen.eval()
    all_l2 = torch.zeros(len(test_loader.dataset))
    all_epsilons = torch.zeros(len(test_loader.dataset))
    correct = 0
    with torch.no_grad():
        for batch_idx, (test_data, test_labels) in enumerate(test_loader):
            test_data, test_labels = test_data.to(device), test_labels.to(device)

            # Get delta = NM(z)
            delta_img = noise_gen(test_data)
            adv_sample = fix_perturbation_size(test_data, delta_img, epsilon=noise_gen.epsilon)
            imgs, epsilons = zero_accuracy_perturbation(test_data, test_labels, delta_img, model)

            adv_logits = model(adv_sample)
            pred = adv_logits.max(1, keepdim=True)[1]
            correct += pred.eq(test_labels.view_as(pred)).sum().item()

            start_ind = batch_idx * args.test_batch_size
            end_ind = batch_idx * args.test_batch_size + len(test_labels)

            l2 = l2_manual_batch(adv_sample, test_data)
            all_l2[start_ind:end_ind] = l2
            all_epsilons[start_ind:end_ind] = epsilons             
                
    accuracy = correct / len(test_loader.dataset)
    return accuracy, torch.median(all_l2), torch.median(all_epsilons)



def load_robust_model(model_name, device):
    """loads robust model specified by modelname"""

    model = models.resnet50(pretrained=True)
   
    model_paths = {
        'ANT+SIN': './Models/ANT_SIN_Model.pth',
        'ANT': './Models/ANT_Model.pth',
        'Speckle': './Models/Speckle_Model.pth',
        'Gauss_mult': './Models/Gauss_mult_Model.pth',
        'Gauss_sigma_0.5': './Models/Gauss_sigma_0.5_Model.pth',
    }
    
    if not model_name == 'clean':
        checkpoint = torch.load(model_paths[model_name])
        model.load_state_dict(checkpoint['model_state_dict'])

    # wrap model:
    model_wrapped = ZeroOneResNet50_robust(model)
    model_wrapped = model_wrapped.eval().to(device)
    
    return model_wrapped


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


def accuracy_on_imagenet_c(data_loaders, model, args, writer, num_iteration):
    """Computes model accuracy and mCE on ImageNet-C"""

    print("Performance on ImageNet-C:")
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
            top5_tmp = AverageMeter('Acc_tmp@5', ':6.2f')
            for severity, loader in data_loader.items():
                top1_sev_tmp = AverageMeter('Acc_sev_tmp@1', ':6.2f')
                ct = 0
                for data, labels in loader:
                    data, labels = data.to(args.device), labels.to(args.device)
                    logits = model(data)
                    acc1, acc5 = get_accuracy(logits, labels, (1, 5))

                    top1_in_c.update(acc1[0], data.size(0))
                    top5_in_c.update(acc5[0], data.size(0))
                    top1_sev_tmp.update(acc1[0], data.size(0))
                    top1_tmp.update(acc1[0], data.size(0))
                    top5_tmp.update(acc5[0], data.size(0))
                    
                    if name not in ['Gaussian Noise', 'Shot Noise', 'Impulse Noise']:
                        top1_in_c_wo_noises.update(acc1[0], data.size(0))
                        top5_in_c_wo_noises.update(acc5[0], data.size(0))
                        
                    ct += 1
                    if ct == 50 and not args.evaluate:
                        break
                args.IN_C_Results[name][int(severity)+1] = top1_sev_tmp.avg.item()
                print("{0}: Severity: {1}, Top1 accuracy {2:.2f}".format(name, severity, top1_sev_tmp.avg.item()),
                      file=args.file)
            # get Corruption Error CE:
            CE = get_mce_from_accuracy(top1_tmp.avg.item(), ce_alexnet[name])
            mce += CE
            counter += 1

            # Logging:
            print("{0}: Top1 accuracy {1:.2f}, Top5 accuracy: {2:.2f}, CE: {3:.2f}\n".format(
                name, top1_tmp.avg.item(), top5_tmp.avg.item(), 100. * CE))
            writer.add_scalar(f'IN-C/Accuracy {name}', top1_tmp.avg.item(), num_iteration)    
            writer.add_scalar(f'IN-C/Accuracy {name}top5', top5_tmp.avg.item(), num_iteration)   
            
            args.IN_C_Results[name][int(num_iteration/10)] = top1_tmp.avg.item()
            
        mce /= counter
        print("Full ImageNet-C: Top1 accuracy {0:.2f}, Top5 accuracy: {1:.2f}, mCE: {2:.2f}\n".format(
            top1_in_c.avg.item(),
            top5_in_c.avg.item(),
            mce * 100.), file=args.file)
        print("ImageNet-C w/o Noises: : Top1 accuracy: Top1 accuracy {0:.2f}, Top5 accuracy: {1:.2f}\n".format(
            top1_in_c_wo_noises.avg.item(),
            top5_in_c_wo_noises.avg.item()), file=args.file)
       
        writer.add_scalar('IN-C/mCE', mce * 100., num_iteration)
        writer.add_scalar('IN-C/Accuracy Full ImageNet-C', top1_in_c.avg.item(), num_iteration)
        writer.add_scalar('IN-C/Accuracy ImageNet-C w/o noises', top1_in_c_wo_noises.avg.item(), num_iteration)       
        writer.add_scalar('IN-C/Accuracy Full ImageNet-C top5', top5_in_c.avg.item(), num_iteration)
        writer.add_scalar('IN-C/Accuracy ImageNet-C w/o noises top5', top5_in_c_wo_noises.avg.item(), num_iteration)
        writer.add_scalar('Val/Accuracy Full ImageNet-C', top1_in_c.avg.item(), num_iteration)
        path = args.exp_path + '/IN_C_Results_resnet_50.npy'
        np.save(path, args.IN_C_Results)
        
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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


def validate(val_loader, model, device='cuda'):
    """Computes accuracy on ImageNet val"""

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)

            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return top1.avg, top5.avg


def get_mce_from_accuracy(accuracy, error_alexnet):
    """Computes mean Corruption Error from accuracy"""
    error = 100. - accuracy
    ce = error / (error_alexnet * 100.)

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
