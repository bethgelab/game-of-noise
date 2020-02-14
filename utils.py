import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import random
import glob
import datetime
from pytz import timezone

from Networks import get_noise_generator
from evaluation_utils import *


def BoolArg(arg):
    if str(arg).lower() in ['1', 't', 'y', 'true', 'yes']:
        return True
    elif str(arg).lower() in ['0', 'f', 'n', 'false', 'no']:
        return False
    raise ValueError("Bool arg can only take values 0/1, t[rue]/f[alse], y[es]/n[o]")


def set_seeds(seed=1234):
    """Set random seeds for replicable results"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for replicable results this can be activated
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_models(model, noise_gen, optimizer_cl, exp_path, epoch, args):
    """Save the current classifier and noise generator states"""

    # save noise generator
    current_model_name = exp_path + 'Noise_model' + '_epoch_' + \
                         str(epoch) + '.pth'
    torch.save(noise_gen.state_dict(), current_model_name)

    # save classifier
    if epoch % 10 == 0:
        current_classifier_name = exp_path + \
                                  f'{args.mode}_NET_trained_epoch_{epoch}.pth'
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_cl.state_dict(),
                    }, current_classifier_name)
    
    return


def save_model(model, optimizer_cl, exp_path, epoch, args):
    """Save the current classifier state"""
    
    # save classifier
    if epoch % 10 == 0:
        current_classifier_name = exp_path + \
                                  f'{args.mode}_NET_trained_epoch_{epoch}.pth'
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_cl.state_dict(),
                    }, current_classifier_name)
    
    return
                        

def train_noise_generator(noise_gen, model, data, labels, num_iteration, writer, optimizer_gen,
                          retrain=False, batch_idx=0):
    """Train the noise generator"""

    model.eval()
    noise_gen.train()

    delta_img = noise_gen(data)
    data_noisy = fix_perturbation_size(data, delta_img, noise_gen.epsilon)

    adv_logits = model(data_noisy)

    criterion_for_training = nn.CrossEntropyLoss()
    loss = - criterion_for_training(adv_logits, labels)

    # LOGGING
    if not retrain and batch_idx % 10 == 0:
        writer.add_scalar('Train/Loss Noise Generator', loss, num_iteration)

    optimizer_gen.zero_grad()
    loss.backward()
    optimizer_gen.step()


def train_classifier(model, training_data, labels, optimizer_cl, writer, num_iteration):
    """Train the classifier"""

    model.train()

    output = model(training_data)
    criterion_for_training = nn.CrossEntropyLoss()
    loss = criterion_for_training(output, labels)

    optimizer_cl.zero_grad()
    loss.backward()
    optimizer_cl.step()
    writer.add_scalar('Train/Loss Classifier', loss, num_iteration)
    

def retrain_ng_and_evaluate(model, train_loader, test_loader_subsample, test_loader, args,
                                     writer, epoch, save_path=None, tmp=''):
    """Train a new noise generator from scratch"""
    device = args.device
    model.eval()

    print('get new noise gen')
    noise_gen_tmp = get_noise_generator(args)
    noise_gen_tmp.train()

    optimizer_gen_tmp = optim.Adam(noise_gen_tmp.parameters(), lr=args.lr_generator_retrain)

    for batch_idx, (data, labels) in enumerate(train_loader):
        if batch_idx > 1000:
            # We abort training after 1000 iterations.
            print('batch_idx: ', batch_idx)
            break
        num_iteration = batch_idx + epoch * len(train_loader.dataset) / args.batch_size
        data, labels = data.to(device), labels.to(device)  # clean data with labels
        train_noise_generator(noise_gen_tmp, model, data, labels, num_iteration,
                              writer, optimizer_gen_tmp, retrain=True)

        if batch_idx % 50 == 0:
            acc_noisy_ret, l2_noisy_ret, eps = \
                test_classifier_noisy(noise_gen_tmp, test_loader_subsample, model, args, device)
            writer.add_scalar(f'Z Val epoch: {epoch}/{tmp}Accuracy Noisy Retrain', acc_noisy_ret, batch_idx)
            writer.add_scalar(f'Z Val epoch: {epoch}/{tmp}Eps Retrain', eps, batch_idx)
            writer.add_scalar(f'Z Val epoch: {epoch}/{tmp}Eps NG', noise_gen_tmp.epsilon, batch_idx)
    
    # test accuracy and epsilon on the whole test set
    acc_noisy_ret, l2_noisy_ret, eps = \
        test_classifier_noisy(noise_gen_tmp, test_loader, model, args, device)
    if save_path is not None:
        ng_type = args.ng_type
        torch.save(noise_gen_tmp.state_dict(),
                   save_path + f'retrain_ng_{ng_type}_{epoch}.pth')
    writer.add_scalar('Val/Epsilon Retrain', eps, epoch)
    writer.add_scalar('Val/Accuracy Noisy Retrain', acc_noisy_ret, epoch)
    
    noise_gen_tmp.train()

    return noise_gen_tmp


def generate_noisy_data(noise_gen, data, labels, args, epoch, path, data_stylized=torch.zeros((1, 1, 1, 1)),
                        labels_stylized=torch.zeros(1)):
    batch_size = data.shape[0]
    training_data = data

    if args.mode == 'ANT' or args.mode == 'ANT+SIN':
        noise_gen.eval()
        if epoch == 0:  # In the 0-th epoch we don't have a history yet.
            p_current = 1 - args.p_clean
        else:
            p_current = args.p_current

        end_index = int(np.round(batch_size * p_current))

        delta_img = noise_gen(data[:end_index])
        adv_sample = fix_perturbation_size(data[:end_index], delta_img, noise_gen.epsilon)
        training_data[:end_index] = adv_sample

        if epoch > 0:
            counter = p_current  # get 4 random Noise models 
            if not os.path.exists(path):
                os.makedirs(path)
            allfiles = glob.glob(path + "/Noise_model*.pth")
            random.shuffle(allfiles)
            for file_ in allfiles:
                if counter >= args.p_clean - 0.001: 
                    break

                # load noise model
                noise_gen_prev = get_noise_generator(args)
                noise_gen_prev.load_state_dict(torch.load(file_))

                noise_gen_prev.eval()

                # get batch to add noise to
                start_b = int(np.round(batch_size * counter))
                end_b = int(np.round(batch_size * (counter + args.p_history)))

                # make unit vector
                delta_img = noise_gen_prev(data[start_b:end_b])
                adv_sample = fix_perturbation_size(data[start_b:end_b], delta_img, noise_gen_prev.epsilon)

                # put back in batch
                training_data[start_b:end_b] = adv_sample.detach()

                counter += args.p_history
        
        if args.mode == 'ANT+SIN':

            training_data[-data_stylized.shape[0]:] = data_stylized
            labels[-data_stylized.shape[0]:] = labels_stylized
            
        return training_data.detach(), labels
    
    elif args.mode == 'Gauss_single':
        
        start_index = int(np.round(batch_size * args.p_clean))
        noise = torch.empty(data[start_index:].shape, device=data.device).normal_(std=args.std_gauss)
        data_noisy = torch.clamp(data[start_index:] + noise, 0, 1)
        training_data[start_index:] = data_noisy
        
        return training_data.detach(), labels
        
    elif args.mode == 'Gauss_mult':
        
        start_index = int(np.round(batch_size * (1 - args.p_clean)))
        data_noisy = gaussian_noise_torch(data[start_index:])
        training_data[start_index:] = data_noisy
        
        return training_data.detach(), labels
    
    elif args.mode == 'Speckle':
        start_index = int(np.round(batch_size * (1 - args.p_clean)))
        data_noisy = speckle_noise_torch(data[start_index:])
        training_data[start_index:] = data_noisy
        
        return training_data.detach(), labels
    
    else:
        raise Exception(f'mode: {args.mode} is unknown')


def gaussian_noise_torch(data):
    """samples Gaussian noise according to the list stds, adds it to data
    and returns the noisy data.
    """
    stds = [.08, .12, 0.18, 0.26, 0.38]
    c = np.random.choice(stds, data.shape[0], replace=True)
    noise = torch.empty(data.shape, device=data.device).normal_() * torch.Tensor(c).view(-1, 1, 1, 1).to(data.device)
    
    return torch.clamp(data + noise, 0, 1)
        
    
def speckle_noise_torch(data):
    """samples speckle noise according to the list stds, adds it to data
    and returns the noisy data.
    """
    stds = [.15, .2, 0.35, 0.45, 0.6]
    c = np.random.choice(stds, data.shape[0], replace=True)
    noise = torch.empty(data.shape, device=data.device).normal_() * torch.Tensor(c).view(-1, 1, 1, 1).to(data.device)
    scaled_noise = data * noise
    
    assert (scaled_noise.shape == data.shape), "Shape of scaled speckle noise does not equal the shape of the input!"
    
    return torch.clamp(data + scaled_noise, 0, 1)
    
    
def get_exp_name(args_old, args_new):
    """
    Returns a convenient experiment name for tensorboard that compares
    arguments given to argparse to the default settings. It then
    writes the arguments where the values differ from the
    default settings into the experiment name.
    """
    
    args_new = args_new.__dict__
    for key, val in args_new.items():
        if val == 'false' or val == 'False':
            args_new[key] = False
        if val == 'true' or val == 'True':
            args_new[key] = True
    
    exp_name = args_new['name'] + '_'
    for key in args_old:
        old_val = args_old[key]
        if old_val != args_new[key] and key != 'device' and key != 'name' and key != 'expfolder':
            val = args_new[key]
            if isinstance(val, float):
                exp_name += f'{key}{val:.3f}-'
            elif isinstance(val, str):
                exp_name += f'{key}' + val[:5] + '-'
            else:
                exp_name += f'{key}' + str(val) + '-'

    tz = timezone("Europe/Berlin")
    
    return exp_name + f'--{datetime.datetime.now(tz=tz).strftime("%Y-%m-%d-%H-%M-%S")}'


class DefaultArguments:
    """default arguments for argparse"""
    
    # general arguments
    name = ''
    evaluate = False
    arch = 'resnet50'
    mode = 'ANT'
    model_name = 'clean'
    epochs = 120
    seed = 1234

    # directory path and batch size arguments
    test_batch_size = 100
    batch_size = 70
    sin_batch_size = 15
    test_subset_size = 1000

    # classifier related arguments
    gamma_LR_decay = 0.1
    step_size_LR_decay = 60
    lr_classifier = 1e-3
    momentum = 0.9

    # noise generator related arguments
    lr_generator = 1e-4
    lr_generator_retrain = 1e-5
    ng_type = '1x1'
    epsilon_generator = 135.
    
    # joint training related arguments
    p_clean = 0.5
    p_current = 0.3
    p_history = 0.1
    std_gauss = 0.5

    
def get_default_args(**passed_args):
    """get the default arguments for argparse"""
                     
    default_arguments = class_to_dict(DefaultArguments)
    if passed_args:
        default_arguments = {**default_arguments, **passed_args}

    return default_arguments


def class_to_dict(o):          
    keys = [f for f in dir(o) if not callable(getattr(o, f)) and not f.startswith('__')]
    new_dict = {}
    for key in keys:
        new_dict[key] = o.__dict__[key]
    return new_dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
