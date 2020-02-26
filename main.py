from tensorboardX import SummaryWriter
import torch.optim as optim
import torch
import argparse

from datasets import setup_data_loader
from utils import *
from evaluation_utils import *
from Networks import load_imagenet_classifier


parser = argparse.ArgumentParser(description='Evaluation of Models')

# general arguments
parser.add_argument('--name', default='', type=str,
                    help='name of the experiment')
parser.add_argument('-e', '--evaluate', default='False', type=BoolArg,
                    help='evaluate model on validation set (default: False)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='choose model architecture (default: resnet50)',
                    choices=['resnet18', 'resnet50', 'resnet152', 'inceptionV3'])
parser.add_argument('--mode', type=str, default='ANT',
                    help='which training mode should be used (default: ANT)',
                    choices=['ANT', 'ANT+SIN', 'Gauss_single', 'Gauss_mult', 'Speckle'])
parser.add_argument('--model_name', default='clean', type=str,
                    help='which robust trained model should be '
                         'evaluated with the provided model weights (default: clean)',
                    choices=['clean', 'ANT', 'ANT+SIN', 'Speckle', 'Gauss_mult', 'Gauss_sigma_0.5', 'ANT3x3', 'ANT3x3+SIN'])
parser.add_argument('--device', default='cuda', type=str,
                    help='cuda or cpu (default: cuda)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run (default: 120)')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. (default: 1234)')

# directory path and batch size arguments
parser.add_argument('--imagenetc-path', metavar='DIR',
                    default='./data/ImageNet-C/imagenet-c/')
parser.add_argument('--datadir-clean', metavar='DIR',
                    default='./data/IN/raw-data/', help='path to clean ImageNet')
parser.add_argument('--datadir-sin', metavar='DIR',
                    default='./data/SIN/raw-data/', help='path to stylized ImageNet')
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 30)')
parser.add_argument('-tb', '--test-batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('-b', '--batch-size', default=70, type=int,
                    metavar='N', help='mini-batch size (default: 70)')
parser.add_argument('--sin-batch-size', default=15, type=int,
                    metavar='N', help='mini-batch size for stylized images (default: 15)')
parser.add_argument('--test-subset-size', default=1000, type=int,
                    metavar='N', help='Subset size of clean images for intermediate testing (default: 1000)')

# classifier related arguments
parser.add_argument('--gamma_LR_decay', default=0.1, type=float,
                    help='gamma for StepLR decay for classifier (default: 0.1)')
parser.add_argument('--step_size_LR_decay', default=60, type=int,
                    help='step size for StepLR decay for classifier (default: 60)')
parser.add_argument('--lr_classifier', default=0.001, type=float,
                    metavar='LR', help='initial learning rate of the classifier (default: 1e-3)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for training the classifier (default: 0.9)')

# noise generator related arguments
parser.add_argument('--lr_generator', default=0.0001, type=float,
                    help='learning rate of the generator (default: 1e-4)')
parser.add_argument('--lr_generator_retrain', default=0.0005, type=float,
                    help='learning rate of the generator due training from scratch (default: 5e-4)')
parser.add_argument('--custom_init_ng', default='True', type=BoolArg,
                    help='initialize the noise generator such that it outputs a Gaussian distribution (default: True)')
parser.add_argument('--ng_type', default='1x1', type=str,
                    help='which noise generator should be used (default: 1x1)',
                    choices=['1x1', '3x3'])
parser.add_argument('--epsilon_generator', default=135., type=float,
                    help='epsilon ball for the perturbation size of noise during training (default: 135.)')
parser.add_argument('--channels', default=3, type=int, help='number of channels in the dataset (default: 3)')

# joint training related arguments
parser.add_argument('--p_clean', default=0.5, type=float,
                    help='percentage of clean images during joint training (default: 0.5)')
parser.add_argument('--p_current', default=0.3, type=float,
                    help='percentage of images with noise from the current state of the noise generator (default: 0.3)')
parser.add_argument('--p_history', default=0.1, type=float,
                    help='percentage of images with noise from one'
                         'previous noise generator; experience replay (default: 0.1)')
# clarification: 1 - p_clean - p_current will be filled
# with previous noise generator states of up to (1 - p_clean - p_current) / p_history times
parser.add_argument('--std_gauss', default=0.5, type=float,
                    help='if the mode is Gauss_single, this is the std (default: 0.5)')


def main():
    default_args = get_default_args()
    args = parser.parse_args()
    print('Parsed arguments:', args)
    exp_name = get_exp_name(default_args, args)
    
    # TENSORBOARD
    if args.evaluate:
        if not os.path.exists('./exp_eval/'):
            os.makedirs('./exp_eval/')
        args.exp_path = './exp_eval/' + exp_name
    else:
        if not os.path.exists('./exp/'):
            os.makedirs('./exp/')
        args.exp_path = './exp/' + exp_name
    print('Experiment name: ', exp_name)
    writer = SummaryWriter(args.exp_path)
    args.writer = writer
    for arg, val in args.__dict__.items():
        writer.add_text(arg, str(val), 0)
    args.exp_path += '/'
    
    # Get data loaders:
    # Clean ImageNet
    train_loader, test_loader, test_loader_subsample, retrain_loader \
        = setup_data_loader('IN', args)

    # ImageNet-C
    in_c_data_loaders = setup_data_loader('IN-C', args)
    args.IN_C_Results = dict()
    for name, data_loader in in_c_data_loaders.items():
        args.IN_C_Results[name] = [[0]] * 600
            
    # SIN
    if args.mode == 'ANT+SIN':
        train_loader_sin = setup_data_loader('SIN', args)

    if args.evaluate:
        # evaluate
        print("Start evaluation for model {}".format(args.model_name))
        
        # Get robust model
        model = load_robust_model(args.model_name, args.device)

        outfile_name = './Results/' + args.model_name + '_evaluation_results.txt'
        file = open(outfile_name, 'w')
        args.file = file

        acc1, acc5 = validate(test_loader, model, device=args.device)
        print("ImageNet val: Top1 accuracy: {0:.2f}, Top5 accuracy: {1:.2f}\n".format(acc1.item(),
                                                                                      acc5.item()), file=file)
        writer.add_scalar('Val/Accuracy Clean Top1', acc1, 0)
        writer.add_scalar('Val/Accuracy Clean Top5', acc5, 0)

        accuracy_on_imagenet_c(in_c_data_loaders, model, args, writer, 0)
        
        print('Training a new noise generator to test model robustness against adversarial noise')
        _ = retrain_ng_and_evaluate(model, retrain_loader, test_loader_subsample,
                                             test_loader, args, writer, 0, args.exp_path)

        file.close()

        return
    
    outfile_name = './Results/' + args.mode + '_training_results.txt'
    file = open(outfile_name, 'w')
    args.file = file
    
    # Get the classifier
    model = load_imagenet_classifier(args)
    optimizer_cl = optim.SGD(model.parameters(), lr=args.lr_classifier, momentum=args.momentum)
    scheduler_cl = torch.optim.lr_scheduler.StepLR(optimizer_cl, step_size=args.step_size_LR_decay,
                                                   gamma=args.gamma_LR_decay)
    
    # Get the noise generator
    noise_gen = get_noise_generator(args)
    optimizer_gen = optim.Adam(noise_gen.parameters(), lr=args.lr_generator)
    noise_models = {'ANT': noise_gen, 'ANT+SIN': noise_gen, 'Gauss_single': 'Gauss_single', 
                    'Gauss_mult': 'Gauss_mult', 'Speckle': 'Speckle'}

    args.iteration = 0
    # Training
    for epoch in range(args.epochs + 1):
        print('epoch: ', epoch)
        
        scheduler_cl.step()
        
        # eval
        if True:
            print('Evaluate at epoch: {}\n'.format(epoch), file=file)
            # To make the training faster, we test on a subset of the test set

            acc1, acc5 = validate(test_loader_subsample, model, device=args.device)
            print("ImageNet val: Top1 accuracy: {0:.2f}, Top5 accuracy: {1:.2f}\n".format(acc1.item(),
                                                                                          acc5.item()), file=file)
            writer.add_scalar('Val/Accuracy Clean Top1', acc1, epoch)
            writer.add_scalar('Val/Accuracy Clean Top5', acc5, epoch)
            
            print('Evaluating on IN-C')
            accuracy_on_imagenet_c(in_c_data_loaders, model, args, writer, epoch)
            
            if args.mode == 'ANT' or args.mode == 'ANT+SIN':
                acc_noisy, mse_noisy, eps_gen = test_classifier_noisy(noise_gen, test_loader_subsample,
                                                                      model, args, args.device)
                writer.add_scalar('Val/Epsilon Current NG', eps_gen, epoch)
                print('retraining the noise generator')
                # we test the performance on a subset of the
                # test set to speed up the testing and return to the training.
                noise_gen_tmp = retrain_ng_and_evaluate(model, retrain_loader, test_loader_subsample,
                                                                 test_loader_subsample, args, writer, epoch,
                                                                 args.exp_path)
                noise_gen = noise_gen_tmp
                optimizer_gen = optim.Adam(noise_gen.parameters(), lr=args.lr_generator)

        # save model states once per epoch
        if args.mode == 'ANT' or args.mode == 'ANT+SIN':
            save_models(model, noise_gen, optimizer_cl, args.exp_path, epoch, args)
        else:
            save_model(model, optimizer_cl, args.exp_path, epoch, args)
        
        # train
        if not args.mode == 'ANT+SIN':
            for batch_idx, (data, labels) in enumerate(train_loader):
                num_iteration = args.iteration
                args.iteration += 1

                data, labels = data.to(args.device), labels.to(args.device)
                
                if args.mode == 'ANT':
                    train_noise_generator(noise_gen, model, data, labels, num_iteration, writer,
                                          optimizer_gen, batch_idx=batch_idx)
                data_training, _ = generate_noisy_data(noise_models[args.mode], data, labels, args, epoch, args.exp_path)

                train_classifier(model, data_training, labels, optimizer_cl, writer, num_iteration)     
        else:
            # for ANT+SIN, we also need a batch of stylized data
            for batch_idx, (data_clean_zip, data_stylized_zip), in enumerate(zip(train_loader, train_loader_sin)):
                num_iteration = args.iteration
                args.iteration += 1

                data_clean, labels_clean = data_clean_zip                 
                data_clean, labels_clean = data_clean.to(args.device), labels_clean.to(args.device)

                data_stylized, labels_stylized = data_stylized_zip                 
                data_stylized, labels_stylized = data_stylized.to(args.device), labels_stylized.to(args.device)

                train_noise_generator(noise_gen, model, data_clean, labels_clean, num_iteration, writer,
                                      optimizer_gen, batch_idx=batch_idx)

                data_training, labels = generate_noisy_data(noise_models[args.mode], data_clean, labels_clean,
                                                            args, epoch, args.exp_path, data_stylized, labels_stylized)

                train_classifier(model, data_training, labels, optimizer_cl, writer, num_iteration)   

    file.close()

    return


if __name__ == '__main__':
    main()

