import argparse

from utils import *

parser = argparse.ArgumentParser(description='Evaluation of Models')
parser.add_argument('--model_name', default='ANT-SIN', type=str,
                    help='which model should be evaluated')
parser.add_argument('--imagenetc-path', type=str,
                    default='./data/ImageNet-C/imagenet-c/')
parser.add_argument('--datadir-clean', metavar='DIR',
                    default='./data/ImageNet/' , help='path to dataset')
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 30)')
parser.add_argument('-tb', '--test-batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')

def main():

    args = parser.parse_args()

    model = load_model(args.model_name)
    
    valdir   = osp.join(args.datadir_clean, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    
    in_c_data_loaders = get_IN_C_data_loaders(args)
    args.IN_C_Results = dict()
    for name, data_loader in in_c_data_loaders.items():
        args.IN_C_Results[name] = [[0]] * 7
        
        
    # evaluate
    print("Start evaluation for model {}".format(args.model_name))
    outfile_name = './Results/' + args.model_name + '_results.txt'
    file = open(outfile_name, 'w')
    args.file = file
    
    acc1, acc5 = validate(val_loader, model, args)
    print(acc1)
    file.write("Top1 accuracy on ImageNet val: {0:.2f}\n".format(acc1.item()))
    file.write("Top5 accuracy on ImageNet val: {0:.2f}\n".format(acc5.item()))
    print("Top1 accuracy on ImageNet val: {0:.2f}".format(acc1.item()))
    print("Top5 accuracy on ImageNet val: {0:.2f}".format(acc5.item()))
    
    accuracy_on_imagenet_c(in_c_data_loaders, model, args)
    
    file.close()
    
    return
    
if __name__ == '__main__':
    main()

