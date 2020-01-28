import argparse

from utils import *

parser = argparse.ArgumentParser(description='Evaluation of Models')
parser.add_argument('--model_name', default='clean', type=str,
                    help='which model should be evaluated',
                    choices='clean, ANT, ANT-SIN, Speckle, Gauss_mult, Gauss_sigma_0.5')
parser.add_argument('--imagenetc-path', metavar='DIR',
                    default='./data/ImageNet-C/imagenet-c/')
parser.add_argument('--datadir-clean', metavar='DIR',
                    default='./data/IN/raw-data/', help='path to dataset')
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 30)')
parser.add_argument('-tb', '--test-batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--device', default='cuda', type=str,
                    help='cuda or cpu')


def main():
    args = parser.parse_args()

    model = load_model(args.model_name, args.device)

    valdir = osp.join(args.datadir_clean, 'val')
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

    in_c_data_loaders = get_in_c_data_loaders(args)

    # evaluate
    print("Start evaluation for model {}".format(args.model_name))
    outfile_name = './Results/' + args.model_name + '_results.txt'
    file = open(outfile_name, 'w')
    args.file = file

    acc1, acc5 = validate(val_loader, model)
    print("ImageNet val: Top1 accuracy: {0:.2f}, Top5 accuracy: {1:.2f}\n".format(acc1.item(), acc5.item()), file=file)

    accuracy_on_imagenet_c(in_c_data_loaders, model, args)

    file.close()

    return


if __name__ == '__main__':
    main()

