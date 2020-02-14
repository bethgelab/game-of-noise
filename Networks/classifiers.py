from Networks.imagenet_model import *


def load_imagenet_classifier(args):
    if args.arch == 'resnet18':
        classifier = ZeroOneResNet18(device=args.device, pretrained=True)
    elif args.arch == 'resnet50':
        classifier = ZeroOneResNet50(device=args.device, pretrained=True)
    elif args.arch == 'resnet152':
        classifier = ZeroOneResNet152(device=args.device, pretrained=True)
    elif args.arch == 'inceptionV3':
        classifier = ZeroOneInceptionV3(device=args.device, pretrained=True)
    else:
        raise Exception(f'classifier for dataset: {args.arch} is not available')
    
    classifier = classifier.to(args.device)
    
    return classifier
