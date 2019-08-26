"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from datasets import cityscapes
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader

def setup_loaders(args):
    '''
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    '''

    if args.dataset == 'cityscapes':
        args.dataset_cls = cityscapes
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
    else:
        raise

    args.num_workers = 4 * args.ngpu
    if args.test_mode:
        args.num_workers = 0 #1

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Geometric image transformations
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                           False,
                                           pre_size=args.pre_size,
                                           scale_min=args.scale_min,
                                           scale_max=args.scale_max,
                                           ignore_index=args.dataset_cls.ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]
 
    #if args.rotate:
    #    train_joint_transform_list += [joint_transforms.RandomRotate(args.rotate)]

    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # Image appearance transformations
    train_input_transform = []
    if args.color_aug:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=args.color_aug,
            contrast=args.color_aug,
            saturation=args.color_aug,
            hue=args.color_aug)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
    else:
        pass

    train_input_transform += [standard_transforms.ToTensor(),
                              standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()
    
    target_train_transform = extended_transforms.MaskToTensor()

    if args.dataset == 'cityscapes':
        city_mode = 'train' ## Can be trainval
        city_quality = 'fine'
        train_set = args.dataset_cls.CityScapes(
            city_quality, city_mode, 0, 
            joint_transform=train_joint_transform,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            cv_split=args.cv)
        val_set = args.dataset_cls.CityScapes('fine', 'val', 0, 
                                              transform=val_input_transform,
                                              target_transform=target_transform,
                                              cv_split=args.cv)
    else:
        raise
    
    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False, sampler = val_sampler)

    return train_loader, val_loader,  train_set

