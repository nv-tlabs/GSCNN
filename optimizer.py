"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torch import optim
import math
import logging
from config import cfg

def get_optimizer(args, net):

    param_groups = net.parameters()

    if args.sgd:
        optimizer = optim.SGD(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    elif args.adam:
        amsgrad=False
        if args.amsgrad:
            amsgrad=True
        optimizer = optim.Adam(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=amsgrad
                               )
    else:
        raise ('Not a valid optimizer')

    if args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / args.max_epoch, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    if args.snapshot:
        logging.info('Loading weights from model {}'.format(args.snapshot))
        net, optimizer = restore_snapshot(args, net, optimizer, args.snapshot)
    else:
        logging.info('Loaded weights from IMGNET classifier')

    return optimizer, scheduler

def restore_snapshot(args, net, optimizer, snapshot):
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Load Compelete")
    if args.sgd_finetuned:
     print('skipping load optimizer')
    else:
        if 'optimizer' in checkpoint and args.restore_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer

def forgiving_state_restore(net, loaded_dict):
    # Handle partial loading when some tensors don't match up in size.
    # Because we want to use models that were trained off a different
    # number of classes.
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            logging.info('Skipped loading parameter {}'.format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

