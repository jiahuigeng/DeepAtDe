import time
import argparse
import glob
import os
import os.path as osp
import numpy as np

import torch.optim

from src.logger import get_logger
from src.saver import Saver
from src.models import *
from src.training import train
from src.utils import *
from src.data import *

def main(args):

    if args.task == "train":
        save_dir_name = args.dataset + '_' + args.task + '_' + args.model + '_' + str(args.batchsize) + "_" + str(
            args.load_model) + '_' + str(args.exp_name)
    elif args.task == "attack":
        save_dir_name = args.dataset + '_' + args.task + '_' + args.model + '_' + args.data_mode + '_' + str(
            args.batchsize) + "_" + str(
            args.load_model) + '_' + str(args.exp_name)
    else:
        NotImplementedError("undefined task")

    torch.manual_seed(7)
    np.random.seed(7)
    directory = osp.join('results', save_dir_name)
    if not osp.exists(directory):
        os.mkdir(directory)
    runs = sorted(glob.glob(osp.join(directory, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    experiment_dir = osp.join(directory, 'experiment_{}'.format(str(run_id)))

    if not osp.exists("results"):
        os.mkdir("results")
    if not osp.exists(directory):
        os.mkdir(directory)
    if not osp.exists(experiment_dir):
        os.mkdir(experiment_dir)

    logger = get_logger(experiment_dir)
    saver = Saver(experiment_dir)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    logger.info(args)
    logger.info(experiment_dir)


    if args.task == "train":
        trn_loader, tst_loader, channel, num_classes, img_shape = get_dataset(args)
        logger.info("channel: %d, num_classes: %d, img_shape: %s" % (channel, num_classes, str(img_shape)))
        if args.model == "lenet":
            model = LeNet(args.vb, img_shape=img_shape, channel=channel, hidden=args.hidden, num_classes=num_classes)
        elif args.model == "resnet":
            model = ResNet18()
        elif args.model == "mlp":
            model = MLP(args.vb, img_shape=img_shape, channel=channel, hidden=args.hidden, num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train(model, optimizer, trn_loader, tst_loader, args.local_epochs, False, device)

    else:
        dst, channel, num_classes, img_shape = get_dataset(args)











if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("--task", type=str, default="train", choices=["train", "attack"])
    parser.add_argument("--vb", type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagen'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'resnet', 'mlp'],
                        help='model name (default: lenet)')
    parser.add_argument('--num-layers', type=int, default=2, help='number of layers')
    parser.add_argument("--hidden", type=int, default=0, help="hidden layer dimension")
    parser.add_argument('--load-model', type=bool, default=False, help='whether to use trained model (default: False)')
    parser.add_argument('--model-path', type=str, default=None, help='path to the trained model (default: False)')
    parser.add_argument('--mode', type=str, default='gradients', choices=['gradients', 'weights'],
                        help='method name (default: gradients)')
    # parser.add_argument('--task', type=str, default='image', choices=['image', 'label'],
    #                     help='restoration task name (default: image)')
    parser.add_argument('--num-seeds', type=int, default=5, help='number of seeds for dummy image initialization')
    parser.add_argument('--batchsize', type=int, default=64, help='number of dummy images in a batch (batch size)')
    # parser.add_argument('--method', type=str, default='dlg', choices=['dlg', 'idlg', 'gradinversion'], help='method name (default: dlg)')
    parser.add_argument('--data-init', type=str, default="uniform")
    parser.add_argument('--data-mode', type=str, default="random",
                        choices=['random', 'repeat', "unique", "single", "factor"])
    parser.add_argument('--data-scale', type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--local-lr', type=float, default=0.1)
    parser.add_argument('--local-epochs', type=int, default=100)
    parser.add_argument('--local-bs', type=int, default=1)
    parser.add_argument("--multi-steps", type=int, default=0)
    parser.add_argument('--dummy-norm', type=str, default="scale", choices=['clip', 'scale'])
    parser.add_argument('--tv-alpha', type=float, default=200)
    parser.add_argument('--clip-alpha', type=float, default=200)
    parser.add_argument('--scale-alpha', type=float, default=200)
    parser.add_argument('--l2-alpha', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--Iteration', type=int, default=600)
    parser.add_argument('--num-exp', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default="", required=True)
    parser.add_argument('--skip', type=bool, default=False)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument('--period', type=int, default=120)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    # args.cuda = torch.cuda.is_available()
    main(args)







