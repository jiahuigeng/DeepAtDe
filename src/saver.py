import os
import os.path as osp
import glob
import numpy as np
import torchvision
import skimage.io
import torch
import logging

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

class Saver(object):

    def __init__(self, experiment_dir):

        self.experiment_dir = experiment_dir
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        if not osp.exists(osp.join(self.experiment_dir, "log.csv")):
            log_headers = ["id","gt_label","pred_label","label_acc","grad_diff","mse","ssim","psnr","lpips","method","lr","num_dummy","tv_alpha","l2_alpha","clip_alpha","scale_alpha"]
            with open(osp.join(self.experiment_dir, "log.csv"), 'w') as f:
                f.write(','.join(log_headers) + '\n')

    def save_batch(self, data, label, start_idx=0, iteration=-1):
        """
        @:param data: image data, torch tensor [(N,C,H,W)]
        @:param label: labels, torch tensor (N,)
        @:param iteration: int
        """
        imgs = torchvision.utils.make_grid(data)
        imgs = imgs.numpy()
        lbls = label.numpy()
        imgs = np.transpose(imgs, (1, 2, 0))
        logger.info(imgs.shape)
        file_name = 'start_%d_lbls_'%start_idx + '-'.join('%d' % lbl for lbl in lbls)+'_iter%d'%iteration+'.png'
        skimage.io.imsave(osp.join(self.experiment_dir, file_name), imgs)

    def save_result_imgs(self, data, label, start_idx=0, method='dlg'):
        """
        @:param data: image data, list of torch tensor (N,C,H,W)
        @:param label: labels, torch tensor (N,)
        @:param iteration: int
        """
        viz = None
        for imgs in data:
            for i in range(len(imgs)):
                imgs[i] = (imgs[i] - torch.min(imgs[i]))/(torch.max(imgs[i]) - torch.min(imgs[i]))
            imgs = torchvision.utils.make_grid(imgs)
            imgs = imgs.numpy()
            imgs = np.transpose(imgs, (1, 2, 0))
            empty_space = np.zeros((5, imgs.shape[1], 3))
            if viz is not None:
                viz = np.vstack((viz, imgs, empty_space))
            else:
                viz = np.vstack((imgs, empty_space))
        lbls = label.numpy()
        file_name = 'start_%d_lbls_'%start_idx + '-'.join('%d' % lbl for lbl in lbls) + '%s'%method +'.png'
        skimage.io.imsave(osp.join(self.experiment_dir, file_name), viz)

    def save_metrics(self, id, gt_label, pred_label, label_acc, mse, ssim, psnr, lpips, method, lr, num_dummy, tv_alpha, l2_alpha, clip_alpha, scale_alpha):
        gt_label = gt_label.numpy()
        pred_label = pred_label.numpy()
        with open(osp.join(self.experiment_dir, 'log.csv'), 'a') as f:
            log = [id, '_'.join('%d' % lbl for lbl in gt_label), '_'.join('%d' % lbl for lbl in pred_label), label_acc, mse, ssim, psnr, lpips, method,lr, num_dummy, tv_alpha, l2_alpha, clip_alpha, scale_alpha]
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def save_metrics1(self, id, gt_label, pred_label, label_acc, grad_diff, mse, ssim, psnr, lpips, method):
        gt_label = gt_label.numpy()
        pred_label = pred_label.numpy()
        with open(osp.join(self.experiment_dir, 'log.csv'), 'a') as f:
            log = [id, '-'.join('%d' % lbl for lbl in gt_label), '_'.join('%d' % lbl for lbl in pred_label), label_acc, grad_diff, mse, ssim, psnr, lpips, method]
            log = map(str, log)
            f.write(','.join(log) + '\n')


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    # parser.add_argument('--save_dir_name', type=str, default='test', help='save dir name')
    # args = parser.parse_args()
    # saver = Saver(args)
    #
    # import torch
    # import torchvision.transforms as transforms
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # saver.save_batch(images, labels)

    data = [torch.rand(3, 32, 32*8) for i in range(8)]
    viz = None
    for imgs in data:
        imgs = torchvision.utils.make_grid(imgs)
        imgs = imgs.numpy()
        imgs = np.transpose(imgs, (1, 2, 0))
        empty_space = np.zeros((5, imgs.shape[1], 3))
        if viz is not None:
            viz = np.vstack((viz, imgs, empty_space))
        else:
            viz = np.vstack((imgs, empty_space))

    # a = np.random.rand(32, 32 * 8, 3)
    # b = np.zeros((10, 32*8, 3))
    # c = np.random.rand(32,32*8,3)
    # viz = np.vstack((a, b, c))
    import matplotlib.pyplot as plt
    plt.imshow(viz)
    plt.show()
    logger.info(viz.shape)
