import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import data
from experiment import Experiment
import os
import faulthandler
from torch.utils.tensorboard import SummaryWriter


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
faulthandler.enable()


parser = argparse.ArgumentParser(description='Acquire some parameters for fusion restore')
parser.add_argument('--lr', type=float, default = 1e-4, help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=24, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GP Us to use')
parser.add_argument('--num_workers', type=int, default=12, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('output'), help='the output directory')
parser.add_argument('--logpath', type=Path, default=Path('output'))
# 获取对输入数据进行预处理时的一些参数=========
parser.add_argument('--train_dir', type=Path, default=('.../train_data'), help='the training data directory')
parser.add_argument('--val_dir', type=Path, default=('.../test_data'), help='the validation data directory')
parser.add_argument('--test_dir', type=Path, default=('.../test_data'), help='the test data directory')
parser.add_argument('--image_size', type=int, nargs='+', default=[1640, 1640], help='the size of the coarse image (width, height)')
parser.add_argument('--patch_size', type=int, nargs='+', default=[128, 128], help='the coarse image patch size for training restore')  #128
parser.add_argument('--patch_stride', type=int, nargs='+', default=64, help='the coarse patch stride for image division')    #64
opt = parser.parse_args()


if not torch.cuda.is_available():
    opt.cuda = False
if opt.cuda:
    cudnn.benchmark = True
    cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()


if __name__ == '__main__':
    experiment = Experiment(opt)
    if opt.epochs > 0:
        experiment.train(opt.train_dir, opt.val_dir,
                         opt.patch_size, opt.patch_stride, opt.batch_size,
                         num_workers=opt.num_workers, epochs=opt.epochs)
    experiment.test(opt.test_dir, num_workers=opt.num_workers)









