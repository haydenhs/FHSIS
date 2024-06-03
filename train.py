import os
import numpy as np
import logging
import argparse
import sys
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import HSIDataset, HSITestset
from losses import *
from models.FHSIS import Net


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(args):
    # device
    device = torch.device("cuda" if args.gpu is not None else "cpu")
    # args.seed = random.randint(1, 10000)
    logging.info("Start seed: %d" % (args.seed))
    set_seed(args.seed)

    # dataset
    print('===> Loading datasets')
    train_dir = os.path.join(args.input_dir, '%s_train.h5' % args.dataset)
    test_dir = os.path.join(args.input_dir, '%s_test.h5' % args.dataset)
    trainset = HSIDataset(train_dir, sr_factor=args.sr_factor, dataset_name=args.dataset, 
        lr_type=args.lr_type, rgb_type=args.rgb_type, use_aug=True)
    evalset = HSIDataset(train_dir, sr_factor=args.sr_factor, dataset_name=args.dataset, 
        lr_type=args.lr_type, rgb_type=args.rgb_type, use_aug=False, is_eval=True)
    #testset = HSITestset(input_dir=test_dir, sr_factor=args.sr_factor, dataset_name=args.dataset,
    #    lr_type=args.lr_type, rgb_type=args.rgb_type)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    eval_loader = DataLoader(evalset, batch_size=8, num_workers=4, shuffle=False)
    #test_loader = DataLoader(testset, batch_size=8, num_workers=4, shuffle=False)

    # model
    print('===> Building model')
    model = Net(n_basis=args.num_basis, factor=args.factor)
    model_type =  args.dataset + '_' + args.lr_type + '_' + args.rgb_type + '_x' + str(args.sr_factor) + '_' + str(args.factor) + '_' + str(args.num_basis)
    #model_type = model_type + '_lvl_' + str(args.noise_level)
    print(model_type)
    if torch.cuda.device_count() > 1:
        print("===> Using {} GPUs: {}".format(torch.cuda.device_count(), args.gpus))
        model = torch.nn.DataParallel(model)

    # resume training
    starting_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
        starting_epoch = int(args.resume[-6:-4]) # assume less than 100 epochs
        logging.info('resume at %d epoch' % starting_epoch)
    model.to(device).train()
    writer = SummaryWriter()
    # loss function
    color_loss = nn.MSELoss()
    L1_loss = nn.L1Loss()
    TV_loss = TVLoss()
    r_loss = RGBLoss()

    print("===> Setting optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    ckpt_dir = os.path.join(args.checkpoint_dir, model_type)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print('Checkpoint save directory is {}'.format(ckpt_dir))
    # training
    print("===> Start training")
    progress_bar = tqdm(total= (args.num_epoch-starting_epoch) * len(trainset), dynamic_ncols=True)
    for epoch in range(starting_epoch, args.num_epoch):
        losses = []
        adjust_learning_rate(args.lr, optimizer, epoch+1)
        for i, (lr_rgb, lr, rgb, gt) in enumerate(train_loader):
            progress_bar.update(n=args.batch_size)

            #if args.noise_level > 0:
            #    lr = lr + torch.randn(lr.shape) * args.noise_level / 255.
            lr_rgb = lr_rgb.to(device, non_blocking=True)
            lr, rgb, gt = lr.to(device, non_blocking=True), rgb.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            # back prop
            optimizer.zero_grad()
            out_lr, out = model(lr, rgb, lr_rgb)
            loss1 = color_loss(out, gt)
            loss2 = color_loss(out_lr, lr)
            loss3 = r_loss(out, rgb)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()

            # print statistics
            losses.append(loss.item())
            if (i+1) % args.log_interval == 0:
                writer.add_scalar('Total Loss', loss.item(), (epoch+1)*len(train_loader)+i+1)
                writer.add_scalar('Loss/HR-HSI', loss1.item(), (epoch+1)*len(train_loader)+i+1)
                writer.add_scalar('Loss/RGB', loss3.item(), (epoch+1)*len(train_loader)+i+1)
                writer.add_scalar('Loss/LR-HSI', loss2.item(), (epoch+1)*len(train_loader)+i+1)
                progress_bar.set_description(
                        "===> Epoch[{}]({}/{}): Loss:{:.6f} L2:{:.6f} Recon:{:.6f}".format(
                    epoch + 1, i + 1, len(train_loader), loss.item(), loss1.item(), loss2.item()+loss3.item()))

        #scheduler.step()
        val_loss = validate(args, eval_loader, model)
        logging.info("===> {}\tEpoch {} Training Complete: Avg. Train Loss: {:.6f} Avg. Test Loss: {:.6f}".format(
                time.ctime(), epoch+1, np.mean(losses), val_loss))

        # save models
        if (epoch+1) % args.model_save_freq == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(ckpt_dir, 'model_%d.pth' % (epoch+1)))
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model_%d.pth' % (epoch+1)))
    progress_bar.close()
    torch.save(model.state_dict(), args.result_dir + model_type + '_%d.pth' % (epoch+1))
    logging.info("Finish training!")


def validate(args, loader, model):
    device = torch.device("cuda" if args.gpu else "cpu")
    # switch to evaluate mode
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (lr_rgb, lr, rgb, gt) in enumerate(loader):
            lr_rgb = lr_rgb.to(device)
            lr, rgb, gt = lr.to(device), rgb.to(device), gt.to(device)
            # no back prop
            _, y = model(lr, rgb, lr_rgb)
            loss = F.mse_loss(y, gt)
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = start_lr * (0.2 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="command options for training")
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument('--input_dir', type=str, default='/home/CAVE_h5')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='CAVE')
    parser.add_argument('--lr_type', type=str, default='wald')
    parser.add_argument('--rgb_type', type=str, default='Nikon')
    parser.add_argument('--noise_level', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=40)
    parser.add_argument('--model_save_freq', type=int, default=10)
    parser.add_argument('--sr_factor', type=int, default=32)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--num_basis', type=int, default=5)
    parser.add_argument('--resume', type=str, help='continue training')
    args = parser.parse_args()

    # Create checkpoint & result dirs
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # specify which gpu to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set logger
    log_name = "Fusion_train_%s.txt" % time.ctime()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        filename=os.path.join(args.result_dir, log_name),
                        filemode='w')

    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)
    train(args)
