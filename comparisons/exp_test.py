import os
import numpy as np
import argparse
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import scipy.io as sio
from metrics import quality_assessment
from dataset import HSITestset
from models.GPPNN import GPPNN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def test(args):
    # device
    device = torch.device("cuda" if args.gpu is not None else "cpu")
    # args.seed = random.randint(1, 10000)
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    input_dir = os.path.join(os.path.join(args.input_dir, args.dataset + '_h5'), args.dataset + '_test.h5')

    # data
    print('===> Loading datasets')
    testset = HSITestset(input_dir=input_dir, sr_factor=args.sr_factor, dataset_name=args.dataset, lr_type=args.lr_type, rgb_type=args.rgb_type)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, shuffle=False)
    test_name = 'GPPNN_' + args.dataset + '_' + args.lr_type + '_' + args.rgb_type + '_x' + str(args.sr_factor)
    # model
    print('===> Building model')
    model = GPPNN()
    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model)

    # resume training and start testing
    model.load_state_dict(torch.load(args.resume))
    model.to(device)

    # loss function
    color_loss = nn.MSELoss()

    # testing
    print("===> Start testing")
    #noise_level = 25. / 255
    logger = pd.DataFrame()
    result = []
    with torch.no_grad():
        model.eval()
        for i, (lr_rgb, lr, rgb, gt) in enumerate(test_loader):
            lr = lr.to(device)
            rgb = rgb.to(device)
            gt = gt.to(device)
            y = model(lr, rgb)
            
            # print statistics for HSI
            y = y.squeeze().permute(1,2,0).cpu().numpy()
            gt = gt.squeeze().permute(1,2,0).cpu().numpy()
            rgb = rgb.squeeze().permute(1,2,0).cpu().numpy()
            indices = quality_assessment(gt, y, data_range=1., ratio=args.sr_factor)
            logger = logger.append([indices], ignore_index=True)
            result.append(y)
            """
            psnrs.append(compare_psnr(rgb_gt, result_rgb))
            ssims.append(compare_ssim(rgb_gt, result_rgb, multichannel=True))
            """


            # save comparison images to file
            #temp = np.squeeze(np.concatenate((gt[:,:,[2,24,29]], y[:,:,[2,24,29]]), axis=1))*24.
            #temp = np.clip(temp*255, 0, 255).astype(np.uint8)
            #temp = Image.fromarray(temp)
            #result_path = os.path.join(args.result_dir, test_name)
            #if not os.path.exists(result_path):
            #    os.makedirs(result_path)
            #temp.save(os.path.join(result_path, 'test_%d.jpg' % (i+1)))
        result = np.stack(result)
        sio.savemat('./'+ test_name + '.mat',{'out':result})
        logger.to_csv('./' + test_name + '.csv')
        print("===> {}\t Testing RGB Complete: Avg. RMSE: {:.6f}, Avg. PSNR: {:.2f}, Avg. SSIM: {:.2f}".format(
            time.ctime(), logger['RMSE'].mean(), logger['PSNR'].mean(), logger['SSIM'].mean()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="command options for training")
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--sr_factor", type=int, default=32)
    parser.add_argument("--dataset", type=str, default='CAVE')
    parser.add_argument('--lr_type', type=str, default='wald')
    parser.add_argument('--rgb_type', type=str, default='Nikon')
    parser.add_argument('--input_dir', type=str, default='/home/')
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--resume', type=str, help='continue training')
    args = parser.parse_args()
    test(args)
