import numpy as np
import torch
import time
from models.FHSIS import Net
from models.GPPNN import GPPNN
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def computeTime(model, device='cuda'):
    if device == 'cuda':
        model = model.cuda()

    model.eval()
    i = 0
    time_spent = []
    with torch.no_grad():
        while i < 1000:
            rgb = torch.randn(1, 3, 512, 512)
            lr = torch.randn(1, 31, 16, 16)
            lr_rgb = torch.randn(1, 3, 16, 16)
            start_time = time.time()
            rgb, lr, lr_rgb = rgb.cuda(), lr.cuda(), lr_rgb.cuda()
            #rgb, lr = rgb.cuda(), lr.cuda()
            #_ = model(lr, rgb, lr_rgb)
            out,_ = model(lr, rgb, lr_rgb)
            out = out.cpu()
            if device == 'cuda':
                torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            if i != 0:
                time_spent.append(time.time() - start_time)
            i += 1
        print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)*1e3))
    

model = Net(n_basis=5)
#model = GPPNN()
computeTime(model, device='cuda')
