import numpy as np
import cv2    

# Nikon D700 spectral response matrix
srf_nikon = [[0.00784314, 0.00492611, 0.02788845],
       [0.00392157, 0.00492611, 0.03984064],
       [0.00392157, 0.00492611, 0.05976096],
       [0.00392157, 0.00492611, 0.07569721],
       [0.00392157, 0.00492611, 0.0996016 ],
       [0.00392157, 0.00492611, 0.11553785],
       [0.        , 0.00985222, 0.11952192],
       [0.        , 0.01970443, 0.11553785],
       [0.        , 0.02955665, 0.10756972],
       [0.        , 0.03940887, 0.08764941],
       [0.        , 0.05418719, 0.06374502],
       [0.        , 0.07881773, 0.03585657],
       [0.        , 0.09359606, 0.00796813],
       [0.        , 0.10344828, 0.        ],
       [0.        , 0.09852216, 0.        ],
       [0.        , 0.08866995, 0.        ],
       [0.00784314, 0.07881773, 0.        ],
       [0.02352941, 0.06896552, 0.        ],
       [0.04313726, 0.05418719, 0.        ],
       [0.06666667, 0.03448276, 0.        ],
       [0.08235294, 0.02463054, 0.00398406],
       [0.08627451, 0.01477833, 0.00398406],
       [0.08235294, 0.00985222, 0.00398406],
       [0.07843138, 0.00985222, 0.00398406],
       [0.07843138, 0.00492611, 0.00398406],
       [0.07450981, 0.00492611, 0.00398406],
       [0.07450981, 0.00985222, 0.00398406],
       [0.07058824, 0.00985222, 0.00398406],
       [0.07058824, 0.00985222, 0.00398406],
       [0.06666667, 0.00985222, 0.00398406],
       [0.06666667, 0.00985222, 0.00398406]]

# CIE_1964 spectral response matrix (from NTIRE2020 challenge)
srf_cie = [[0.00163989, 0.00017189, 0.00738091],
       [0.00727159, 0.0007509 , 0.03341292],
       [0.0175484 , 0.00183445, 0.08345738],
       [0.02700405, 0.00331678, 0.13330979],
       [0.03292998, 0.0053236 , 0.16881948],
       [0.03181165, 0.00767158, 0.17118107],
       [0.02593944, 0.01099427, 0.14977657],
       [0.01678688, 0.01588154, 0.11306463],
       [0.00690868, 0.02174731, 0.06625886],
       [0.00138779, 0.0290834 , 0.03563446],
       [0.00032747, 0.03951536, 0.01875045],
       [0.00321504, 0.05203295, 0.0096149 ],
       [0.01010458, 0.06532683, 0.00520966],
       [0.02029438, 0.07505642, 0.00261311],
       [0.03233254, 0.08249825, 0.00117359],
       [0.04546681, 0.08505153, 0.00034222],
       [0.06051852, 0.08552997, 0.        ],
       [0.07540143, 0.08194631, 0.        ],
       [0.08702974, 0.07451812, 0.        ],
       [0.09598535, 0.06666877, 0.        ],
       [0.09645476, 0.05645807, 0.        ],
       [0.08843023, 0.0452771 , 0.        ],
       [0.07348279, 0.03413661, 0.        ],
       [0.05556212, 0.02431182, 0.        ],
       [0.03703475, 0.01542171, 0.        ],
       [0.02302655, 0.0092304 , 0.        ],
       [0.01309256, 0.00516958, 0.        ],
       [0.00697335, 0.00272714, 0.        ],
       [0.0035056 , 0.00136399, 0.        ],
       [0.00171125, 0.00066452, 0.        ],
       [0.00082184, 0.00031883, 0.        ]]

# MHF-net spectral response matrix
srf_mhf = [[ 1.81917539,  1.91090986,  2.31788059],
       [-0.80037055, -0.67901933, -0.98395347],
       [ 0.34628659, -0.1648808 ,  0.35592346],
       [-0.95393052, -1.08797541, -0.94794481],
       [ 0.81049198,  0.72168977,  1.12478105],
       [ 0.72803277,  1.13135596,  0.34264406],
       [-0.27175234, -0.45433045,  0.64637416],
       [-1.63988654, -0.66018172, -0.39871266],
       [ 0.3609929 , -1.01968252, -1.33302549],
       [ 0.18591477,  0.78421091,  1.74980773],
       [ 0.89964696,  2.04158313,  0.54322832],
       [-0.62337427, -1.90995508, -0.93078649],
       [-0.19473638,  1.14634224,  0.39670599],
       [-0.58764393, -0.33292838, -0.44983951],
       [-0.04780366,  0.18994016,  0.07169621],
       [ 1.26456747,  1.20471584,  0.44089013],
       [-0.48711782, -1.33262283, -0.36560245],
       [ 0.9644351 ,  1.34467968, -0.02758613],
       [-1.64015235, -2.15581789, -0.75912557],
       [ 1.28672635,  2.16236456,  1.12162198],
       [-0.43492538, -0.1989385 , -0.76597727],
       [ 1.01979613,  0.32843133,  0.4600139 ],
       [-0.68402533, -0.89053625, -0.16595202],
       [-0.79537862, -0.28565256,  0.0038303 ],
       [ 1.92048821,  0.97414956,  0.31337209],
       [-0.99405394, -1.29194543, -1.36139351],
       [ 0.72783617,  0.93611668,  1.34039412],
       [ 0.73274281, -0.12402873, -0.06628292],
       [-1.468649  , -0.66163413, -0.6411167 ],
       [ 0.57134337,  0.92687732,  0.16147575],
       [ 0.24192663, -0.22956816,  0.243834  ]]


def real_spadown(img, factor):
    filter_zoo = [[5, 1], [7, 1/2], [9, 2], [13, 4], [15, 1.5]]
    chosen = np.random.randint(0, 5, 1)[0]
    k_size, sigma = filter_zoo[chosen]
    return wald_protocol(img, factor, k_size=k_size, sigma=sigma)


def wald_protocol(img, factor, k_size=7, sigma=None):
    H, W, C = img.shape
    start_pos = factor // 2 - 1
    margin = k_size // 2
    if sigma is None:
        sigma = (1 / (2 * 2.7725887 / factor**2)) ** 0.5
    """
    gaussian_kernel = cv2.getGaussianKernel(k_size, sigma)
    kernel_2D = gaussian_kernel * gaussian_kernel.T
    blur = np.zeros_like(img)
    for i in range(C):
        blur[:,:,i] = signal.convolve2d(img[:,:,i], kernel_2D, mode='same', boundary='wrap')
    """
    blur = cv2.copyMakeBorder(img, margin, margin, margin, margin, borderType=cv2.BORDER_WRAP)
    blur = cv2.GaussianBlur(blur, (k_size, k_size), sigma)
    out = blur[margin+start_pos:margin+H:factor, margin+start_pos:margin+W:factor, :]
    return out

def box_filter(img, factor):
    H, W, C = img.shape
    out = np.zeros((H//factor, W//factor, C))

    for i in range(factor):
        for j in range(factor):
            out = out + img[i:H:factor, j:W:factor, :] / factor / factor
    return out

def downsample(img, factor, mode='bicubic'):
    H, W, _ = img.shape
    if mode == 'bicubic':
        out = cv2.resize(img, (H//factor, W//factor), interpolation=cv2.INTER_CUBIC)
    elif mode == 'wald':
        out = wald_protocol(img, factor)
    elif mode == 'box':
        out = box_filter(img, factor)
    elif mode == 'real':
        out = real_spadown(img, factor)
    else:
        raise NotImplementedError
    return out

def choose_srf(rgb_type):
    if rgb_type == 'Nikon':
        R = srf_nikon
    elif rgb_type == 'CIE':
        R = srf_cie
    elif rgb_type == 'MHF':
        R = srf_mhf
    else:
        raise NotImplementedError
    return R
