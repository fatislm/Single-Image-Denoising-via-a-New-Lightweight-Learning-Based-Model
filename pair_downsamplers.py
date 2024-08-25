import torch
import torch.nn.functional as F

def pair_downsampler1(img):
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

def pair_downsampler2(img):
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.5, 0], [0.5, 0, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0, 0], [0, 0.5, 0]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

def pair_downsampler3(img):
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.35], [0.65, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.35, 0], [0, 0.65]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2
