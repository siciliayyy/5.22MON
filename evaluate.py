import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from torch.utils.data import DataLoader

from ShowingOutput import show_slices
from dataset.prepare_dataset import prepare_dataloader
from evaluation.dice_score import dice_score
from model.DenseUNet3d import DenseUNet3d
from model.vnet import *


def show_featuremap(output, mask, dice_scores):
    output = (torch.sigmoid(output) > 0.5).float()
    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    output = np.squeeze(output)  # output

    mask = mask.cpu().detach().numpy()
    mask.astype(np.int64)
    mask = np.squeeze(mask)
    mask = np.squeeze(mask)

    fig = plt.figure()
    plt.title(dice_scores)

    a = fig.add_subplot(2, 3, 1)
    a.set_title('output[16]')
    plt.imshow(output[16])

    b = fig.add_subplot(2, 3, 4)
    b.set_title('mask[16]')
    plt.imshow(mask[16])

    c = fig.add_subplot(2, 3, 2)
    c.set_title('output[26]')
    plt.imshow(output[26])

    d = fig.add_subplot(2, 3, 5)
    d.set_title('mask[26]')
    plt.imshow(mask[26])

    e = fig.add_subplot(2, 3, 3)
    e.set_title('output[36]')
    plt.imshow(output[36])

    f = fig.add_subplot(2, 3, 6)
    f.set_title('mask[36]')
    plt.imshow(mask[36])

    num = len(os.listdir('./result/20240423'))
    plt.savefig('./result/20240423/predict_{}'.format(num), dpi=600)


def evaluate(
        model: VNet,
        device: torch.device,
        dataloader: DataLoader,
        dim: Optional[int] = 1
) -> float:
    """
    Evaluates the model by dice score on given data

    :param model:       model to evaluate
    :param device:      device to evaluate on, usually cpu or gpu
    :param dataloader:  DataLoader object which contains the images
    :param dim:         dimension to evaluate dice score over
    :return:            average dice score
    """
    dice_scores = []

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            id, volume, segmentation = data
            eee = volume.numpy()
            volume = volume.to(device, dtype=torch.float)

            if dim == 1:
                segmentation = torch.clamp(segmentation, 0, 1)

            segmentation = segmentation.to(device, dtype=torch.uint8)

            output = model(volume)

            bbb = (torch.sigmoid(output) > 0.5).float().cpu().detach().numpy()
            bbb = np.squeeze(bbb)
            bbb = np.squeeze(bbb)
            bbb = bbb*2000

            eee = np.squeeze(eee)
            eee = np.squeeze(eee)

            ccc = segmentation.float().cpu().detach().numpy()
            ccc = np.squeeze(ccc)
            ccc = np.squeeze(ccc)

            dice_scores.append(dice_score((torch.sigmoid(output) > 0.5).float(), segmentation.float()))
            show_slices(str(id), eee, bbb, ccc, dice_scores[-1], True)
            show_featuremap(output, segmentation, dice_scores[-1])
    return sum(dice_scores) / len(dice_scores)


if __name__ == '__main__':
    files_saving_nii = './OUTPUT_nii'
    files_saving_featuremap = './OUTPUT_featuremap'
    weight_path = 'result/bestmodel/9.11_128128128/Epoch149_LS-0.951_DC-0.826.pth'

    with open("./config.yaml", "r") as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info('Using device {}'.format(device))

    if not os.path.exists(files_saving_nii):
        os.makedirs(files_saving_nii)
    if not os.path.exists(files_saving_featuremap):
        os.makedirs(files_saving_featuremap)


    if config["model"] == 'DenseUNet3d':
        net = DenseUNet3d()
    else:
        net = VNet()
    net.to(device=device)
    s = torch.load(weight_path)
    net.load_state_dict(s)
    torch.backends.cudnn.benchmark = True

    testloader, n_test = prepare_dataloader(config, train=1)
    dice_coeff = evaluate(model=net, device=device, dataloader=testloader, dim=1)
    print(dice_coeff)
