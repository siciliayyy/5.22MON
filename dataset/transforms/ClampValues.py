from typing import Tuple

import numpy as np
import torch


class ClampValues:
    """
    Clamp voxels into the correct range
    Any values outside of the provided range are set as the min or max of that range
    """

    def __init__(self, voxel_range: Tuple):
        """
        Initialize the ClampValues transform

        :param voxel_range: Inclusive range of voxel values to clamp to

        """
        self.voxel_range = voxel_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Call logic for ClampValues

        :param img: image tensor with dimensions (B x D x H x W)
        :return:    image tensor with values clamped to the provided range
        """

        ccc = img.cpu().numpy()
        if np.min(ccc) < -10:
            # img = (img - np.min(ccc)) / (np.max(ccc) - np.min(ccc)) * 255
            img = torch.clamp(img, 0, 255)

        elif np.min(ccc) == 0:
            img = torch.clamp(img, 0, 1)

        '''
        output featuremap of preprocess raw_img, output ,mask
        '''
        channel = 36
        # bbb = ccc.squeeze()
        # bbb = bbb[channel]
        # ddd = img.cpu().numpy()
        # ddd = ddd.squeeze()
        # ddd = ddd[channel]
        # plot_img_mask_and_real_and_change(bbb, ddd, channel)
        return img
