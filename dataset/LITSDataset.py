import json
import os
import re
from os import listdir
from os.path import splitext
from typing import List, Tuple, Optional, Any

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int64)  # spacing格式转换
    resampler.SetReferenceImage(itkimage)  # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


class LITSDataset(Dataset):
    """
    Class for LiTS Dataset
    """

    def __init__(
            self,
            img_dirs: List[str],
            mask_dirs: List[str],
            config,
            transform: Optional[Any] = None,

    ):
        """
        Initialize the LiTS Dataset

        :param img_dirs:            list of image directories to pull images from
        :param detect_tumors:       boolean to tell whether to add a tumor class to the dataset
                                    If false, all tumors are treated as livers.
        :param transform:           list of transforms to conduct on volumes and segmentations
        """

        self.value_img_num = 0
        self.train_img_num = 0
        self.volume_img_paths = []  # 存储路径
        self.segmentation_img_paths = []
        self.volume = []
        self.segmentation = []
        self.json_paths = []
        self.img_ID = []
        img_dirs = ''.join(img_dirs)  # 读入的是列表，应该转化为str
        mask_dirs = ''.join(mask_dirs)
        self.img_ids = [splitext(file)[0] for file in listdir(img_dirs) if not file.startswith('.')]



        for i, idx in enumerate(self.img_ids):
            if i >= 10:
                break
            #todo:Old
            else:
                img_file = ""
                for s in os.listdir(img_dirs + idx):
                    if re.search('.nii.gz', s):
                        img_file = img_dirs + idx + '/' + s
                        self.img_ID.append(re.sub('[a-zA-Z-_.]', '', s))
                        break

            # todo:New
            # else:
            #     img_file = ""
            #     for s in os.listdir(img_dirs):
            #         if re.search('.nii.gz', s):
            #             img_file = img_dirs + '/' + s
            #             self.img_ID.append(re.sub('[a-zA-Z-_.]', '', s))
            #             break

                mask_file = ""
                mask_json = ""
                # todo:Old
                for s in os.listdir(mask_dirs + idx):
                    if re.search('.nii.gz', s):
                        mask_file = mask_dirs + idx + '/' + s
                    if re.search('ctd.json', s):
                        mask_json = mask_dirs + idx + '/' + s
                # todo:New
                # for s in os.listdir(mask_dirs):
                #     if re.search('.nii.gz', s):
                #         mask_file = mask_dirs + '/' + s
                #     if re.search('ctd.json', s):
                #         mask_json = mask_dirs + idx + '/' + s

                self.volume_img_paths.append(img_file)
                self.segmentation_img_paths.append(mask_file)
                self.json_paths.append(mask_json)

        self.transform = transform

        self.W = config["dataset"]["resize_dims"]["W"]
        self.H = config["dataset"]["resize_dims"]["H"]
        self.D = config["dataset"]["resize_dims"]["D"]

        pbar = tqdm(enumerate(self.volume_img_paths), unit='preprocessing',
                    total=len(self.volume_img_paths))

        for idx, volume_path in pbar:
            img_ID = self.img_ID[idx]
            sitk_img = sitk.ReadImage(self.volume_img_paths[idx])
            sitk_mask = sitk.ReadImage(self.segmentation_img_paths[idx])
            # todo:OLd
            with open(self.json_paths[idx], 'r') as f:
                mask_json = f.read()
                mask_json = json.loads(mask_json)
                ori = mask_json[0]['direction']  # [L, A, S], [W, H, D] -> [0, 1, 2]

            # ori = ['L','A','S']  # [L, A, S], [W, H, D] -> [0, 1, 2]

            # L, A, S as W, H, D
            # R, P, I as W, H, D

            ori_dict = {'L': 0, 'R': 0, 'A': 1, 'P': 1, 'S': 2, 'I': 2}
            output = [ori_dict[ori[0]], ori_dict[ori[1]], ori_dict[ori[2]]]
            shape = sitk_img.GetSize()

            resize_dict = [self.W, min(shape[output.index(0)] * self.W // shape[output.index(1)], self.H),
                           shape[output.index(2)] * self.W // shape[output.index(1)]]

            # todo 1429
            # resize_dict = [880, 880, 12]

            deep = resize_dict[2]

            sitk_img = resize_image_itk(sitk_img,
                                        (resize_dict[output[0]], resize_dict[output[1]], resize_dict[output[2]]),
                                        resamplemethod=sitk.sitkNearestNeighbor)
            sitk_mask = resize_image_itk(sitk_mask,
                                         (resize_dict[output[0]], resize_dict[output[1]], resize_dict[output[2]]),
                                         resamplemethod=sitk.sitkNearestNeighbor)

            sitk_img = resize_image_itk(sitk_img, (self.W, self.H, self.D), resamplemethod=sitk.sitkNearestNeighbor)
            sitk_mask = resize_image_itk(sitk_mask, (self.W, self.H, self.D), resamplemethod=sitk.sitkNearestNeighbor)


            volume = sitk.GetArrayFromImage(sitk_img)
            segmentation = sitk.GetArrayFromImage(sitk_mask)
            output = [output[2], output[1], output[0]]
            volume = volume.transpose((output.index(2), output.index(0), output.index(1)))
            segmentation = segmentation.transpose((output.index(2), output.index(0), output.index(1)))

            MIN_BOUND = 100
            MAX_BOUND = volume.max()
            volume = (volume - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
            volume[volume > 1] = 1.
            volume[volume < 0] = 0.

            self.volume.append(tuple([img_ID, volume]))
            self.segmentation.append(segmentation)

        self.img_num = len(self.volume)

    def show_featuremap(self, output, mask, idx):
        fig = plt.figure()
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

        num = len(os.listdir('./valuation_featuremap'))
        aaa = self.segmentation_img_paths[idx]
        plt.savefig('./valuation_featuremap/{}_{}'.format(num, aaa.split('/')[-1][:-2] + "png"), dpi=600)

    def __len__(self) -> int:  # 迭代器长度
        """
        Get the length of the dataset

        :return:    length of the dataset
        """
        return self.img_num

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a volume and segmentation pair given an index

        :param idx: index of the volume-segmentation pair
        :return:    tuple containing the volume-segmentation pair
        """
        id, volume = self.volume[idx]
        segmentation = self.segmentation[idx]
        if self.transform:
            volume = self.transform(volume)
            segmentation = self.transform(segmentation.astype(np.int64))

        segmentation = torch.where(segmentation > 0, 1, 0)

        # Phase 1 of training only detects liver regions
        # if not self.detect_tumors:
        #     segmentation = torch.clamp(segmentation, 0, 1)

        return id, volume, segmentation
