import os
import re

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.reorint import reorient
from dataset.resample import resampleImg, resize_image_itk
from show_slice import show_slices


class PatchDataset(Dataset):
    def __init__(self, img_path, mask_path, config: yaml, IsTrain=True):
        self.img_dirs = []
        self.mask_dirs = []
        self.preprocessed = []
        self.img_ID = []
        self.mask_ID = []
        self.size = [int(x) for x in re.sub(r'[\(\)\[\]]', '', config['dataset']['size']).split(',')]
        self.patch_size = [int(x) for x in re.sub(r'[\(\)\[\]]', '', config['dataset']['patch_size']).split(',')]
        self.patch_stride = [int(x) for x in re.sub(r'[\(\)\[\]]', '', config['dataset']['patch_stride']).split(',')]
        self.spacing = [float(x) for x in re.sub(r'[\(\)\[\]]', '', config['dataset']['spacing']).split(',')]

        img_path = ''.join(img_path)
        mask_path = ''.join(mask_path)

        count = -1
        for x, i in enumerate(os.listdir(img_path)):
            if IsTrain and x >= 2000:
                break
            else:
                img_inside = os.listdir(img_path + i)
                for j in img_inside:
                    if re.search('.nii.gz', j):
                        self.img_dirs.append(img_path + i + '/' + j)
                        self.img_ID.append(re.sub('[a-zA-Z-_.]', '', j))
                        count += 1
                        mask_inside = os.listdir(mask_path + i)
                        for k in mask_inside:  # todo:读取一张图片后马上读取他对应的mask,如果把for循环写在外面的话会导致第二张图的img匹配第一张图的mask
                            if re.search('.nii.gz', k):
                                if re.sub('[a-zA-Z-_.]', '', k) == self.img_ID[count]:
                                    self.mask_dirs.append(mask_path + i + '/' + k)
                                    self.mask_ID.append(re.sub('[a-zA-Z-_.]', '', k))

        # todo:delete?
        self.img_ID = [int(x) for x in self.img_ID]
        self.mask_ID = [int(x) for x in self.mask_ID]
        pbar = tqdm(enumerate(self.img_dirs), unit='preprocessing', total=len(self.img_dirs))
        for idx, img_dirs in pbar:
            itk_img = sitk.ReadImage(img_dirs)
            itk_mask = sitk.ReadImage(self.mask_dirs[idx])
            nib_img = nib.load(img_dirs)
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo:不加会报错
            ori = nib.aff2axcodes(nib_img.affine)

            if IsTrain:
                itk_img = resampleImg(itk_img, self.spacing, resamplemethod=sitk.sitkLinear)
                itk_mask = resampleImg(itk_mask, self.spacing, resamplemethod=sitk.sitkNearestNeighbor)
            else:
                itk_img = resize_image_itk(itk_img, self.size, resamplemethod=sitk.sitkLinear)
                itk_mask = resize_image_itk(itk_mask, self.size, resamplemethod=sitk.sitkNearestNeighbor)

            itk_mask = sitk.BinaryThreshold(itk_mask, lowerThreshold=1, upperThreshold=3000, insideValue=1,
                                            outsideValue=0)
            itk_img = sitk.GetArrayFromImage(itk_img)
            itk_img = np.clip(itk_img, 0, 3000, out=None)
            itk_mask = sitk.GetArrayFromImage(itk_mask)
            itk_img, itk_mask = reorient(itk_img, itk_mask, ori)  # C,W,H
            ID = self.img_ID[idx]
            if IsTrain:
                for i in range(0, itk_img.shape[0] - self.patch_size[0], self.patch_stride[0]):
                    for j in range(0, itk_img.shape[1] - self.patch_size[1], self.patch_stride[1]):
                        for k in range(0, itk_img.shape[2] - self.patch_size[2], self.patch_stride[2]):
                            img_patch = itk_img[i:i + self.patch_size[0], j:j + self.patch_size[1],
                                        k:k + self.patch_size[2]]
                            mask_patch = itk_mask[i:i + self.patch_size[0], j:j + self.patch_size[1],
                                         k:k + self.patch_size[2]]
                            show_slices(ID, i, j, k, img_patch)
                            self.preprocessed.append(tuple([img_patch, mask_patch]))
            else:
                self.preprocessed.append(tuple([itk_img, itk_mask]))

        self.num_of_batch = len(self.preprocessed)

    def __len__(self) -> int:
        return self.num_of_batch

    def __getitem__(self, idx: int) -> tuple:
        img, mask, = self.preprocessed[idx]
        from torchvision import transforms
        to_tensor = transforms.ToTensor()  # todo:输入transform的数据的维度是3维的，别放扩充 后4维的数据进去
        img = to_tensor(img)
        mask = to_tensor(mask.astype(np.int64))
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return img, mask
