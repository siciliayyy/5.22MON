import re

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

nii_path = './OUTPUT_nii/'
featuremap_path = './OUTPUT_featuremap/'

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

def show_slices(img_ID, raw: np.ndarray, img: np.ndarray, mask, DC, Show_nii: bool):
    img_ID = re.sub('[\"\'\(\),]', '', img_ID)
    img_size = img.shape
    mask_size = img.shape
    if Show_nii:
        raw_ = sitk.GetImageFromArray(raw)
        sitk.WriteImage(raw_, nii_path + '{}_raw.nii.gz'.format(str(img_ID)))
        img_ = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_, nii_path + '{}.nii.gz'.format(str(img_ID)))
        mask_ = sitk.GetImageFromArray(mask)
        sitk.WriteImage(mask_, nii_path + '{}_mask.nii.gz'.format(str(img_ID)))
        mask_position_of_raw_image = mask>0
        mask_position_of_raw_image = raw*mask_position_of_raw_image
        mask_position_of_raw_image = mask_position_of_raw_image*6000
        mask_position_of_raw_image = sitk.GetImageFromArray(mask_position_of_raw_image)
        mask_position_of_raw_image = resize_image_itk(mask_position_of_raw_image, (512,512,512), resamplemethod=sitk.sitkNearestNeighbor)
        sitk.WriteImage(mask_position_of_raw_image, nii_path + '{}_mask_position_of_raw_image.nii.gz'.format(str(img_ID)))

    slice_0 = np.flip(img[img_size[0] // 2, :, :])
    slice_3 = np.flip(mask[mask_size[0] // 2, :, :])

    slice_1 = np.flip(img[:, img_size[1] // 2, :])
    slice_4 = np.flip(mask[:, mask_size[1] // 2, :])

    slice_2 = np.flip(img[:, :, img_size[2] // 2])
    slice_5 = np.flip(mask[:, :, mask_size[2] // 2])
    fig = plt.figure()
    ax_0 = fig.add_subplot(2, 3, 1)
    ax_0.set_title('')
    plt.imshow(slice_0, cmap="gray", origin="lower")
    ax_1 = fig.add_subplot(2, 3, 2)
    ax_1.set_title('')
    plt.imshow(slice_1, cmap="gray", origin="lower")
    ax_2 = fig.add_subplot(2, 3, 3)
    ax_2.set_title('')
    plt.imshow(slice_2, cmap="gray", origin="lower")
    ax_3 = fig.add_subplot(2, 3, 4)
    ax_3.set_title('')
    plt.imshow(slice_3, cmap="gray", origin="lower")
    ax_4 = fig.add_subplot(2, 3, 5)
    ax_4.set_title('')
    plt.imshow(slice_4, cmap="gray", origin="lower")
    ax_5 = fig.add_subplot(2, 3, 6)
    ax_5.set_title('')
    plt.imshow(slice_5, cmap="gray", origin="lower")

    plt.suptitle("ID={}__DC={}".format(str(img_ID), str(DC)))
    plt.savefig(featuremap_path + 'ID{}.png'.format(str(img_ID)), dpi=600)
