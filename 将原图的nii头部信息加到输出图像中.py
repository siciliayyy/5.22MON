import SimpleITK as sitk
import nibabel as nib
import numpy as np

raw = 'E:/savefromxftp/5.22MON/data/dataset-verse20training/rawdata/sub-gl003/sub-gl003_dir-ax_ct.nii.gz'
mask_img = 'E:/savefromxftp/5.22MON/data/dataset-verse20training/derivatives/sub-gl003/sub-gl003_dir-ax_seg-vert_msk.nii.gz'
save_place = 'E:/savefromxftp/ZZZ_FROM_TUTOR'
raw_img = sitk.ReadImage(raw)
mask_img = sitk.ReadImage(mask_img)
raw_img = sitk.GetArrayFromImage(raw_img)
mask_img = sitk.GetArrayFromImage(mask_img)
mask_img = mask_img.clip(0,1)
raw_img = raw_img * mask_img
raw_img = raw_img*2

# raw_img = sitk.GetImageFromArray(raw_img)
# sitk.WriteImage(raw_img, save_place+'/img.nii.gz')
# raw_img = nib.load('E:/savefromxftp/ZZZ_FROM_TUTOR/img.nii.gz')
aaa = nib.load(raw)
a = aaa.header
b = aaa.affine
raw_img = raw_img.transpose((2,1,0))
clipped_img = nib.Nifti1Image(raw_img, aaa.affine, aaa.header)      # 利用nib将原图的头部信息保存在切割后的图像中
nib.save(clipped_img, 'clipped_image.nii')
