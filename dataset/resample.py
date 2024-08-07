import SimpleITK as sitk
import re
import numpy as np
def resampleImg(image, targetSpacing, resamplemethod=sitk.sitkLinear):
    """
    将体数据重采样的指定的spacing大小
    paras：
    image：sitk读取的image信息，这里是体数据
    targetSpacing：指定的spacing，例如[1,1,1]
    resamplemethod:插值类型  sitk.sitkLinear-线性img  sitk.sitkNearestNeighbor-最近邻mask
    return：重采样后的数据
    """
    target_size = [0, 0, 0]
    # 读取原始数据的size和spacing信息
    ori_size = image.GetSize()
    ori_spacing = image.GetSpacing()
    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    target_size[0] = round(ori_size[0] * ori_spacing[0] * 1.0 / targetSpacing[0])
    target_size[1] = round(ori_size[1] * ori_spacing[1] * 1.0 / targetSpacing[1])
    target_size[2] = round(ori_size[2] * ori_spacing[2] * 1.0 / targetSpacing[2])
    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(targetSpacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(resamplemethod)
    if resamplemethod == sitk.sitkNearestNeighbor:
        # mask用最近邻插值，保存为uint8
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    else:
        # 体数据用线性插值，保存为float32
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    newImage = resampler.Execute(image)
    return newImage

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
