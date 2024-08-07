def reorient(itk_img, itk_mask, ori) -> tuple:
    ori_dict = {'L': 0, 'R': 0, 'A': 1, 'P': 1, 'S': 2, 'I': 2}
    output = [ori_dict[ori[2]], ori_dict[ori[1]], ori_dict[ori[0]]]
    itk_img = itk_img.transpose((output.index(2), output.index(1), output.index(0)))
    itk_mask = itk_mask.transpose((output.index(2), output.index(1), output.index(0)))
    return itk_img, itk_mask


def figure_patch(ID, img_size, patch_size, patch_stride) -> tuple:
    Step_C = (img_size[0] - patch_size[0]) / patch_stride[0] + 1
    Step_H = (img_size[1] - patch_size[1]) / patch_stride[1] + 1
    Step_W = (img_size[2] - patch_size[2]) / patch_stride[2] + 1
    NewSize_C = ((img_size[0] - patch_size[0]) // patch_stride[0]) * patch_stride[0] + patch_size[0]
    NewSize_H = ((img_size[1] - patch_size[1]) // patch_stride[1]) * patch_stride[1] + patch_size[1]
    NewSize_W = ((img_size[2] - patch_size[2]) // patch_stride[2]) * patch_stride[2] + patch_size[2]

    return (Step_C, Step_H, Step_W), (NewSize_C, NewSize_H, NewSize_W)



from torch import nn
import torch
class GetPatch(nn.Module):
    def __init__(self, patch_size, patch_stride):    # 初始化patch的两个参数
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.preprocessed = []

    def forward(self, itk_img, itk_mask):               # 切的时候调用cut函数
        # TODO for i in range(0, itk_img.shape[0] - self.patch_size[0] + 1, self.patch_stride[0]):
        for i in range(0, itk_img.shape[0] - self.patch_size[0] + self.patch_stride[0], self.patch_stride[0]):
            for j in range(0, itk_img.shape[1] - self.patch_size[1] + self.patch_stride[1], self.patch_stride[1]):
                for k in range(0, itk_img.shape[2] - self.patch_size[2] + self.patch_stride[2],
                               self.patch_stride[2]):

                    img_patch = torch.tensor(itk_img[i:i + self.patch_size[0], j:j + self.patch_stride[1],
                                k:k + self.patch_size[2]], dtype=torch.float32)

                    mask_patch = torch.tensor(itk_mask[i:i + self.patch_size[0], j:j + self.patch_size[1],
                                 k:k + self.patch_size[2]], dtype=torch.float32)

                    # img_patch = itk_img[i:i + self.patch_size[0], j:j + self.patch_stride[1],
                    #             k:k + self.patch_size[2]].astype(np.float32)
                    # todo:mask分为裁剪和不裁剪，分别看拼接前和拼接后的效果

                    direction = (i, j, k)
                    patch_setting = (1, 1, 1)

                    self.preprocessed.append(
                        tuple[patch_setting, direction, img_patch, mask_patch])
        return self.preprocessed


