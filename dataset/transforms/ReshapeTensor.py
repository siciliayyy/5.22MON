import torch


class ReshapeTensor:
    """
    Reshapes a 3d image tensor to the correct dimensions
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Call logic for ReshapeTensor

        :param img: image tensor with dimensions (D x H x W)
        :return:    image tensor with dimensions (B x D x H x W)
        """

        return img.transpose(0, 1).transpose(1, 2).unsqueeze(0)
