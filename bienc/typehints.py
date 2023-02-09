from typing import Union

from bienc.losses import BidirectionalMarginLoss, UnidirectionalMarginLoss

LossFunction = Union[BidirectionalMarginLoss, UnidirectionalMarginLoss]
