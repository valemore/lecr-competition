from typing import Union

from torch.nn import CrossEntropyLoss

from bienc.losses import BidirectionalMarginLoss, UnidirectionalMarginLoss

LossFunction = Union[BidirectionalMarginLoss, UnidirectionalMarginLoss, CrossEntropyLoss]
