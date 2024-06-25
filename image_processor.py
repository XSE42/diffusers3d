# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ====================================================================================================
#
# Modifications copyright 2024 XSE42
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import CONFIG_NAME, deprecate


PipelineImageInput = Union[
    np.ndarray,
    torch.Tensor,
    List[np.ndarray],
    List[torch.Tensor],
]


def is_valid_image(image):
    return isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (3, 4)


def is_valid_image_imagelist(images):
    # check if the image input is one of the supported formats for image and image list:
    # it can be either one of below 3
    # (1) a 5d pytorch tensor or numpy array,
    # (2) a valid image: 3d np.ndarray or torch.Tensor (grayscale image), 4d np.ndarray or torch.Tensor
    # (3) a list of valid image
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 5:
        return True
    elif is_valid_image(images):
        return True
    elif isinstance(images, list):
        return all(is_valid_image(image) for image in images)
    return False


class VaeImageProcessor3D(ConfigMixin):
    """
    3D Image processor for VAE.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (depth, height, width) dimensions to multiples of `vae_scale_factor`.
            Can accept `depth`, `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`].
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `False`):
            Whether to binarize the image to 0/1.
        do_convert_rgb (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to RGB format.
        do_convert_grayscale (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to grayscale format.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 4,
        resample: str = "lanczos",
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__()
        if do_convert_rgb:
            raise NotImplementedError("Converting images to RGB format is not supported yet.")
        if do_convert_rgb and do_convert_grayscale:
            raise ValueError(
                "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
            )

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        """
        Convert a 3D NumPy image to a PyTorch tensor.
        """
        if images.ndim == 4:  # (N, D, H, W)
            images = images[..., None]  # (N, D, H, W, C)

        images = torch.from_numpy(images.transpose(0, 4, 1, 2, 3))  # (N, C, D, H, W)
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 4, 1).float().numpy()  # (N, D, H, W, C)
        return images

    @staticmethod
    def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    def resize(
        self,
        image: Union[np.ndarray, torch.Tensor],
        depth: int,
        height: int,
        width: int,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Resize image.

        Args:
            image (`np.ndarray` or `torch.Tensor`):
                The image input, can be a numpy array or pytorch tensor.
            depth (`int`):
                The depth to resize to.
            height (`int`):
                The height to resize to.
            width (`int`):
                The width to resize to.

        Returns:
            `np.ndarray` or `torch.Tensor`:
                The resized image.
        """

        if isinstance(image, torch.Tensor):
            image = torch.nn.functional.interpolate(
                image,
                size=(depth, height, width),
            )
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pt(image)
            image = torch.nn.functional.interpolate(
                image,
                size=(depth, height, width),
            )
            image = self.pt_to_numpy(image)
        return image

    def binarize(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Create a mask.

        Args:
            image (`Union[np.ndarray, torch.Tensor]`):
                The image input, should be a PIL image.

        Returns:
            `Union[np.ndarray, torch.Tensor]`:
                The binarized image. Values less than 0.5 are set to 0, values greater than 0.5 are set to 1.
        """
        image[image < 0.5] = 0
        image[image >= 0.5] = 1

        return image

    def get_default_depth_height_width(
        self,
        image: Union[np.ndarray, torch.Tensor],
        depth: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        This function return the depth, height and width that are downscaled to the next integer multiple of
        `vae_scale_factor`.

        Args:
            image(`np.ndarray` or `torch.Tensor`):
                The image input, can be a numpy array or pytorch tensor. if it is a numpy array, should have
                shape `[batch, depth, height, width]` or `[batch, depth, height, width, channel]` if it is a pytorch
                tensor, should have shape `[batch, channel, depth, height, width]`.
            depth (`int`, *optional*, defaults to `None`):
                The depth in preprocessed image. If `None`, will use the depth of `image` input.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed image. If `None`, will use the height of `image` input.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed. If `None`, will use the width of the `image` input.
        """

        if depth is None:
            if isinstance(image, torch.Tensor):
                depth = image.shape[2]
            else:
                depth = image.shape[1]

        if height is None:
            if isinstance(image, torch.Tensor):
                height = image.shape[3]
            else:
                height = image.shape[2]

        if width is None:
            if isinstance(image, torch.Tensor):
                width = image.shape[4]
            else:
                width = image.shape[3]

        depth, width, height = (
            x - x % self.config.vae_scale_factor for x in (depth, width, height)
        )  # resize to integer multiple of vae_scale_factor

        return depth, height, width

    def preprocess(
        self,
        image: PipelineImageInput,
        depth: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Preprocess the image input.

        Args:
            image (`pipeline_image_input`):
                The image input, accepted formats are NumPy arrays, PyTorch tensors; Also accept list of
                supported formats.
            depth (`int`, *optional*, defaults to `None`):
                The depth in preprocessed image. If `None`, will use the `get_default_depth_height_width()` to get
                default depth.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed image. If `None`, will use the `get_default_depth_height_width()` to get
                default height.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed image. If `None`, will use get_default_depth_height_width()` to get
                default width.
        """
        supported_formats = (np.ndarray, torch.Tensor)

        # Expand the missing dimension for 4-dimensional pytorch tensor or numpy array that represents grayscale image
        if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 4:
            if isinstance(image, torch.Tensor):
                # if image is a pytorch tensor could have 2 possible shapes:
                #   1. batch x depth x height x width: we should insert the channel dimension at position 1
                #   2. channel x depth x height x width: we should insert batch dimension at position 0,
                #      however, since both channel and batch dimension has same size 1, it is same to insert at position 1
                #   for simplicity, we insert a dimension of size 1 at position 1 for both cases
                image = image.unsqueeze(1)
            else:
                # if it is a numpy array, it could have 2 possible shapes:
                #   1. batch x depth x height x width: insert channel dimension on last position
                #   2. depth x height x width x channel: insert batch dimension on first position
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 5:
            warnings.warn(
                "Passing `image` as a list of 5d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 5d np.ndarray",
                FutureWarning,
            )
            image = np.concatenate(image, axis=0)
        if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 5:
            warnings.warn(
                "Passing `image` as a list of 5d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 5d torch.Tensor",
                FutureWarning,
            )
            image = torch.cat(image, axis=0)

        if not is_valid_image_imagelist(image):
            raise ValueError(
                f"Input is in incorrect format. Currently, we only support {', '.join(str(x) for x in supported_formats)}"
            )
        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 5 else np.stack(image, axis=0)

            image = self.numpy_to_pt(image)

            depth, height, width = self.get_default_depth_height_width(image, depth, height, width)
            if self.config.do_resize:
                image = self.resize(image, depth, height, width)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 5 else torch.stack(image, axis=0)

            if self.config.do_convert_grayscale and image.ndim == 4:
                image = image.unsqueeze(1)

            channel = image.shape[1]
            # don't need any preprocess if the image is latents
            if channel == self.vae_latent_channels:
                return image

            depth, height, width = self.get_default_depth_height_width(image, depth, height, width)
            if self.config.do_resize:
                image = self.resize(image, depth, height, width)

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pt",
        do_denormalize: Optional[List[bool]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Postprocess the image output from tensor to `output_type`.

        Args:
            image (`torch.Tensor`):
                The image input, should be a pytorch tensor with shape `B x C x D x H x W`.
            output_type (`str`, *optional*, defaults to `pt`):
                The output type of the image, can be one of `np`, `pt`, `latent`.
            do_denormalize (`List[bool]`, *optional*, defaults to `None`):
                Whether to denormalize the image to [0,1]. If `None`, will use the value of `do_normalize` in the
                `VaeImageProcessor` config.

        Returns:
            `np.ndarray` or `torch.Tensor`:
                The postprocessed image.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )
        if output_type not in ["latent", "pt", "np"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `pt`. Please make sure to set it to one of these instead: "
                "`np`, `pt`, `latent`"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "pt"

        if output_type == "latent":
            return image

        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]

        image = torch.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )

        if output_type == "pt":
            return image

        image = self.pt_to_numpy(image)

        if output_type == "np":
            return image


class IPAdapterMaskProcessor3D(VaeImageProcessor3D):
    """
    3D Image processor for IP Adapter image masks.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (depth, height, width) dimensions to multiples of `vae_scale_factor`.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `True`):
            Whether to binarize the image to 0/1.
        do_convert_grayscale (`bool`, *optional*, defaults to be `True`):
            Whether to convert the images to grayscale format.

    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = False,
        do_binarize: bool = True,
        do_convert_grayscale: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )

    @staticmethod
    def downsample(mask: torch.Tensor, batch_size: int, num_queries: int, value_embed_dim: int):
        """
        Downsamples the provided mask tensor to match the expected dimensions for scaled dot-product attention. If the
        aspect ratio of the mask does not match the aspect ratio of the output image, a warning is issued.

        Args:
            mask (`torch.Tensor`):
                The input mask tensor generated with `IPAdapterMaskProcessor3D.preprocess()`.
            batch_size (`int`):
                The batch size.
            num_queries (`int`):
                The number of queries.
            value_embed_dim (`int`):
                The dimensionality of the value embeddings.

        Returns:
            `torch.Tensor`:
                The downsampled mask tensor.

        """
        o_d = mask.shape[1]
        o_h = mask.shape[2]
        o_w = mask.shape[3]
        ratio_d = o_d / o_h
        ratio_w = o_w / o_h
        mask_h = math.cbrt(num_queries / ratio_d / ratio_w)  # cbrt((n * h^2) / (d * w))
        mask_d = mask_h * ratio_d  # cbrt((n * d^2) / (h * w))
        mask_w = mask_h * ratio_w  # cbrt((n * w^2) / (d * h))
        mask_h = round(mask_h)
        mask_d = round(mask_d)
        mask_w = round(mask_w)

        mask_downsample = F.interpolate(mask.unsqueeze(0), size=(mask_d, mask_h, mask_w), mode="trilinear").squeeze(0)

        # Repeat batch_size times
        if mask_downsample.shape[0] < batch_size:
            mask_downsample = mask_downsample.repeat(batch_size, 1, 1, 1)

        mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1)

        downsampled_area = mask_d * mask_h * mask_w
        # If the output image and the mask do not have the same aspect ratio, tensor shapes will not match
        # Pad tensor if downsampled_mask.shape[1] is smaller than num_queries
        if downsampled_area < num_queries:
            warnings.warn(
                "The aspect ratio of the mask does not match the aspect ratio of the output image. "
                "Please update your masks or adjust the output size for optimal performance.",
                UserWarning,
            )
            mask_downsample = F.pad(mask_downsample, (0, num_queries - mask_downsample.shape[1]), value=0.0)
        # Discard last embeddings if downsampled_mask.shape[1] is bigger than num_queries
        if downsampled_area > num_queries:
            warnings.warn(
                "The aspect ratio of the mask does not match the aspect ratio of the output image. "
                "Please update your masks or adjust the output size for optimal performance.",
                UserWarning,
            )
            mask_downsample = mask_downsample[:, :num_queries]

        # Repeat last dimension to match SDPA output shape
        mask_downsample = mask_downsample.view(mask_downsample.shape[0], mask_downsample.shape[1], 1).repeat(
            1, 1, value_embed_dim
        )

        return mask_downsample

