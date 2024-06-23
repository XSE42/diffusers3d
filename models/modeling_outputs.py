from dataclasses import dataclass

from diffusers.utils import BaseOutput


@dataclass
class Autoencoder3DOutput(BaseOutput):
    """
    Output of Autoencoder3D encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution3D`):
            Encoded outputs of `Encoder3D` represented as the mean and logvar of `DiagonalGaussianDistribution3D`.
            `DiagonalGaussianDistribution3D` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution3D"


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer3DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, depth, height, width)`):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.Tensor
