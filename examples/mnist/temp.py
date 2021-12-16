import torch
import numpy as np
import matplotlib.pyplot as plt



   
from typing import Optional, Union, Tuple, Sequence, Optional, Union

from matplotlib.image import AxesImage
from torch.nn.modules.utils import _pair
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.parameter import Parameter

from bindsnet.network.topology import AbstractConnection
from bindsnet.network.nodes import Nodes
plt.ion()

import math
import torch
import numpy as np

from torch.nn.modules.utils import _pair



class LocalConnection2D(AbstractConnection):
    """
    2D Local connection between one or two population of neurons supporting multi-channel 3D inputs;
    the logic is different from the BindsNet implementaion, but some lines are unchanged
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        input_shape: Tuple,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        """
        Instantiates a 'LocalConnection` object. Source population can be multi-channel
        Neurons in the post-synaptic population are ordered by receptive field; that is,
        if there are `n_conv` neurons in each post-synaptic patch, then the first
        `n_conv` neurons in the post-synaptic population correspond to the first
        receptive field, the second ``n_conv`` to the second receptive field, and so on.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        :param input_shape: The 2D shape of each input channel
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        Keyword arguments:
        :param LearningRule update_rule: Modifies connection parameters according to
            some rule. For now, only PostPre has been implemented for the multi-channel-input implementation
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """

        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = kwargs.get('padding', 0)

        shape = input_shape

        if kernel_size == shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                (shape[0] - kernel_size[0]) // stride[0] + 1,
                (shape[1] - kernel_size[1]) // stride[1] + 1,
            )

        self.conv_size = conv_size
        self.conv_prod = int(np.prod(conv_size))
        self.kernel_prod = int(np.prod(kernel_size))
         
        assert (
            target.n == out_channels * self.conv_prod
        ), "Target layer size must be n_filters * (kernel_size ** 2)."

        w = kwargs.get("w", None)

        if w is None:
            w = torch.rand(
                self.in_channels, 
                self.out_channels * self.conv_prod,
                self.kernel_prod
            )
        else:
            assert w.shape == (
                self.in_channels, 
                self.out_channels * self.conv_prod,
                self.kernel_prod
                )
        if self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get("b", None), requires_grad=False)


    def compute(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using layer weights.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        # Compute multiplication of pre-activations by connection weights
        # s: batch, ch_in, w_in, h_in => s_unfold: batch, ch_in, ch_out * w_out * h_out, k ** 2
        # w: ch_in, ch_out * w_out * h_out, k ** 2
        # a_post: batch, ch_in, ch_out * w_out * h_out, k ** 2 => batch, ch_out * w_out * h_out (= target.n)

        self.s_unfold = s.unfold(
            -2,self.kernel_size[0],self.stride[0]
        ).unfold(
            -2,self.kernel_size[1],self.stride[1]
        ).reshape(
            s.shape[0], 
            self.in_channels,
            self.conv_prod,
            self.kernel_prod,
        ).repeat(
            1,
            1,
            self.out_channels,
            1,
        )

        a_post = self.s_unfold.to(self.w.device) * self.w.unsqueeze(0)
        
        return a_post.sum(-1).sum(1).view(
            a_post.shape[0], self.out_channels, *self.conv_size,
            )

    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            # get a view and modify in-place
            # w: ch_in, ch_out * w_out * h_out, k ** 2
            w = self.w.view(
                self.w.shape[0]*self.w.shape[1], self.w.shape[2]
            )

            for fltr in range(w.shape[0]):
                w[fltr,:] *= self.norm / w[fltr,:].sum(0)


    def reset_state_variables(self) -> None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.target.reset_state_variables()





def reshape_locally_connected_weights(
    w: torch.Tensor,
    n_filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    input_sqrt: Union[int, Tuple[int, int]],
) -> torch.Tensor:

    """
    Get the weights from a locally connected layer slice (channel) and reshape them to be two-dimensional and square.
    :param w: Weights from a locally connected layer.
    :param n_filters: No. of neuron filters.
    :param kernel_size: Side length(s) of convolutional kernel.
    :param conv_size: Side length(s) of convolution population.
    :param input_sqrt: Sides length(s) of input neurons.
    :return: Locally connected weights reshaped as a collection of spatially ordered square grids.
    """

    kernel_size = _pair(kernel_size)
    conv_size = _pair(conv_size)
    input_sqrt = _pair(input_sqrt)

    k1, k2 = kernel_size
    c1, c2 = conv_size
    i1, i2 = input_sqrt
    fs = int(math.ceil(math.sqrt(n_filters)))

    w_ = torch.zeros((n_filters * k1, k2 * c1 * c2))

    for n1 in range(c1):
        for n2 in range(c2):
            for feature in range(n_filters):
                n = n1 * c2 + n2
                filter_ = w[feature, n1, n2, :, :
                ].view(k1, k2)
                w_[feature * k1 : (feature + 1) * k1, n * k2 : (n + 1) * k2] = filter_

    if c1 == 1 and c2 == 1:
        square = torch.zeros((i1 * fs, i2 * fs))

        for n in range(n_filters):
            square[
                (n // fs) * i1 : ((n // fs) + 1) * i2,
                (n % fs) * i2 : ((n % fs) + 1) * i2,
            ] = w_[n * i1 : (n + 1) * i2]

        return square
    else:
        square = torch.zeros((k1 * fs * c1, k2 * fs * c2))

        for n1 in range(c1):
            for n2 in range(c2):
                for f1 in range(fs):
                    for f2 in range(fs):
                        if f1 * fs + f2 < n_filters:
                            square[
                                k1 * (n1 * fs + f1) : k1 * (n1 * fs + f1 + 1),
                                k2 * (n2 * fs + f2) : k2 * (n2 * fs + f2 + 1),
                            ] = w_[
                                (f1 * fs + f2) * k1 : (f1 * fs + f2 + 1) * k1,
                                (n1 * c2 + n2) * k2 : (n1 * c2 + n2 + 1) * k2,
                            ]

        return square

def plot_locally_connected_feature_maps(weights: torch.Tensor,
    n_filters: int,
    in_chans: int,
    slice_to_plot: int,
    input_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    wmin: float = 0.0,
    wmax: float = 1.0,
    im: Optional[AxesImage] = None,
    lines: bool = True,
    figsize: Tuple[int, int] = (5, 5),
    cmap: str = "hot_r",
    color: str='r'
    ) -> AxesImage:

    """
    Plot a 2D local connection slice feature map
    :param weights: Weight matrix of Conv2dConnection object.
    :param n_filters: No. of convolution kernels in use.
    :param in_channels: No. of input channels
    :param slice_to_plot: The 2D slice to plot
    :param input_size: The square input size
    :param kernel_size: Side length(s) of 2D convolution kernels.
    :param conv_size: Side length(s) of 2D convolution population.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param lines: Whether or not to draw horizontal and vertical lines separating input
        regions.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: Used for re-drawing the weights plot.
    """

    n_sqrt = int(np.ceil(np.sqrt(n_filters)))

    sel_slice = weights.view(in_chans, n_filters, conv_size, conv_size, kernel_size, kernel_size).cpu()
    sel_slice = sel_slice[slice_to_plot, ...]
    
    reshaped = reshape_locally_connected_weights(
        sel_slice, n_filters, kernel_size, conv_size, input_size
    )

    if not im:
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(reshaped.cpu(), cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        kernel_size = _pair(kernel_size)
        conv_size = _pair(conv_size)
        
        if lines:
            for i in range(
                n_sqrt * kernel_size[0],
                n_sqrt * conv_size[0] * kernel_size[0],
                n_sqrt * kernel_size[0],
            ):
                ax.axhline(i - 0.5, color=color, linestyle="--")

            for i in range(
                n_sqrt * kernel_size[1],
                n_sqrt * conv_size[1] * kernel_size[1],
                n_sqrt * kernel_size[1],
            ):
                ax.axvline(i - 0.5, color=color, linestyle="--")

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect("auto")

        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(reshaped)

    return im