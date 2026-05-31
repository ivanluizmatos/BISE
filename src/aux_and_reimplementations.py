import torch
from torch.nn import init
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import numbers
from typing import Union, List, Tuple
import copy


# =========================== SoftMax ==========================

# Keep original softmax
if not hasattr(F, "_original_softmax"):
    F._original_softmax = F.softmax


# NOTE: Need to do "Monkey-patch" in the code:
# nn.functional.softmax = modified_softmax


def modified_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Mask-aware softmax that treats zero entries as pruned features,
    preserving gradients with a straight-through estimator (STE).
    
    Args:
        input: Tensor of any shape
        dim: Dimension over which to apply softmax
    Returns:
        Tensor of same shape as input
    """

    # infer mask from nonzero entries
    mask = (input != 0).float()

    # forward: set masked positions (-inf) to exclude them from softmax
    masked_input = input.masked_fill(mask == 0, float('-inf'))
    soft = F._original_softmax(masked_input, dim=dim)

    # STE: forward is zero at masked positions, backward flows to input
    out = soft * mask + input * (1 - mask).detach()

    return out


# def softmax(weights, scores, dim):
#     num = weights * torch.exp(scores)
#     den = torch.sum(weights * torch.exp(scores), dim = dim)
#     return num/(den.unsqueeze(-1) + 1e-8)


# =========================== LayerNorm ==========================


_shape_t = Union[int, List[int], Tuple[int, ...]]

'''
class ModifiedLayerNorm(nn.Module):

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool
    
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # mask of non-zero entries
        mask = (input != 0).float()
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

        # masked mean
        mean = (input * mask).sum(dim=-1, keepdim=True) / count     # maybe the 'input * mask' could be simply replaced by 'input'

        # masked variance
        var = ((input - mean) * mask).pow(2).sum(dim=-1, keepdim=True) / count

        # normalize
        x_norm = (input - mean) / torch.sqrt(var + self.eps)

        # keep zeroed positions at zero (but allow gradient flow through STE)
        out = x_norm * mask + input * (1 - mask).detach()

        if self.weight is not None:
            # out = out * self.weight
            out = (out * self.weight) * mask + input * (1 - mask).detach()
        if self.bias is not None:
            # out = out + self.bias
            out = (out + self.bias) * mask + input * (1 - mask).detach()

        return out

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
'''

class ModifiedLayerNorm(nn.Module):

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    # normalized_shape: tuple[int, ...]
    # eps: float
    # elementwise_affine: bool
    
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()


        self.mask = torch.ones_like(self.bias)


    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # mask of non-zero entries
        # mask = (input != 0).float()
        mask = self.mask 

        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

        # masked mean
        mean = (input * mask).sum(dim=-1, keepdim=True) / count

        # masked variance
        var = ((input - mean) * mask).pow(2).sum(dim=-1, keepdim=True) / count

        # normalize
        x_norm = (input - mean) / torch.sqrt(var + self.eps)

        # keep zeroed positions at zero (but allow gradient flow through STE)
        out = x_norm * mask + input * (1 - mask).detach()

        if self.weight is not None:
            # out = out * self.weight
            out = (out * self.weight) * mask + input * (1 - mask).detach()
        if self.bias is not None:
            # out = out + self.bias
            out = (out + self.bias) * mask + input * (1 - mask).detach()

        return out

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )



# Function to replace LayerNorm

def replace_layernorm_with_modified(module: nn.Module):
    '''
    Recursively replace LayerNorm layers by our modified, "mask-aware", implementation.
    '''
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            # create new LN with same config
            new_ln = ModifiedLayerNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                bias=(child.bias is not None),
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            # copy parameters if affine
            if child.elementwise_affine:
                with torch.no_grad():
                    new_ln.weight.copy_(child.weight)
                    if child.bias is not None:
                        new_ln.bias.copy_(child.bias)

            # replace in parent
            setattr(module, name, new_ln)
        else:
            replace_layernorm_with_modified(child)



# Function to update LayerNorm masks
def update_layernorm_masks(model):
    mask_index = 1
    for i in range(12):
        model.bert.encoder.layer[i].output.LayerNorm.mask = copy.deepcopy((model.mask_layers[mask_index].m_i >= 0).float())
        mask_index += 2

# ================================================================