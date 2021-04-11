import torch
from torch import nn
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, back_coeff):
        ctx.back_coeff = back_coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        back_coeff = ctx.back_coeff
        reverse_with_coeff = -grad_output*back_coeff
        return reverse_with_coeff, None

class GradReverseLayer(nn.Module):
    def __init__(self, coeff_fn = lambda: 1):
        super().__init__()
        self.coeff_fn = coeff_fn
    
    def forward(self, x):
        x = GradReverse.apply(x, self.coeff_fn())
        return x

# def input_combine_split(fun):
#     # This function is what we "replace" hello with
#     def wrapper(*args, **kw):
#         args = list(args)
#         inp = args[1]
#         need_split = isinstance(inp, list)
#         inp = torch.cat(inp, dim=0) if need_split else inp
#         args[1] = inp
#         results = fun(*args, **kw)
#         if need_split:
#             if isinstance(results, list):
#                 f_result = []
#                 s_result = []
#                 for i in results:
#                     B = i.shape[0]/2    
#                     assert B.is_integer()
#                     B = int(B)
#                     f, s = torch.split(i, B/2, dim=0)
#                     f_result.append(f)
#                     s_result.append(s)
#             else:
#                 B = results.shape[0]/2    
#                 assert B.is_integer()
#                 B = int(B)
#                 f_result, s_result = torch.split(results, B, dim=0)
#             return f_result, s_result  
#         else:
#             return results
#     return wrapper