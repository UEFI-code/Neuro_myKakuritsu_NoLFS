import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import myKakuritsu_Linear

class myKakuritsu_Linear_Function(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, Kakuritsu):
        ctx.save_for_backward(input, weight)
        #output = input.mm(weight.t())
        #output = mylinear_cpp.forward(input, weight)
        output = myKakuritsu_Linear.forward(input, weight, Kakuritsu)

        return output[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        #print(grad_output)
        #grad_input = grad_weight = None
        #grad_input = grad_output.mm(weight)
        #grad_weight = grad_output.t().mm(input)
        #grad_input, grad_weight = mylinear_cpp.backward(grad_output, input, weight)
        grad_input, grad_weight = myKakuritsu_Linear.backward(grad_output, input, weight)

        #print(grad_input)

        return grad_input, grad_weight, None

class myKakuritsu_Linear_Obj(nn.Module):
    def __init__(self, input_features, output_features, p=0.5):
        super(myKakuritsu_Linear_Obj, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
        self.p = p

    def forward(self, input):
        return myKakuritsu_Linear_Function.apply(input, self.weight, self.p)
