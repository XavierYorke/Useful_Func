#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/03/04 00:48:46
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import torch
import torchkeras
import netron
# from model import Net


def export(save_pth, model_path):

    net = 'Net()'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)
    # state_dict = checkpoint['state_dict']
    net.load_state_dict(checkpoint)
    # net.load_state_dict(torch.load(args.resume, map_location=device))

    #Check the network structure
    # torchkeras.summary(net, input_shape=(3, 480, 496))
    net.eval()

    batch_size = 1
    c, h, w = 3, 512, 512

    # Input to the model
    x = torch.randn(batch_size, c, h, w, requires_grad=True)
    torch_out = net(x)

    # Export the model
    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_pth,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})


if __name__ == '__main__':
    save_path = 'exports/Net.onnx'
    model_path = ''

    # export(save_path, model_path)
    # netron.start(onnx_path)