# -*- coding: utf-8 -*-
# @Time   : 2020/8/14 15:26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : base_model.py

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        prefix = '-------- model info --------\n'
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return prefix + super().__str__() + '\nTrainable parameters: {}'.format(params)
