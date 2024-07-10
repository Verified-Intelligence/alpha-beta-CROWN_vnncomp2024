#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
This file shows how to use a customized graph optimizers, and it defines a
few customized optimizers for VNN-COMP models as examples. See example
configuration files in the `exp_configs` folder, e.g.,
`exp_configs/vnncomp23/gtrsb.yaml`.
"""

import torch
from auto_LiRPA import BoundedModule
from auto_LiRPA.bound_ops import (BoundSign, BoundAdd, BoundSignMerge,
                                  BoundReduceSum, BoundRelu, BoundConstant,
                                  BoundBuffers, BoundMul, BoundSub,
                                  BoundUnsqueeze, BoundMultiPiecewiseNonlinear)


def default_optimizer(model: BoundedModule):
    pass


def merge_sign(model: BoundedModule):
    nodes = list(model.nodes())
    for i, node in enumerate(nodes):
        if (i+2 < len(nodes) and type(node) == BoundSign
            and type(nodes[i+1]) == BoundAdd and type(nodes[i+2]) == BoundSign):
            print('Merging Sign node: %s', node)
            node_merge = BoundSignMerge(inputs=[node.inputs[0]],
                                        options=model.bound_opts)
            node_merge.name = f'{node.name}/merge'
            model.add_nodes([node_merge])
            model.replace_node(node, node_merge)
            model.replace_node(nodes[i+1], node_merge)
            model.replace_node(nodes[i+2], node_merge)


def merge_relu_lookup_table(model: BoundedModule):
    nodes = list(model.nodes())
    for node in nodes:
        if (isinstance(node, BoundReduceSum)
                and isinstance(node.inputs[0], BoundMul)
                and isinstance(node.inputs[1], BoundConstant)
                and node.inputs[1].value.item() == -1):
            node_mul = node.inputs[0]
            if (isinstance(node_mul.inputs[1], BoundBuffers)
                    and node_mul.inputs[1].buffer.ndim == 1
                    and isinstance(node_mul.inputs[0], BoundRelu)
                    and isinstance(node_mul.inputs[0].inputs[0], BoundSub)
            ):
                node_sub = node_mul.inputs[0].inputs[0]
                node_weight = node_mul.inputs[1]
                if (isinstance(node_sub.inputs[1], BoundBuffers)
                        and node_sub.inputs[1].buffer.ndim == 1
                        and isinstance(node_sub.inputs[0], BoundUnsqueeze)
                        and not node_sub.inputs[0].inputs[1].perturbed
                        and node_sub.inputs[0].inputs[1].value == -1
                        and len(node_sub.inputs[0].inputs[0].output_shape) == 2
                ):
                    node_offset = node_sub.inputs[1]
                    node_input = node_sub.inputs[0].inputs[0]
                    # (weight*ReLU(node_input.unsqueeze(-1)-offset)).sum(-1)
                    print('Found a ReLU-based lookup table')
                    print('Input node:', node_input)
                    print('Offset:', node_offset.buffer)
                    print('Weight:', node_weight.buffer)
                    node_merged = BoundMultiPiecewiseNonlinear(
                        inputs=[node_input, node_weight, node_offset])
                    node_merged.name = f'{node.name}/merged'
                    model.add_nodes([node_merged])
                    model.replace_node(node, node_merged)
                    print('New node created:', node_merged)
