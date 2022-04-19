import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


# Precomputed aliases
MODELS = {
    'efficientnet-b0': [
        ('reduction_1', (0, 2)),
        ('reduction_2', (2, 4)),
        ('reduction_3', (4, 6)),
        ('reduction_4', (6, 12))
    ],
    'efficientnet-b4': [
        ('reduction_1', (0, 3)),
        ('reduction_2', (3, 7)),
        ('reduction_3', (7, 11)),
        ('reduction_4', (11, 23)),
    ]
}


class EfficientNetExtractor(torch.nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = EfficientNetExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, layer_names, image_height, image_width, model_name='efficientnet-b4'):
        super().__init__()

        assert model_name in MODELS
        assert all(k in [k for k, v in MODELS[model_name]] for k in layer_names)

        idx_max = -1
        layer_to_idx = {}

        # Find which blocks to return
        for i, (layer_name, _) in enumerate(MODELS[model_name]):
            if layer_name in layer_names:
                idx_max = max(idx_max, i)
                layer_to_idx[layer_name] = i

        # We can set memory efficient swish to false since we're using checkpointing
        net = EfficientNet.from_pretrained(model_name)
        net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        blocks = [nn.Sequential(net._conv_stem, net._bn0, net._swish)]

        # Only run needed blocks
        for idx in range(idx_max):
            l, r = MODELS[model_name][idx][1]

            block = SequentialWithArgs(*[(net._blocks[i], [i * drop]) for i in range(l, r)])
            blocks.append(block)

        self.layers = nn.Sequential(*blocks)
        self.layer_names = layer_names
        self.idx_pick = [layer_to_idx[l] for l in layer_names]

        # Pass a dummy tensor to precompute intermediate shapes
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True)

        result = []

        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)

            result.append(x)

        return [result[i] for i in self.idx_pick]


class SequentialWithArgs(nn.Sequential):
    def __init__(self, *layers_args):
        layers = [layer for layer, args in layers_args]
        args = [args for layer, args in layers_args]

        super().__init__(*layers)

        self.args = args

    def forward(self, x):
        for l, a in zip(self, self.args):
            x = l(x, *a)

        return x


if __name__ == '__main__':
    """
    Helper to generate aliases for efficientnet backbones
    """
    device = torch.device('cuda')
    dummy = torch.rand(6, 3, 224, 480).to(device)

    for model_name in ['efficientnet-b0', 'efficientnet-b4']:
        net = EfficientNet.from_pretrained(model_name)
        net = net.to(device)
        net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        conv = nn.Sequential(net._conv_stem, net._bn0, net._swish)

        record = list()

        x = conv(dummy)
        px = x
        pi = 0

        # Terminal early to save computation
        for i, block in enumerate(net._blocks):
            x = block(x, i * drop)

            if px.shape[-2:] != x.shape[-2:]:
                record.append((f'reduction_{len(record)+1}', (pi, i+1)))

                pi = i + 1
                px = x

        print(model_name, ':', {k: v for k, v in record})
