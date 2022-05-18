from torch.autograd import Variable
from operation import *
from utils import *
from metalayer import *

NEEDS_PARAM_OP = ['FactorizedReduce', 'SepConv', 'DilConv']


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, device, args):
        super(Cell, self).__init__()

        self.device = device
        self.args = args

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, device, args)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, device, args)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, device, args)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction, device, args)

    def _compile(self, C, op_names, indices, concat, reduction, device, args):
        assert len(op_names) == len(indices)

        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, device, args)
            self._ops += [op]  # ModuleList append
        self._indices = indices

    def forward(self, s0, s1, drop_prob, step, params=None, training=True, backup_running_statistics=False):

        if params is not None:
            param_dict = extract_top_level_dict(params)
            param_pre0 = param_dict['preprocess0']
            param_pre1 = param_dict['preprocess1']
            param_ops = extract_top_level_dict(param_dict['_ops'])

        s0 = self.preprocess0(s0, step, param_pre0, training=training, backup_running_statistics=backup_running_statistics)
        s1 = self.preprocess1(s1, step, param_pre1, training=training, backup_running_statistics=backup_running_statistics)


        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            if op1._get_name() in NEEDS_PARAM_OP:
                h1 = op1(h1, step, param_ops[str(2*i)], training=training, backup_running_statistics=backup_running_statistics)
            else:
                h1 = op1(h1)
            if op2._get_name() in NEEDS_PARAM_OP:
                h2 = op2(h2, step, param_ops[str(2*i+1)], training=training, backup_running_statistics=backup_running_statistics)
            else:
                h2 = op2(h2)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkMiniImageNet(nn.Module):

    def __init__(self, args, C, num_classes, layers,  auxiliary, genotype, device, steps=4, stem_multiplier=3):
        super(NetworkMiniImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        C_curr = stem_multiplier * C

        # self.stem0 = nn.Sequential(
        #     nn.Conv2d(3, C_curr // 2, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(C_curr // 2),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(C_curr // 2, C_curr, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(C_curr),
        #     nn.MaxPool2d(2, 2),
        # )
        #
        self.stem0 = stem0(C_curr, kernel_size=3, stride=1, padding=1, bias=True, device=device, args=args)



        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = nn.ModuleList()
        reduction_prev = False
        '''
        Construct network according to layers. If layers == 1, only reduction cell is used.
        '''
        if layers == 1:
            C_curr *= 2
            reduction = True
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        elif layers == 2:
            for i in range(layers):
                if i == 1:
                    C_curr *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, device, args)
                reduction_prev = reduction
                self.cells += [cell]
                C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        else:
            for i in range(layers):
                if i in [layers // 3, 2 * layers // 3]:
                    C_curr *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                reduction_prev = reduction
                self.cells += [cell]
                C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
                if i == 2 * layers // 3:
                    C_to_auxiliary = C_prev

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier_meta_nas = MetaLinearLayer(C_prev, num_classes, use_bias=True)

    def forward(self, input,  step,  params=None, training=True, backup_running_statistics=True):
        logits_aux = None
        if params is not None:
            params = extract_top_level_dict(params)
            params_stem0 = params['stem0']
            params_cell = extract_top_level_dict(params['cells'])
            params_classifier = params['classifier_meta_nas']


        s0 = s1 = self.stem0(input, params_stem0, step=step, training=training,
                             backup_running_statistics=backup_running_statistics)
        for i, cell in enumerate(self.cells):
            #s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            s0, s1 = s1, cell(s0, s1, 0, step, params=params_cell[str(i)], training=training,
                              backup_running_statistics=backup_running_statistics)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier_meta_nas(out.view(out.size(0), -1), params=params_classifier)
        return logits

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if (
                        param.requires_grad == True
                        and param.grad is not None
                        and torch.sum(param.grad) > 0
                ):
                    print(param.grad)
                    param.grad.zero_()
        else:
            for name, param in params.items():
                if (
                        param.requires_grad == True
                        and param.grad is not None
                        and torch.sum(param.grad) > 0
                ):
                    print(param.grad)
                    param.grad.zero_()
                    params[name].grad = None
