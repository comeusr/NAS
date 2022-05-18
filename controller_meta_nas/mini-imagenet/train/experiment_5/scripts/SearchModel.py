import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
from IPython.core.debugger import set_trace
from SearchModel import *
from utils import *
from genotype import PRIMITIVES, Genotype
from operation import *
from metalayer import *
from torchviz import make_dot

NEEDS_PARAM_OP = ['FactorizedReduce', 'SepConv', 'DilConv']


class MixedOp(nn.Module):

    def __init__(self, C, stride, idx=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, params=None, cell_idx=None, mix_idx=None):
        if params is not None:
            params = extract_top_level_dict(params)['_ops']
            params_list = extract_top_level_dict(params)

        result = []

        for i in range(len(weights)):

            if self._ops[i]._get_name() in NEEDS_PARAM_OP:
                result.append(weights[i] * self._ops[i](x, params_list[str(i)]))

            else:
                result.append(weights[i] * self._ops[i](x))

        return sum(result)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, j)
                self._ops.append(op)

    def forward(self, s0, s1, weights, params=None, cell_idx=None):
        if params is not None:
            params_dict = extract_top_level_dict(params)
            params_pre0 = params_dict['preprocess0']
            params_pre1 = params_dict['preprocess1']
            params_ops = extract_top_level_dict(params_dict['_ops'])

        s0 = self.preprocess0(s0, params_pre0)
        s1 = self.preprocess1(s1, params_pre1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):

            temp = []
            for j, s in enumerate(states):
                temp.append(
                    self._ops[offset + j](s, weights[offset + j], params_ops[str(offset + j)], cell_idx=cell_idx, mix_idx=offset + j))
            new_s = sum(temp)
            offset += len(states)
            states.append(new_s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, args, C, num_classes, layers, criterion, device, steps=4, multiplier=4, stem_multiplier=3, pretrained=False):
        super(Network, self).__init__()
        self._args = args
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.temp = args.temp
        self.num_step = args.update_step
        self.device = device

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

        self.stem0 = stem0(C_curr, kernel_size=3, stride=1, padding=1, bias=False, device=device, args=args)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        if layers == 2:
            for i in range(layers):
                if i == 1:
                    C_curr *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                reduction_prev = reduction
                self.cells += [cell]
                C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier_meta_nas = MetaLinearLayer(C_prev, num_classes, use_bias=True)  # nn.Linear(C_prev, num_classes)

        if pretrained == True:
            self._load_pretrained_alphas()
        else:
            self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

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

    def forward(self, input, alphas, temp, step, params=None, training=True, backup_running_statistics=True):
        if params is not None:
            params = extract_top_level_dict(params)
            params_stem = params['stem0']
            params_cell = extract_top_level_dict(params['cells'])
            params_classifier = params['classifier_meta_nas']

        s0 = s1 = self.stem0(input, params_stem, step=step, training=training,
                             backup_running_statistics=backup_running_statistics)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self._args.layers == 1:
                    weights = self.arch_dist(temp, alphas[0])
                else:
                    weights = self.arch_dist(temp, alphas[1])
            else:
                weights = self.arch_dist(temp, alphas[0])
            s0, s1 = s1, cell(s0, s1, weights, params=params_cell[str(i)], cell_idx=i)

        out = self.global_pooling(s1)
        # make_dot(out, dict(self.named_parameters())).render('debug_partial', format='pdf')
        logits = self.classifier_meta_nas(out.view(out.size(0), -1), params_classifier)

        # make_dot(logits, params_stem).render('stem0', format='pdf')

        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = 1e-3 * nn.init.normal_(torch.empty(k, num_ops)).to(self.device)  # self.alphas_normal.shape = [14, 8]
        self.alphas_normal.requires_grad = True
        self.alphas_reduce = 1e-3 * nn.init.normal_(torch.empty(k, num_ops)).to(self.device)
        self.alphas_reduce.requires_grad = True
        if self._args.layers == 1:
            self._arch_parameters = [
                self.alphas_reduce
            ]
        else:
            self._arch_parameters = [
                self.alphas_normal,
                self.alphas_reduce,
            ]

    def _load_alphas(self, alphas):
        self.alphas_normal.data.copy_(alphas[0])
        self.alphas_reduce.data.copy_(alphas[1])
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def _load_pretrained_alphas(self):
        self.alphas_normal = torch.load(self._args.pretrained_model)['state_dict_theta'][0].to(
            self.device)  # self.alphas_normal.shape = [14, 8]
        self.alphas_normal.requires_grad = True
        self.alphas_reduce = torch.load(self._args.pretrained_model)['state_dict_theta'][1].to(self.device)
        self.alphas_reduce.requires_grad = True
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def arch_dist(self, temp, alpha):

        m = F.gumbel_softmax(alpha, tau=temp, hard=False)

        return m

    def hard_arch(self, temp, alpha):
        m = F.gumbel_softmax(alpha, tau=temp, hard=True)
        return m

    def evaluate_forward(self, input, alphas, temp):
        s0 = s1 = self.stem0(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self._args.layers == 1:
                    weights = self.hard_arch(temp, alphas[0])
                else:
                    weights = self.hard_arch(temp, alphas[1])
            else:
                weights = self.arch_dist(temp, alphas[0])
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        if self._args.checkname == 'darts' or self._args.checkname == 'enas':
            logits = self.classifier(out.view(out.size(0), -1))
        else:
            logits = self.classifier_meta_nas(out.view(out.size(0), -1))

        return logits

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n

                W = weights[start:end].copy()

                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        if self._args.layers == 1:
            gene_reduce = _parse(F.gumbel_softmax(self.alphas_reduce, tau=self.temp, dim=-1).data.cpu().numpy())
            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            genotype = Genotype(
                normal=[], normal_concat=[],
                reduce=gene_reduce, reduce_concat=concat
            )
        else:
            gene_normal = _parse(F.gumbel_softmax(self.alphas_normal, tau=self.temp, dim=-1).data.cpu().numpy())
            gene_reduce = _parse(F.gumbel_softmax(self.alphas_reduce, tau=self.temp, dim=-1).data.cpu().numpy())

            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            genotype = Genotype(
                normal=gene_normal, normal_concat=concat,
                reduce=gene_reduce, reduce_concat=concat
            )

        return genotype

    def evaluate(self, x_qry, y_qry):

        with torch.no_grad():
            loggits = self.evaluate_forward(x_qry, self.arch_parameters(), self.temp)
            pred = F.softmax(loggits, dim=1).argmax(dim=1)
            correct = torch.eq(pred, y_qry).sum().item()
            total = y_qry.shape[0]

        return correct / total
