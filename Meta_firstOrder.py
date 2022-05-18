import time
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from Model_firstOrder import Network
from controller import LSTMController
from collections import OrderedDict

from copy import deepcopy

import pdb


class Meta(nn.Module):

    def __init__(self, args, criterion):

        super().__init__()

        self.controller_step = args.controller_step
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_lr_w = args.update_lr_w
        self.inner_lr_alpha = args.update_lr_alpha
        self.inner_lr_controller = args.update_lr_controller
        self.meta_lr_w = args.meta_lr_w
        self.meta_lr_alpha = args.meta_lr_alpha
        self.meta_lr_controller = args.meta_lr_controller
        self.criterion = criterion

        self.model = Network(args, args.init_channels, args.n_way, args.layers, criterion)
        self.controller = LSTMController(args)

        self.inner_optimizer_w = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr_w)
        self.inner_optimizer_alpha = torch.optim.SGD(self.model.arch_parameters(), lr=self.inner_lr_alpha)
        self.meta_optimizer_w = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr_w)
        self.meta_optimizer_alpha = torch.optim.Adam(self.model.arch_parameters(), lr=self.meta_lr_alpha)
        self.inner_optimizer_controller = torch.optim.SGD(self.controller.parameters(), lr=self.inner_lr_controller)
        self.meta_optimizer_controller = torch.optim.Adam(self.controller.parameters(), lr=self.meta_lr_controller)

    def update(self, x_spt, y_spt, x_qry, y_qry):

        meta_batch_size, setsz, c_, h, w = x_spt.shape
        query_size = x_qry.shape[1]

        corrects_spt = [0 for _ in range(self.update_step + self.controller_step + 1)]
        corrects_qry = [0 for _ in range(self.update_step + self.controller_step + 1)]

        ''' copy weight and gradient '''
        controller_clone = OrderedDict([(k, v.clone()) for k, v in self.controller.named_parameters()])
        for p in self.controller.parameters():
            p.grad = torch.zeros_like(p.data)
        controller_grad_clone = [p.grad.clone() for p in self.controller.parameters()]
        theta_clone = [v.clone() for v in self.model.arch_parameters()]
        for p in self.model.arch_parameters():
            p.grad = torch.zeros_like(p.data)
        theta_grad_clone = [p.grad.clone() for p in self.model.arch_parameters()]
        w_clone = dict([(k, v.clone()) for k, v in self.model.named_parameters()])
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)
        w_grad_clone = [p.grad.clone() for p in self.model.parameters()]

        for i in range(meta_batch_size):
            ''' Update w '''
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.model(x_qry[i], alphas=self.model.arch_parameters(), temp=self.model.temp)
                logits = self.model(x_spt[i], alphas=self.model.arch_parameters(), temp=self.model.temp)

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects_qry[0] = corrects_qry[0] + correct

                pred = F.softmax(logits, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y_spt[i]).sum().item()
                corrects_spt[0] = corrects_spt[0] + correct

            for j in range(self.controller_step):
                alphas = self.controller()
                self.model._load_alphas(alphas)
                out = self.model(x_spt[i], self.model.arch_parameters(), temp=self.model.temp)

                loss = self.criterion(out, y_spt[i])

                self.inner_optimizer_alpha.zero_grad()
                self.inner_optimizer_w.zero_grad()
                self.inner_optimizer_controller.zero_grad()
                loss.backward()

                alpha_grad = [v.grad.clone() for v in self.model.arch_parameters()]
                controller_grad = torch.autograd.grad(alphas, self.controller.parameters(),
                                                      grad_outputs=alpha_grad, allow_unused=True)

                # if torch.isnan(controller_grad).any() or torch.isinf(controller_grad).any():
                #   print('Bad Gradients for Controller')

                for k, v in zip(self.controller.parameters(), controller_grad):
                    k.grad.copy_(v)

                self.inner_optimizer_controller.step()
                self.inner_optimizer_w.step()

                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.model(x_qry[i], alphas=self.model.arch_parameters(), temp=self.model.temp)
                    logits = self.model(x_spt[i], alphas=self.model.arch_parameters(), temp=self.model.temp)
                    loss_q = self.criterion(logits_q, y_qry[i])
                    # losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects_qry[j + 1] = corrects_qry[j + 1] + correct

                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    correct = torch.eq(pred, y_spt[i]).sum().item()
                    corrects_spt[j + 1] = corrects_spt[j + 1] + correct

            alphas = self.controller()
            self.model._load_alphas(alphas)

            for k in range(self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.model(x_spt[i], alphas=self.model.arch_parameters(), temp=self.model.temp)
                loss = self.criterion(logits, y_spt[i])

                self.inner_optimizer_w.zero_grad()
                self.inner_optimizer_alpha.zero_grad()
                loss.backward()
                self.inner_optimizer_w.step()
                self.inner_optimizer_alpha.step()

                logits_q = self.model(x_qry[i], alphas=self.model.arch_parameters(), temp=self.model.temp)
                loss_q = self.criterion(logits_q, y_qry[i])

                with torch.no_grad():
                    logits = self.model(x_spt[i], alphas=self.model.arch_parameters(), temp=self.model.temp)
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects_qry[self.controller_step + k + 1] = corrects_qry[self.controller_step + k + 1] + correct

                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    correct = torch.eq(pred, y_spt[i]).sum().item()
                    corrects_spt[self.controller_step + k + 1] = corrects_spt[self.controller_step + k + 1] + correct

            ''' Use first-order gradient average '''
            self.inner_optimizer_w.zero_grad()
            self.inner_optimizer_alpha.zero_grad()
            loss_q.backward()

            alpha_grad = [v.grad.clone() for v in self.model.arch_parameters()]
            controller_grad = torch.autograd.grad(self.controller(), self.controller.parameters(),
                                                  grad_outputs=alpha_grad, allow_unused=True)



            theta_grad_clone = [k + v.grad.clone() for k, v in zip(theta_grad_clone, self.model.arch_parameters())]
            w_grad_clone = [k + v.grad.clone() for k, v in zip(w_grad_clone, self.model.parameters())]
            controller_grad_clone += controller_grad
            for k, v in self.model.named_parameters():
                v.data.copy_(w_clone[k])
            for k, v in self.controller.named_parameters():
                v.data.copy_(controller_clone[k])

        self.meta_optimizer_w.zero_grad()
        for k, v in zip(w_grad_clone, self.model.parameters()):
            v.grad.copy_(k / meta_batch_size)
        self.meta_optimizer_w.step()

        self.meta_optimizer_controller.zero_grad()
        for k, v in zip(controller_grad_clone, self.controller.parameters()):
            v.grad.copy_(k / meta_batch_size)
        self.meta_optimizer_controller.step()

        accs_w = np.array(corrects_qry) / (query_size * meta_batch_size)
        accs_spt = np.array(corrects_spt) / (setsz * meta_batch_size)

        return accs_w, accs_spt

    def forward(self, x_spt, y_spt, x_qry, y_qry, update_theta_w_time):

        start = time.time()
        accs, accs_spt = self.update(x_spt, y_spt, x_qry, y_qry)
        update_theta_w_time.update(time.time() - start)

        return accs, accs_spt, update_theta_w_time

    def _finetune(self, model, controller, optimizer_alpha, optimizer_w, optimizer_controller, x_spt, y_spt, x_qry, y_qry):

        setsz, c, h, w = x_spt.shape
        query_size = x_qry.shape[0]

        # corrects_spt = [0 for _ in range(self.update_step_test + self.controller_step + 1)]
        corrects_qry = [0 for _ in range(self.update_step_test + self.controller_step + 1)]
        for v in controller.parameters():
            v.grad = torch.zeros_like(v.data)

        with torch.no_grad():
            # [setsz, nway]
            logits_q = model(x_qry, alphas=model.arch_parameters(), temp=model.temp)
            # logits = self.model(x_spt[i], alphas=self.model.arch_parameters())
            # loss_q = self.criterion(logits_q, y_qry[i])
            # losses_q[1] += loss_q
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects_qry[0] = corrects_qry[0] + correct

        for j in range(self.controller_step):
            alphas = controller()
            model._load_alphas(alphas)
            out = model(x_spt, model.arch_parameters(), model.temp)
            loss = self.criterion(out, y_spt)

            optimizer_controller.zero_grad()
            optimizer_alpha.zero_grad()
            optimizer_w.zero_grad()
            loss.backward()

            alpha_grad = [v.grad.clone() for v in model.arch_parameters()]
            controller_grad = torch.autograd.grad(alphas, controller.parameters(),
                                                  grad_outputs=alpha_grad, allow_unused=True)

            for k, v in zip(controller_grad, controller.parameters()):
                v.grad.copy_(k)

            optimizer_controller.step()
            optimizer_w.step()

            with torch.no_grad():
                # [setsz, nway]
                alphas = controller()
                model._load_alphas(alphas)
                logits_q = model(x_qry, alphas=model.arch_parameters(), temp=model.temp)
                # logits = self.model(x_spt[i], alphas=self.model.arch_parameters())
                # loss_q = self.criterion(logits_q, y_qry[i])
                # losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects_qry[j + 1] = corrects_qry[j + 1] + correct

        alphas = controller()
        model._load_alphas(alphas)

        for j in range(self.update_step_test):
            out = model(x_spt, model.arch_parameters(), model.temp)
            loss = self.criterion(out, y_spt)

            optimizer_alpha.zero_grad()
            optimizer_w.zero_grad()
            loss.backward()
            optimizer_alpha.step()
            optimizer_w.step()

            with torch.no_grad():
                # [setsz, nway]
                logits_q = model(x_qry, alphas=model.arch_parameters(), temp=model.temp)
                # logits = self.model(x_spt[i], alphas=self.model.arch_parameters())
                # loss_q = self.criterion(logits_q, y_qry[i])
                # losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects_qry[self.controller_step + j + 1] = corrects_qry[self.controller_step + j + 1] + correct

        accs_qry = np.array(corrects_qry) / query_size
        gene = model.genotype()
        normal = model.arch_dist(model.temp, model.alphas_normal)
        reduction = model.arch_dist(model.temp, model.alphas_normal)

        return accs_qry, gene, normal, reduction

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, update_theta_w_time, logging):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [query_size, c_, h, w]
        :param y_qry:   [query_size]
        :return:
        """
        start = time.time()

        model = deepcopy(self.model)
        controller = deepcopy(self.controller)
        inner_optimizer_alpha = torch.optim.SGD(model.arch_parameters(), lr=self.inner_lr_alpha)
        inner_optimizer_w = torch.optim.SGD(model.parameters(), lr=self.inner_lr_w)
        inner_optimizer_controller = torch.optim.SGD(controller.parameters(), lr=self.inner_lr_controller)

        start = time.time()
        # accs_finetunning = self._update_theta_w_together_finetunning(model, inner_optimizer_theta,
        #                                                                                inner_optimizer_w, x_spt, y_spt,
        #                                                                                x_qry, y_qry)
        accs_finetunning, gene, normal, reduction = self._finetune(model, controller, inner_optimizer_alpha, inner_optimizer_w,
                                                                   inner_optimizer_controller, x_spt, y_spt, x_qry, y_qry)

        update_theta_w_time.update(time.time() - start)

        del model
        del controller

        return accs_finetunning, update_theta_w_time, gene, normal, reduction
