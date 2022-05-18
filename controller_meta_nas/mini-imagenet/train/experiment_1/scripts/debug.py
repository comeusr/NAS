from MiniImagenet import MiniImagenet
from searchMeta import MAMLFewShotClassifier
import argparse
import torch.nn as nn
import torch
from utils import *
from torch.utils.data import DataLoader
from SearchModel import Network

parser = argparse.ArgumentParser("mini-imagenet")
parser.add_argument('--dataset', type=str, default='mini-imagenet', help='dataset')
parser.add_argument('--checkname', type=str, default='finetune', help='checkname')
parser.add_argument('--run', type=str, default='controller_meta_nas', help='run_path')
parser.add_argument('--data_path', type=str, default='./data/miniImagenet', help='path to data')
parser.add_argument('--pretrained_model', type=str, default='', help='path to pretrained model')
parser.add_argument('--seed', type=int, default=222, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epoch', type=int, help='epoch number', default=10)
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
parser.add_argument('--n_way', type=int, help='n way', default=5)
parser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
parser.add_argument('--batch_size', type=int, default=15000, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=100, help='test batch size')
parser.add_argument('--meta_batch_size', type=int, help='meta batch size, namely task num', default=4)
parser.add_argument('--meta_test_batch_size', type=int, help='meta test batch size', default=1)
parser.add_argument('--report_freq', type=float, default=30, help='report frequency')
parser.add_argument('--img_size', type=int, help='img_size', default=84)
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--imgc', type=int, help='imgc', default=3)
parser.add_argument('--meta_lr_alpha', type=float, help='meta-level outer learning rate (theta)', default=3e-4)
parser.add_argument('--update_lr_alpha', type=float, help='task-level inner update learning rate (theta)', default=3e-3)
parser.add_argument('--meta_lr_w', type=float, help='meta-level outer learning rate (w)', default=1e-3)
parser.add_argument('--update_lr_w', type=float, help='task-level inner update learning rate (w)', default=0.01)
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-3, help='weight decay')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--controller_step', type=int, default=1, help='Update Step for Controller before Darts')
parser.add_argument('--update_lr_controller', type=float, default=1e-2, help='task-level inner learning reate(controller)')
parser.add_argument('--meta_lr_controller', type=float, default=1e-3, help='meta-level meta learning rate(controller)')
parser.add_argument('--num_nodes', type=int, default=4, help='number of layers in one cell')
parser.add_argument('--controller_hid', type=int, default=5, help='Hidden size of Controller Lstm cell')
parser.add_argument('--nums_arch', type=float, default=4, help='Numbers of Evaluate Architecture')
parser.add_argument('--evaluate_channel', type=int, default=16, help='Evaluate Network Channel')
parser.add_argument('--evaluate_layers', type=int, default=2, help='Evaluate Network Layers')
parser.add_argument('--temp', type=float, default=2.5, help='Gumbel-Softmax Temperature for Controller Output')
parser.add_argument('--anneal_rate', type=float, default=0.00003, help='The Convergence Speed of Temperature')
parser.add_argument('--pretrained', type=str, default='True')
parser.add_argument('--pretrain_path', type=str, default='experiment_11/model_best.pth.tar')
parser.add_argument('--inner_lr_w', type=float, default=0.1, help='The lr for fixed architecture')
parser.add_argument('--retrain_epoch', type=int, default=5, help='Number of Retrain Epochs')
parser.add_argument('--task_learning_rate', type=float, default=0.1, help='Inner Optimizer lr')
parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', type=str, default='True')
parser.add_argument('--enable_inner_loop_optimizable_bn_params', type=str, default='True')
parser.add_argument('--first_order_to_second_order_epoch', type=int, default=2)
parser.add_argument('--second_order', type=str, default='True')
parser.add_argument('--use_multi_step_loss_optimization', type=str, default='True')
parser.add_argument('--min_learning_rate', type=float, default=0.00001, help='Min learning rate')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--multi_step_loss_num_epochs', type=int, default=5)


search_args = get_args(parser.parse_args([]))

criterion = nn.CrossEntropyLoss()
device = torch.device('cuda:1')

debug_data = MiniImagenet('/media/jiangnan/Data/ziyi/NAS/data/miniImagenet', mode='train', n_way=search_args.n_way, k_shot=search_args.k_spt, k_query=search_args.k_qry,
                          batch_size=1, resize=search_args.img_size)
debug_loader = DataLoader(debug_data, 1)

model = MAMLFewShotClassifier(device, search_args, criterion)
update_time = AverageMeter()


# def get_inner_loop_parameter_dict(params):
#     """
#     Returns a dictionary with the parameters to use for inner loop updates.
#     :param params: A dictionary of the network's parameters.
#     :return: A dictionary of the parameters to use for the inner loop optimization process.
#     """
#     return {
#         name: param.to(device=device)
#         for name, param in params
#         if param.requires_grad
#            and (
#                    not True
#                    and "norm_layer" not in name
#                    or True
#            )
#     }


# test = Network(C=search_args.init_channels, args=search_args, num_classes=5, criterion=criterion, layers=2, device=device)
# params = get_inner_loop_parameter_dict(test.named_parameters())
# for k in params:
#     print(k)
for step, batch in enumerate(debug_loader):
    model.run_train_iter(batch, 0, update_time)


