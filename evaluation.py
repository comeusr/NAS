import logging
import numpy as np
import torch
from MetaEval import MAMLFewShotClassifier
from Meta_firstOrder import Meta
import torch.backends.cudnn as cudnn
from utils import *
import random
import sys
import torch.nn as nn
from MiniImagenet import MiniImagenet
from torch.utils.data import DataLoader
import argparse

#'/content/drive/MyDrive/data/miniImagenet'

parser = argparse.ArgumentParser("mini-imagenet")

parser.add_argument('--dataset', type=str, default='mini-imagenet', help='dataset')
parser.add_argument('--checkname', type=str, default='finetune', help='checkname')
parser.add_argument('--run', type=str, default='controller_meta_nas', help='run_path')
parser.add_argument('--data_path', type=str, default='./data/miniImagenet', help='path to data')
parser.add_argument('--pretrained_model', type=str, default='./controller_meta_nas/experiment_26/model_best.pth.tar', help='path to pretrained model')
parser.add_argument('--seed', type=int, default=222, help='random seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
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
parser.add_argument('--temp', type=float, default=0.6, help='Gumbel-Softmax Temperature for Controller Output')
parser.add_argument('--anneal_rate', type=float, default=0.00003, help='The Convergence Speed of Temperature')
parser.add_argument('--pretrained', type=str, default='True')
parser.add_argument('--inner_lr_w', type=float, default=0.1, help='The lr for fixed architecture')
parser.add_argument('--task_learning_rate', type=float, default=0.1, help='Inner Optimizer lr')
parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', type=str, default='True')
parser.add_argument('--enable_inner_loop_optimizable_bn_params', type=str, default='True')
parser.add_argument('--first_order_to_second_order_epoch', type=int, default=2)
parser.add_argument('--second_order', type=str, default='True')
parser.add_argument('--use_multi_step_loss_optimization', type=str, default='True')
parser.add_argument('--min_learning_rate', type=float, default=0.00001, help='Min learning rate')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--multi_step_loss_num_epochs', type=int, default=10)

args = get_args(parser.parse_args([]))


def meta_decode(maml, step, x_spt, y_spt, x_qry, y_qry, logging):
    update_time = AverageMeter()
    end = time.time()

    accs, update_time, gene, normal, reduction = maml.finetunning(x_spt, y_spt, x_qry, y_qry, update_time, logging)

    print('Step [{step}]\t'
          'Update Time [{timer.val:.3f}]\t'
          'Test Accuracy {accs}\t'.format(step=step, timer=update_time, accs=accs)
          )
    print('Normal gene {}, Reduction gene{}'.format(normal, reduction))
    return gene


def meta_train(train_loader, model, epoch, writer):
    accs_all_train = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    update_time = AverageMeter()
    end = time.time()

    for step, data_batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        _, acc_spt, acc_qry, update_time = model.run_train_iter(data_batch, epoch, update_time)
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('Support/acc_iter', acc_spt[-1], step + len(train_loader) * epoch)
        writer.add_scalar('Query/acc_iter', acc_qry[-1], step + len(train_loader) * epoch)
        if step % args.report_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'W {update_w_time.val:.3f} ({update_w_time.avg:.3f})\t'
                         'support acc: {accs}\t query acc: {acc_qry}'.format(
                epoch, step, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                update_w_time=update_time, accs=acc_spt, acc_qry=acc_qry))
        accs_all_train.append(acc_qry)
        accs = np.array(accs_all_train).mean(axis=0).astype(np.float16)

    return accs


def meta_test(batch, model, epoch, nums_epoch, writer):
    accs_all_test = []
    update_time = AverageMeter()
    end = time.time()

    # len(x_spt.shape)=0, args.meta_test_batch_size=1
    accs_spt, acc_qry, update_time = model.run_validation_iter(batch, update_time)
    accs_all_test.append(acc_qry)
    end = time.time()

    logging.info('Epoch: [{0}][{1}]\t'
                 'W {update_w_time.val:.3f} ({update_w_time.avg:.3f})\t'
                 'test acc: {accs}'.format(
        epoch, nums_epoch,
        update_w_time=update_time, accs=acc_qry))

    # [b, update_step+1]
    # stds = np.array(accs_all_test).std(axis=0).astype(np.float16)
    # ci95 = 1.96 * stds / np.sqrt(np.array(accs_all_test).shape[0])

    # writer.add_scalar('val/acc', accs[-1], step // 500 + (len(train_loader) // 500 + 1) * epoch)
    writer.add_scalar('test/acc', acc_qry[-1], epoch)

    return acc_qry


def main(args):
    saver = Saver(args)
    # set log
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p',
                        filename=os.path.join(saver.experiment_dir, 'log.txt'), filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    saver.create_exp_dir(scripts_to_save=glob.glob('*.py') + glob.glob('*.sh') + glob.glob('*.yml'))
    saver.save_experiment_config()
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    logging.info(args)

    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Load the trained Model
    meta = Meta(args, criterion).to(device)
    pretrain_path = args.pretrained_model
    checkpoint = torch.load(pretrain_path)
    meta.load_state_dict(checkpoint['state_dict_w'])

    test_data = MiniImagenet(args.data_path, mode='test', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
                             batch_size=args.nums_arch, resize=args.img_size)
    train_date = MiniImagenet(args.data_path, mode='train', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
                              batch_size=args.batch_size, resize=args.img_size)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=args.num_workers)
    train_loader = DataLoader(train_date, batch_size=4, shuffle=True, num_workers=args.num_workers)

    accs_task = np.zeros(len(test_loader))

    for task_id, (x_spt, y_spt, x_qry, y_qry) in enumerate(test_loader):
        best_pred = 0
        x_spt_test, y_spt_test, x_qry_test, y_qry_test = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        test_batch = (x_spt_test, y_spt_test, x_qry_test, y_qry_test)

        genotype = meta_decode(meta, task_id, x_spt_test, y_spt_test, x_qry_test, y_qry_test, logging)
        # genotype = meta.model.genotype()

        logging.info(genotype)
        maml = MAMLFewShotClassifier(device, args, genotype).to(device)
        # exit()
        # print(step)
        # print(genotype)

        for epoch in range(args.epoch):
            logging.info('--------- Epoch: {} ----------'.format(epoch))
            accs_all_train = []
            # # TODO: how to choose batch data to update theta?
            acc = meta_train(train_loader, model=maml, epoch=epoch, writer=writer)
            logging.info('Train acc {}'.format(acc))
            test_accs = meta_test(test_batch, maml, epoch, nums_epoch=args.epoch, writer=writer)

            new_pred = test_accs[-1]
            if new_pred > best_pred:
                is_best = True
                best_pred = new_pred
            else:
                is_best = False
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': maml.module.state_dict() if isinstance(maml, nn.DataParallel) else maml.state_dict(),
                'best_pred': best_pred,
                'task_id': task_id,
            }, is_best)
            accs_task[task_id] = best_pred
            # accs = np.array(accs_all_train).mean(axis=0).astype(np.float16)
            #
            # return accs

# device = torch.device(args.device)
# criterion = nn.CrossEntropyLoss()
# criterion = criterion.to(device)
#
# # Load the trained Model
# meta = Meta(args, criterion).to(device)
#
# gene = meta.model.genotype()
# maml = MAMLFewShotClassifier(device, args, gene).to(device)
# print(maml.get_inner_loop_parameter_dict(maml.classifier.named_parameters()).keys())

main(args)
