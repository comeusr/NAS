import torch.backends.cudnn as cudnn
from utils import *
import random
import logging
from MiniImagenet import MiniImagenet
import torch.nn as nn
from torch.utils.data import DataLoader
from searchMeta import MAMLFewShotClassifier
import argparse

parser = argparse.ArgumentParser("mini-imagenet")
parser.add_argument('--dataset', type=str, default='mini-imagenet', help='dataset')
parser.add_argument('--checkname', type=str, default='train', help='checkname')
parser.add_argument('--run', type=str, default='controller_meta_nas', help='run_path')
parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/data/miniImagenet', help='path to data')
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
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
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
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--multi_step_loss_num_epochs', type=int, default=5)

search_args = get_args(parser.parse_args([]))


def meta_train(train_loader, model, device, epoch, writer, args):
    accs_all_train = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    update_time = AverageMeter()
    end = time.time()

    for step, data_batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        losses, _, acc_spt, acc_qry, update_time = model.run_train_iter(data_batch, epoch, update_time)
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('Support/acc_iter', acc_spt[-1], step + len(train_loader) * epoch)
        writer.add_scalar('Query/acc_iter', acc_qry[-1], step + len(train_loader) * epoch)
        if step % 1 == 0:
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


def meta_test(valid_loader, model, device, epoch, writer, args):
    accs_all_test = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    update_time = AverageMeter()
    end = time.time()
    for step, data_batch in enumerate(valid_loader):
        data_time.update(time.time() - end)

        losses, _, acc_spt, acc_qry, update_time = model.run_validation_iter(data_batch, update_time)

        accs_all_test.append(acc_qry)
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Theta {update_theta_w_time.val:.3f} ({update_theta_w_time.avg:.3f})\t'
                         'test acc: {accs}'.format(
                epoch, step, len(valid_loader),
                batch_time=batch_time, data_time=data_time,
                update_theta_w_time=update_time,
                accs=acc_qry))

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    writer.add_scalar('val/acc', accs[-1], epoch)

    return accs


def search_main(args):
    saver = Saver(args)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p',
                        filename=os.path.join(saver.experiment_dir, 'log.txt'), filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

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

    mini = MiniImagenet(args.data_path, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batch_size=args.batch_size, resize=args.img_size)
    mini_test = MiniImagenet(args.data_path, mode='val', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batch_size=args.test_batch_size, resize=args.img_size)
    train_loader = DataLoader(mini, args.meta_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(mini_test, args.meta_test_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = MAMLFewShotClassifier(device, args, criterion=criterion).to(device)

    # logging.info(model)
    best_pred = 0

    for epoch in range(args.epoch):
        logging.info('--------- Epoch: {} ----------'.format(epoch))

        train_accs = meta_train(train_loader, model, device, epoch, writer, args)
        logging.info('[Epoch: {}]\t Train acc: {}'.format(epoch, train_accs))
        test_accs_theta = meta_test(valid_loader, model, device, epoch, writer, args)
        logging.info('[Epoch: {}]\t Test acc_theta: {}'.format(epoch, test_accs_theta))

        new_pred = test_accs_theta[-1]
        if new_pred > best_pred:
            is_best = True
            best_pred = new_pred
        else:
            is_best = False
        saver.save_checkpoint({
            'epoch': epoch,
            'state_dict_w': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'state_dict_theta': model.classifier.arch_parameters(),
            'best_pred': best_pred,
        }, is_best)


search_main(search_args)