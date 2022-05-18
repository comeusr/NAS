
from collections import OrderedDict
import glob

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import defaultdict
import logging
import time
from tensorboardX import SummaryWriter


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.run, args.dataset, args.checkname)
        self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
        run_list = sorted([int(m.split('_')[-1]) for m in self.runs])
        run_id = run_list[-1] + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best,  filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""

        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
            best_pred = state['best_pred']
            task_id = state['task_id']
            with open(os.path.join(self.experiment_dir, 'task_{}_best_pred.txt'.format(task_id)), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_acc = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            acc = float(f.readline())
                            previous_acc.append(acc)
                    else:
                        continue
                max_acc = max(previous_acc)
                if best_pred > max_acc:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best_all.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best_all.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        with open(logfile, 'w+') as log_file:
            p = OrderedDict()
            p['dataset'] = self.args.dataset
            p['seed'] = self.args.seed
            p['epoch'] = self.args.epoch
            p['n_way'] = self.args.n_way
            p['k_spt'] = self.args.k_spt
            p['k_qry'] = self.args.k_qry
            p['batch_size'] = self.args.batch_size
            p['test_batch_size'] = self.args.test_batch_size
            p['meta_batch_size'] = self.args.meta_batch_size
            p['meta_test_batch_size'] = self.args.meta_test_batch_size
            p['meta_lr_w'] = self.args.meta_lr_w
            p['update_lr_w'] = self.args.update_lr_w
            p['update_step'] = self.args.update_step
            p['update_step_test'] = self.args.update_step_test

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')

    def create_exp_dir(self, scripts_to_save=None):
        print('Experiment dir : {}'.format(self.experiment_dir))
        if scripts_to_save is not None:
            os.mkdir(os.path.join(self.experiment_dir, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(self.experiment_dir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def get_args(args):
    args_dict = vars(args)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False
        if key == "dataset_path":
            args_dict[key] = os.path.join(os.environ['DATASET_DIR'], args_dict[key])
            print(key, os.path.join(os.environ['DATASET_DIR'], args_dict[key]))

        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)
    return args


class Bunch(object):
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)
