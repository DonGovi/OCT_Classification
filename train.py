import os
import argparse

import time
import numpy as np
import pandas as pd

import torch
from tensorboardX import SummaryWriter

from model import DenseNet, octDataset


parser = argparse.ArgumentParser(description="Pytorch OCT Classification")
parser.add_argument("--exp", required=True, help="experiment name")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
args = parser.parse_args()

writer = SummaryWriter()

def get_list(path, ratio=4., split=True):
    sample_ids = os.listdir(path)
    
    if split:
        num = len(sample_ids)
        train_num = int(num * (ratio/(1.+ratio)))
        train_idx = np.random.choice(num, train_num, replace=False)
        train_ids = [sample_ids[i] for i in train_idx]
        val_ids = [i for i in sample_ids if not i in train_ids]
        print("Training set: %d, Validation set: %d" % (len(train_ids), len(val_ids)))
        
        return train_ids, val_ids
    else:
        print("Training set: %d" % len(sample_ids))
        
        return sample_ids
    
    
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input_var, target_var) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        '''
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        '''
        # compute output
        output = model(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        batch_size = target_var.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target_var.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    for batch_idx, (input_var, target_var) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        '''
        else:
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
        '''
        # compute output
        output = model(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        batch_size = target_var.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target_var.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.data[0], batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, data_path, label_file, save, n_epochs=500, ratio=4,
          batch_size=8, lr=0.1, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    if ratio:
        train_list, val_list = get_list(data_path, ratio=ratio, split=True)
        train_set = octDataset(data_path, train_list, label_file, argument=True)
        val_set = octDataset(data_path, val_list, label_file, argument=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0, collate_fn=train_set.collate_fn)
        valid_loader = torch.utils.data.DataLoader(val_set, batch_size=2, shuffle=True,num_workers=0, collate_fn=val_set.collate_fn)
        
    else:
        train_list = get_list(data_path, ratio=ratio, split=False)
        train_set = octDataset(data_path, train_list, label_file, argument=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=train_set.collate_fn)
        valid_loader = None

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        scheduler.step()
        _, train_loss, train_error = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        _, valid_loss, valid_error = test_epoch(
            model=model,
            loader=valid_loader,
            is_test=False
        )
        
        writer.add_scalars("v1/loss", {"train_loss": train_loss}, epoch)
        writer.add_scalars("v1/loss", {"valid_loss": valid_loss}, epoch)
        writer.add_scalars("v1/error", {"train_error": train_error}, epoch)
        writer.add_scalars("v1/error", {"valid_error": valid_error}, epoch)
        # Determine if model is the best
        if valid_loader and valid_error < best_error:
            best_error = valid_error
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save, args.exp+'_best_model.dat'))
        else:
            torch.save(model.state_dict(), os.path.join(save, args.exp+'_model.dat'))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))

    # Final test of model on test set
    '''
    model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    test_results = test_epoch(
        model=model,
        loader=test_loader,
        is_test=True
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)
    '''

if __name__ == "__main__":
    data_path = "F:/OCT/classification/non_stream/data/"
    label_file = "F:/OCT/classification/non_stream/ns_label.csv"
    save_pth = "E:/oct_classification/"
    densenet = DenseNet(small_inputs=False)
    train(densenet, data_path, label_file, save_pth, batch_size=1)