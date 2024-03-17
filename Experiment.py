# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import torch
import argparse

from tqdm import *

from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed

global log

torch.set_default_dtype(torch.double)

class experiment:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def load_data(args):
        string = args.path + '/' + args.dataset + 'Matrix' + '.txt'
        tensor = np.loadtxt(open(string, 'rb'))
        return tensor

    @staticmethod
    def preprocess_data(data, args):
        data[data == -1] = 0
        return data


def get_pytorch_index(data):
    userIdx, servIdx = data.nonzero()
    values = data[userIdx, servIdx]
    idx = torch.as_tensor(np.vstack([userIdx, servIdx, values]).T)
    return idx


def get_train_valid_test_dataset(tensor, args):
    quantile = np.percentile(tensor, q=100)
    # tensor[tensor > quantile] = 0

    tensor = tensor / (np.max(tensor))  # 如果数据有分布偏移，记得处理数据

    trainsize = int(np.prod(tensor.size) * args.density)
    validsize = int((np.prod(tensor.size)) * 0.05) if args.valid else int((np.prod(tensor.size) - trainsize) * 1.0)

    rowIdx, colIdx = tensor.nonzero()
    p = np.random.permutation(len(rowIdx))
    rowIdx, colIdx = rowIdx[p], colIdx[p]

    trainRowIndex = rowIdx[:trainsize]
    trainColIndex = colIdx[:trainsize]

    traintensor = np.zeros_like(tensor)
    traintensor[trainRowIndex, trainColIndex] = tensor[trainRowIndex, trainColIndex]

    validStart = trainsize
    validRowIndex = rowIdx[validStart:validStart + validsize]
    validColIndex = colIdx[validStart:validStart + validsize]
    validtensor = np.zeros_like(tensor)
    validtensor[validRowIndex, validColIndex] = tensor[validRowIndex, validColIndex]

    testStart = validStart + validsize
    testRowIndex = rowIdx[testStart:]
    testColIndex = colIdx[testStart:]
    testtensor = np.zeros_like(tensor)
    testtensor[testRowIndex, testColIndex] = tensor[testRowIndex, testColIndex]

    return traintensor, validtensor, testtensor, quantile


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.data = exper_type.load_data(args)
        self.data = exper_type.preprocess_data(self.data, args)
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = get_train_valid_test_dataset(self.data,
                                                                                                              args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_tensor, self.valid_tensor,
                                                                         self.test_tensor, exper_type, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set,
                                                                                 self.test_set, args)
        args.log.only_print(
            f'Train_length : {len(self.train_loader) * args.bs} Valid_length : {len(self.valid_loader) * args.bs * 16} Test_length : {len(self.test_loader) * args.bs * 16}')

    def get_dataset(self, train_tensor, valid_tensor, test_tensor, exper_type, args):
        return (
            TensorDataset(train_tensor, exper_type, args),
            TensorDataset(valid_tensor, exper_type, args),
            TensorDataset(test_tensor, exper_type, args)
        )


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor, exper_type, args):
        self.args = args
        self.tensor = tensor
        self.indices = exper_type.get_pytorch_index(tensor)
        self.indices = self.delete_zero_row(self.indices)

    def __getitem__(self, idx):
        output = self.indices[idx, :-1]  # 去掉最后一列
        inputs = tuple(torch.as_tensor(output[i]) for i in range(output.shape[0]))
        value = torch.as_tensor(self.indices[idx, -1])  # 最后一列作为真实值
        return inputs, value

    def __len__(self):
        return self.indices.shape[0]

    def delete_zero_row(self, tensor):
        row_sums = tensor.sum(axis=1)
        nonzero_rows = (row_sums != 0).nonzero().squeeze()
        filtered_tensor = tensor[nonzero_rows]
        return filtered_tensor


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def forward(self, inputs):
        return 1

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            inputs, value = train_Batch
            pred = self.forward(inputs)
            loss = self.loss_function(pred, value)
            optimizer_zero_grad(self.optimizer)
            loss.backward()
            optimizer_step(self.optimizer)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        writeIdx = 0
        val_loss = 0.
        preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        self.prepare_test_model()
        for valid_Batch in tqdm(dataModule.valid_loader, disable=not self.args.program_test):
            inputs, value = valid_Batch
            pred = self.forward(inputs)
            val_loss += self.loss_function(pred, value).item()
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        self.scheduler.step(val_loss)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        for test_Batch in tqdm(dataModule.test_loader, disable=not self.args.program_test):
            inputs, value = test_Batch
            pred = self.forward(inputs)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return test_error


def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(args)
    monitor = EarlyStopping(args.patience)

    # Setup training tool
    model.setup_optimizer(args)
    model.max_value = datamodule.max_value
    train_time = []
    for epoch in range(args.epochs):
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        monitor.track(epoch, model.state_dict(), valid_error['MAE'])
        train_time.append(time_cost)

        if args.verbose and epoch % args.verbose == 0:
            log.only_print(
                f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vMAE={valid_error['MAE']:.4f} vRMSE={valid_error['RMSE']:.4f} vNMAE={valid_error['NMAE']:.4f} vNRMSE={valid_error['NRMSE']:.4f} time={sum(train_time):.1f} s")

        if monitor.early_stop:
            break

    model.load_state_dict(monitor.best_model)

    sum_time = sum(train_time[: monitor.best_epoch])

    results = model.test_one_epoch(datamodule) if args.valid else valid_error

    log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s\n')

    return {
        'MAE': results["MAE"],
        'RMSE': results["RMSE"],
        'NMAE': results["NMAE"],
        'NRMSE': results["NRMSE"],
        'TIME': sum_time,
    }


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash, log)
        for key in results:
            metrics[key].append(results[key])

    log('*' * 20 + 'Experiment Results:' + '*' * 20)

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    if args.record:
        log.save_result(metrics)

    log('*' * 20 + 'Experiment Success' + '*' * 20)

    return metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='rt')  #
    parser.add_argument('--model', type=str, default='CF')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.10)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--program_test', type=int, default=0)
    parser.add_argument('--valid', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--device', type=str, default='cpu')  # gpu cpu mps
    parser.add_argument('--bs', type=int, default=256)  #
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--decay', type=float, default=1e-3)
    parser.add_argument('--lr_step', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--saved', type=int, default=1)

    parser.add_argument('--loss_func', type=str, default='L1Loss')
    parser.add_argument('--optim', type=str, default='AdamW')

    # Hyper parameters
    parser.add_argument('--dimension', type=int, default=32)

    # Other Experiment
    parser.add_argument('--ablation', type=int, default=0)
    args = parser.parse_args([])
    return args


if __name__ == '__main__':
    args = get_args()
    set_settings(args)
    log = Logger(args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)
