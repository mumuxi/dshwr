import torch
import numpy as np
import random
import os
from data.data_loader import load_data
import logging  # 引入logging模块
import time

#models
import models.dhn as dhn # check
import models.ssdh as ssdh # check
import models.hashnet as hashnet # check
import models.dsh as dsh # check
import models.dsdh as dsdh # check
import models.adsh as adsh
import models.dpsh as dpsh  # check
import models.dcn as dcn  # check

def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def dhn_run(query_dataloader, train_dataloader, retrieval_dataloader, arch='alexnet', lr=1e-5,
            code_length=[12,24,32,48], max_iter=500, topk=-1, gpu=None, lamda=1.0, beta=0.2, b_reg=False,
            evaluate_interval=10, pre_train=True, path='/data00/zq/dsh', seed=2020, saveflag=False, results_logger=None):
    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % gpu)

    # Training
    results = dict()
    for cl in code_length:
        checkpoint = dhn.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            arch,
            cl,
            device,
            lr,
            max_iter,
            lamda,
            beta,
            b_reg,
            topk,
            evaluate_interval,
            pre_train
        )
        print('[code_length:{}][map:{:.4f}]'.format(cl, checkpoint['map']))
        results[cl] = {'map_history': checkpoint['map_history'],
                       'time_history': checkpoint['time_history'],
                       'precision': checkpoint['precision'],
                       'recall': checkpoint['recall'],
                       'best_map': checkpoint['map']}

        # Save checkpoint
        if saveflag:
            torch.save(
                checkpoint,
                os.path.join(path, 'checkpoints', '{}_model_{}_code_{}_lamda_{}_beta_{}_map_{:.4f}.pt'.format(
                    'dhn',
                    arch,
                    cl,
                    lamda,
                    beta,
                    checkpoint['map']),
                )
            )
            checkpoint = dict()
        else:
            checkpoint = dict()
    if results_logger is not None:
        results_logger.info('dhn\n' + str(results))
    return results


def hashnet_run(query_dataloader, train_dataloader, retrieval_dataloader, arch='alexnet', lr=1e-3,
                code_length=[12, 24, 32, 48], max_iter=500, topk=-1, gpu=3, gamma=0.5, step=2000, sigmoid=10.,
                beta=0.2, b_reg=False, evaluate_interval=20, pre_train=True, path='/data00/zq/dsh', seed=2020,
                saveflag=False, results_logger=None):
    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % gpu)

    # Training
    results = dict()
    for cl in code_length:
        checkpoint = hashnet.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            arch,
            cl,
            device,
            lr,
            max_iter,
            gamma,
            step,
            sigmoid / cl if sigmoid < cl else 1,
            beta,
            b_reg,
            topk,
            evaluate_interval,
            pre_train
        )
        print('[code_length:{}][map:{:.4f}]'.format(cl, checkpoint['map']))
        results[cl] = {'map_history': checkpoint['map_history'],
                       'time_history': checkpoint['time_history'],
                       'precision': checkpoint['precision'],
                       'recall': checkpoint['recall'],
                       'best_map': checkpoint['map']}

        # Save checkpoint
        if saveflag:
            torch.save(
                checkpoint,
                os.path.join(path, 'checkpoints', '{}_model_{}_code_{}_beta_{}_map_{:.4f}.pt'.format(
                    'hashnet',
                    arch,
                    cl,
                    beta,
                    checkpoint['map']))
            )
    if results_logger is not None:
        results_logger.info('hashnet\n' + str(results))
    return results


def dsh_run(query_dataloader, train_dataloader, retrieval_dataloader, arch='alexnet', lr=1e-5,
            code_length=[12,24,32,48], max_iter=500, topk=-1, gpu=3, eta=0.01, beta=0.2,
            b_reg=False, evaluate_interval=10, pre_train=True, train_all=False,
            path='/data00/zq/dsh', seed=2020, saveflag=False, results_logger=None):
    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % gpu)
    # Training
    results = dict()
    for cl in code_length:
        print('[code length:{}]'.format(cl))
        if train_all:
            checkpoint = dsh.train(
                retrieval_dataloader,
                query_dataloader,
                retrieval_dataloader,
                arch,
                cl,
                device,
                lr,
                max_iter,
                eta,
                beta,
                cl * 2,
                b_reg,
                topk,
                evaluate_interval,
                pre_train
            )
        else:
            checkpoint = dsh.train(
                train_dataloader,
                query_dataloader,
                retrieval_dataloader,
                arch,
                cl,
                device,
                lr,
                max_iter,
                eta,
                beta,
                cl * 2,
                b_reg,
                topk,
                evaluate_interval,
                pre_train
            )
        print('[code_length:{}][map:{:.4f}]'.format(cl, checkpoint['map']))
        results[cl] = {'map_history': checkpoint['map_history'],
                       'time_history': checkpoint['time_history'],
                       'precision': checkpoint['precision'],
                       'recall': checkpoint['recall'],
                       'best_map': checkpoint['map']}

        # Save checkpoint
        if saveflag:
            torch.save(
                checkpoint,
                os.path.join(path, 'checkpoints', '{}_model_{}_code_{}_eta_{}_beta_{}_map_{:.4f}.pt'.format(
                    'dsh',
                    arch,
                    cl,
                    eta,
                    beta,
                    checkpoint['map']),
                             )
            )
    if results_logger is not None:
        results_logger.info('dsh\n' + str(results))
    return results

def dpsh_run(query_dataloader, train_dataloader, retrieval_dataloader, arch='alexnet', lr=1e-5,
             code_length=[12,24,32,48], max_iter=150, topk=-1, gpu=3, eta=0.1,  beta=0.2, b_reg=False,
             evaluate_interval=10, pre_train=True, path='/data00/zq/dsh', seed=2020, saveflag=False,
             results_logger=None):
    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % gpu)
    # Training
    results = dict()
    for cl in code_length:
        checkpoint = dpsh.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            arch,
            cl,
            device,
            lr,
            max_iter,
            eta,
            beta,
            b_reg,
            topk,
            evaluate_interval,
            pre_train
        )
        print('[code_length:{}][map:{:.4f}]'.format(cl, checkpoint['map']))
        results[cl] = {'map_history': checkpoint['map_history'],
                       'time_history': checkpoint['time_history'],
                       'precision': checkpoint['precision'],
                       'recall': checkpoint['recall'],
                       'best_map': checkpoint['map']}

        # Save checkpoint
        if saveflag:
            torch.save(
                checkpoint,
                os.path.join(path, 'checkpoints', '{}_model_{}_code_{}_eta_{}_beta_{}_map_{:.4f}.pt'.format(
                    'dpsh',
                    arch,
                    cl,
                    eta,
                    beta,
                    checkpoint['map']),
                             )
            )
    if results_logger is not None:
        results_logger.info('dpsh\n' + str(results))
    return results


def ssdh_run(query_dataloader, train_dataloader, retrieval_dataloader, arch='alexnet', lr=0.0001,
        code_length=[12,24,32,48], max_iter=150, topk=-1, gpu=3, alpha=1.0, alpha_pre=0.01,
        beta=1.0, gamma=1.0, eta=0.2, b_reg=False, multi_label=False, evaluate_interval=10,
        pre_train=True, path='/data00/zq/dsh', seed=2020, saveflag=False, results_logger=None):

    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % gpu)

    # Training
    results = dict()
    for cl in code_length:
        checkpoint = ssdh.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            arch,
            cl,
            device,
            lr,
            max_iter,
            alpha,
            alpha_pre,
            beta,
            gamma,
            eta,
            b_reg,
            multi_label,
            topk,
            evaluate_interval,
            pre_train
        )
        print('[code_length:{}][map:{:.4f}]'.format(cl, checkpoint['map']))
        results[cl] = {'map_history': checkpoint['map_history'],
                       'time_history': checkpoint['time_history'],
                       'precision': checkpoint['precision'],
                       'recall': checkpoint['recall'],
                       'best_map': checkpoint['map']}

        # Save checkpoint
        if saveflag:
            torch.save(
                checkpoint,
                os.path.join(path, 'checkpoints', '{}_model_{}_code_{}_eta_{}_map_{:.4f}.pt'.format(
                    'ssdh',
                    arch,
                    cl,
                    eta,
                    checkpoint['map']),
                             )
            )
    if results_logger is not None:
        results_logger.info('ssdh\n' + str(results))
    return results


#mu, nu, eta = 1, 0.1 and 55
def dsdh_run(query_dataloader, train_dataloader, retrieval_dataloader,  arch='alexnet', lr=1e-5,
        code_length=[12,24,32,48], max_iter=150, topk=-1, gpu=None, mu=1e-2, nu=1, eta=1e-2, beta=1e-2,
        b_reg=False, evaluate_interval=10, pre_train=True, path='/data00/zq/dsh', seed=2020, saveflag=False,
             results_logger=None):
    results = dict()
    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % gpu)
    # Training
    for cl in code_length:
        print('[code length:{}]'.format(cl))
        checkpoint = dsdh.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            arch,
            cl,
            device,
            lr,
            max_iter,
            mu,
            nu,
            eta,
            beta,
            b_reg,
            topk,
            evaluate_interval,
            pre_train
        )
        print('[code_length:{}][map:{:.4f}]'.format(cl, checkpoint['map']))
        results[cl] = {'map_history': checkpoint['map_history'],
                       'time_history': checkpoint['time_history'],
                       'precision': checkpoint['precision'],
                       'recall': checkpoint['recall'],
                       'best_map': checkpoint['map']}

        # Save checkpoint
        if saveflag:
            torch.save(
                checkpoint,
                os.path.join(path, 'checkpoints', '{}_model_{}_code_{}_beta_{}_map_{:.4f}.pt'.format(
                    'dsdh',
                    arch,
                    cl,
                    beta,
                    checkpoint['map']),
                             )
            )
    if results_logger is not None:
        results_logger.info('dsdh\n' + str(results))
    return results

def dcn_run(query_dataloader, train_dataloader, retrieval_dataloader, arch='alexnet', lr=1e-5,
            code_length=[12,24,32,48], max_iter=500, topk=-1, gpu=3, lamda=0.5, eta=5, beta=0.2,
            b_reg=False, evaluate_interval=10, pre_train=True, path='/data00/zq/dsh',
            seed=2020, saveflag=False, results_logger=None):
    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % gpu)
    # Training
    results = dict()
    for cl in code_length:
        print('[code length:{}]'.format(cl))
        checkpoint = dcn.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            arch,
            cl,
            device,
            lr,
            max_iter,
            lamda,
            eta,
            beta,
            b_reg,
            topk,
            evaluate_interval,
            pre_train
        )
        print('[code_length:{}][map:{:.4f}]'.format(cl, checkpoint['map']))
        results[cl] = {'map_history': checkpoint['map_history'],
                       'time_history': checkpoint['time_history'],
                       'precision': checkpoint['precision'],
                       'recall': checkpoint['recall'],
                       'best_map': checkpoint['map']}

        # Save checkpoint
        if saveflag:
            torch.save(
                checkpoint,
                os.path.join(path, 'checkpoints', '{}_model_{}_code_{}_eta_{}_beta_{}_map_{:.4f}.pt'.format(
                    'dcn',
                    arch,
                    cl,
                    eta,
                    beta,
                    checkpoint['map']),
                             )
            )
    if results_logger is not None:
        results_logger.info('dcn\n' + str(results))
    return results


def adsh_run(query_dataloader, train_dataloader, retrieval_dataloader, bits=[12,24,32,48], gpu=2, arch='alexnet',
             topk=-1, max_iter=150, epochs=3, batch_size=256, num_samples=2000, gamma=200, lr=0.001, beta=0.2,
             b_reg=False, pre_train=True, path='/data00/zq/dsh', seed=2020, saveflag=False, results_logger=None):
    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # Training
    results = dict()
    for cl in bits:
        print('[code length:{}]'.format(cl))
        checkpoint = adsh.train(
            query_dataloader.dataset,
            train_dataloader.dataset,
            retrieval_dataloader.dataset,
            num_samples,
            arch,
            cl,
            lr,
            max_iter,
            epochs,
            batch_size,
            gpu,
            gamma,
            beta,
            b_reg,
            pre_train,
            topk
        )
        print('[code_length:{}][map:{:.4f}]'.format(cl, checkpoint['map']))
        results[cl] = {'map_history': checkpoint['map_history'],
                       'time_history': checkpoint['time_history'],
                       'precision': checkpoint['precision'],
                       'recall': checkpoint['recall'],
                       'best_map': checkpoint['map']}

        # Save checkpoint
        if saveflag:
            torch.save(
                checkpoint,
                os.path.join(path, 'checkpoints', '{}_model_{}_code_{}_gama_{}_beta_{}_map_{:.4f}.pt'.format(
                    'adsh',
                    arch,
                    cl,
                    gamma,
                    beta,
                    checkpoint['map']),
                             )
            )

    if results_logger is not None:
        results_logger.info('adsh\n' + str(results))
    return results

if __name__ == '__main__':
    dataset = 'cifar-10'
    root = 'E:\\Research\\data_codes\\w-hash-experiments'
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(dataset, root, 100, 500, 8, 1)
    results = dcn_run(query_dataloader, train_dataloader, retrieval_dataloader, gpu=None)
    print(results)