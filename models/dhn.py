import time
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
from networks.model_loader import load_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.evaluate import mean_average_precision, pr_curve
from utils import util


def train(
        train_dataloader,
        query_dataloader,
        retrieval_dataloader,
        arch,
        code_length,
        device,
        lr,
        max_iter,
        lamda,
        beta,
        b_reg,
        topk,
        evaluate_interval,
        pre_train
    ):
    """
    Training model.

    Args
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        arch(str): CNN model name.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        lamda(float): Hyper-parameters.
        topk(int): Compute top k map.
        evaluate_interval(int): Interval of evaluation.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Load model
    model = load_model(arch, code_length, pre_train).to(device)
    
    # Create criterion, optimizer, scheduler
    criterion = DHNLoss(lamda, beta)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        max_iter,
        lr/100,
    )

    # Initialization
    best_map = 0.

    # Training
    map_history = list()
    time_history = list()
    mAP = 0.0
    checkpoint = dict()
    training_time = 0.
    iteration = tqdm(range(max_iter))
    for it in iteration:
        tic = time.time()
        running_loss = list()
        model.train()
        for data, targets, index in train_dataloader:
            batch_size_ = targets.size(0)
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            optimizer.zero_grad()

            # Create similarity matrix
            S = (targets @ targets.t() > 0).float()
            outputs = model(data)
            if b_reg:
                s_vector = util.rand_unit_rect(batch_size_, code_length)
                noises = gene_noise(outputs.cpu().data.numpy(), s_vector)
                noises = torch.from_numpy(noises).float().to(device)
            else:
                noises = None
            loss = criterion(outputs, S, noises)

            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        training_time += time.time() - tic

        iteration.set_description('[iter:{}/{}][loss:{:.2f}][map:{:.2f}][time:{:.2f}]'.format(
            it + 1,
            max_iter,
            np.mean(running_loss),
            mAP,
            training_time
        ))

        # Evaluate
        if it % evaluate_interval == evaluate_interval - 1 or it == max_iter - 1:
            # Generate hash code
            query_code = generate_code(model, query_dataloader, code_length, device)
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

            # Compute map
            mAP = mean_average_precision(
                query_code.to(device),
                retrieval_code.to(device),
                query_targets.to(device),
                retrieval_targets.to(device),
                device,
                topk,
            )
            if it == max_iter - 1:
                Precision, Recall = pr_curve(
                    query_code.to(device),
                    retrieval_code.to(device),
                    query_targets.to(device),
                    retrieval_targets.to(device),
                    device,
                )
                checkpoint.update({'precision': Precision, 'recall': Recall})

            map_history.append(mAP)
            time_history.append(training_time)
            # Checkpoint
            if best_map < mAP:
                best_map = mAP
                checkpoint.update({
                    'model': model.state_dict(),
                    'qB': query_code.cpu(),
                    'rB': retrieval_code.cpu(),
                    'qL': query_targets.cpu(),
                    'rL': retrieval_targets.cpu(),
                    'map': best_map
                })
            else:
                Precision, Recall = pr_curve(
                    query_code.to(device),
                    retrieval_code.to(device),
                    query_targets.to(device),
                    retrieval_targets.to(device),
                    device,
                )
                checkpoint.update({'precision': Precision, 'recall': Recall})
    checkpoint.update({'map_history': map_history})
    checkpoint.update({'time_history': time_history})
    return checkpoint

def gene_noise(embeedings, noises):
    data_size = embeedings.shape[0]
    assgined_noise = dict(zip(range(data_size), noises))
    # do forward pass on batch - features
    losses = np.empty(shape=(data_size, data_size), dtype='float64')

    # calculate l2 loss between all...
    # noises randomly assigned and features generated by the network
    for i in range(data_size):
        fts = np.repeat(np.expand_dims(embeedings[i], axis=0), data_size, axis=0)
        l2 = np.linalg.norm(fts - noises, axis=1)
        losses[i] = l2

    # rearrange noises such that the total loss is minimal (hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(losses)
    for r, c in zip(row_ind, col_ind):
        assgined_noise[r] = noises[c]

    # get the same noises as before but in new assignment order
    new_noise = np.empty(shape=noises.shape, dtype='float64')
    for i in range(data_size):
        new_noise[i] = assgined_noise[i]
    return new_noise