import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
from networks.model_loader import load_model
from utils.evaluate import mean_average_precision, pr_curve
from utils.ls_schedule import step_lr_scheduler
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
        gamma,
        step,
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
    criterion = Hashnet_loss(lamda, beta)
    parameter_list = [{"params": model.classifier.parameters(), "lr": 1},
                      {"params": model.hash_layer.parameters(), "lr": 10}]
    optimizer = optim.SGD(
        parameter_list, lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True
    )
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

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
            optimizer = step_lr_scheduler(param_lr, optimizer, it, gamma, step, lr)
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


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code


class Hashnet_loss(nn.Module):
    """
    DHN loss function.
    """
    def __init__(self, lamda, beta):
        super(Hashnet_loss, self).__init__()
        self.lamda = lamda
        self.beta = beta

    def forward(self, H, S, noises):
        # Inner product
        theta = self.lamda * H @ H.t() / 2

        # log(1+e^z) may be overflow when z is large.
        # We convert log(1+e^z) to log(1 + e^(-z)) + z.
        loss = torch.log(1 + torch.exp(-(theta).abs())) + theta.clamp(min=0) - S * theta
        S1 = torch.sum(S > 0)
        S0 = torch.sum(S <= 0)
        S12 = S0 + S1
        masks = torch.ones_like(S)
        masks[S > 0] = S12 / S1
        masks[S <= 0] = S12 / S0

        loss = loss * masks
        if noises is not None:
            loss = loss.mean() - self.beta * H.mul(noises).sum(dim=-1).mean()
        else:
            loss = loss.mean()
        return loss


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