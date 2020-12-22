import torch.optim as optim
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
# from tensorboardX import SummaryWriter

from tqdm import tqdm
from networks.model_loader import load_model
from utils.evaluate import mean_average_precision, pr_curve
from utils import util


def hashing_loss(b, cls, m, eta, beta=None, noises=None):
    """
    compute hashing loss
    automatically consider all n^2 pairs
    """
    y = (cls @ cls.t() == 0).float().view(-1)
    # y = (cls.unsqueeze(0) != cls.unsqueeze(1)).float().view(-1)
    dist = ((b.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(dim=2).view(-1)
    loss = (1 - y) / 2 * dist + y / 2 * (m - dist).clamp(min=0)

    if beta is not None:
        assert noises is None, 'noises should be None'
    if beta is not None and noises is not None:
        noise_loss = b.mul(noises)
        loss = loss.mean() + beta * noise_loss.mean() + eta * (b.abs() - 1).abs().sum(dim=1).mean() * 2
    else:
        print(loss.mean().item())
        loss = loss.mean() + eta * (b.abs() - 1).abs().sum(dim=1).mean() * 2

    return loss

def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor, n*code_length): Hash code.
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

def train(train_dataloader, query_dataloader, retrieval_dataloader, arch, code_length, device, lr, max_iter, eta, beta,
           m, b_reg, topk, evaluate_interval, pre_train):
    model = load_model(arch, code_length, pre_train).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.004)

    best_map = 0.0
    map_history = list()
    checkpoint = dict()
    time_history = list()
    mAP = 0.0
    training_time = 0
    iteration = tqdm(range(max_iter))
    for it in iteration:
        tic = time.time()
        running_loss = list()
        model.train()
        for i, (data, targets, index) in enumerate(train_dataloader):
            batch_size_ = targets.size(0)
            data, targets = data.to(device), targets.to(device)
            model.zero_grad()
            b = model(data)
            if b_reg:
                s_vector = util.rand_unit_rect(batch_size_, code_length)
                noises = gene_noise(b.cpu().data.numpy(), s_vector)
                noises = torch.from_numpy(noises).float().to(device)
                loss = hashing_loss(b, targets, m, eta, beta, noises)
            else:
                loss = hashing_loss(b, targets, m, eta)

            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        training_time += time.time() - tic
        iteration.set_description('[iter:{}/{}][loss:{:.2f}][map:{:.2f}][time:{:.2f}]'.format(
            it + 1,
            max_iter,
            np.mean(running_loss),
            mAP,
            training_time
        ))
        if it % evaluate_interval == evaluate_interval - 1 or it == max_iter - 1:
            test_loss = 0
            model.eval()
            for i, (data, targets, index) in enumerate(query_dataloader):
                data, targets = data.to(device), targets.to(device)

                b = model(data)
                loss = hashing_loss(b, targets, m, eta)
                test_loss += loss.data
            test_loss /= len(query_dataloader)
            query_code = generate_code(model, query_dataloader, code_length, device)
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()
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
            if best_map < mAP:
                best_map = mAP
                checkpoint.update({
                    'qB': query_code,
                    'qL': query_targets,
                    'rB': retrieval_code,
                    'rL': retrieval_targets,
                    'model': model.state_dict(),
                    'map': best_map,
                })
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