import torch
import torch.optim as optim
import time

from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.optimize import linear_sum_assignment
from networks.model_loader import load_model
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
    mu,
    nu,
    eta,
    beta,
    b_reg,
    topk,
    evaluate_interval,
    pre_train
 ):
    """
    Training model.

    Args
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
        arch(str): CNN model name.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter: int
        Maximum iteration
        mu, nu, eta(float): Hyper-parameters.
        topk(int): Compute mAP using top k retrieval result
        evaluate_interval(int): Evaluation interval.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Construct network, optimizer, loss
    model = load_model(arch, code_length, pre_train).to(device)
    criterion = DSDHLoss(eta, beta)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, max_iter, 1e-7)

    # Initialize
    N = len(train_dataloader.dataset)
    B = torch.randn(code_length, N).sign().to(device)
    U = torch.zeros(code_length, N).to(device)
    train_targets = train_dataloader.dataset.get_onehot_targets().to(device)
    S = (train_targets @ train_targets.t() > 0).float()
    Y = train_targets.t()
    best_map = 0.

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
        # CNN-step
        for data, targets, index in train_dataloader:
            _batch_size = targets.size(0)
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            U_batch = model(data).t()
            U[:, index] = U_batch.data
            if b_reg:
                s_vector = util.rand_unit_rect(_batch_size, code_length)
                noises = gene_noise(U_batch.cpu().data.numpy().T, s_vector)
                noises = torch.from_numpy(noises.T).float().to(device)
            else:
                noises = None
            loss = criterion(U_batch, U, S[:, index], B[:, index], noises)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()

        # W-step
        W = torch.inverse(B @ B.t() + nu / mu * torch.eye(code_length, device=device)) @ B @ Y.t()

        # B-step
        B = solve_dcc(W, Y, U, B, eta, mu)

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
            epoch_loss = calc_loss(U, S, Y, W, B, mu, nu, eta)

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

            # Save checkpoint
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


def solve_dcc(W, Y, U, B, eta, mu):
    """
    DCC.
    """
    for i in range(B.shape[0]):
        P = W @ Y + eta / mu * U

        p = P[i, :]
        w = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i+1:, :]))
        B_prime = torch.cat((B[:i, :], B[i+1:, :]))

        B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

    return B


def calc_loss(U, S, Y, W, B, mu, nu, eta):
    """
    Compute loss.
    """
    theta = torch.clamp(U.t() @ U / 2, min=-100, max=50)
    metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).sum()
    classify_loss = ((Y - W.t() @ B) ** 2).sum()
    regular_loss = (W ** 2).sum()
    quantization_loss = ((B - U) ** 2).sum()

    loss = (metric_loss + mu * classify_loss + nu * regular_loss + eta * quantization_loss) / S.shape[0]

    return loss.item()


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


class DSDHLoss(torch.nn.Module):
    def __init__(self, eta, beta):
        super(DSDHLoss, self).__init__()
        self.eta = eta
        self.beta = beta

    def forward(self, U_batch, U, S, B, noises):
        theta = U.t() @ U_batch / 2

        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)

        metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()
        quantization_loss = (B - U_batch).pow(2).mean()
        if noises is not None:
            noise_loss = U_batch.mul(noises).mean()
            loss = metric_loss + self.eta * quantization_loss + self.beta * noise_loss
        else:
            loss = metric_loss + self.eta * quantization_loss
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