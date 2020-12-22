import torch
import torch.optim as optim
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.optimize import linear_sum_assignment
import numpy as np

from networks.model_loader import load_model
from utils.evaluate import mean_average_precision, pr_curve
from utils import util

class DPSHLoss(torch.nn.Module):
    def __init__(self, eta, beta):
        super(DPSHLoss, self).__init__()
        self.eta = eta
        self.beta = beta

    def forward(self, U_cnn, U, S, noise=None):
        theta = U_cnn @ U.t() / 2

        # Prevent overflow
        theta = torch.clamp(theta, min=-100, max=50)

        pair_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()
        regular_term = (U_cnn - U_cnn.sign()).pow(2).mean()
        if noise is not None:
            print('{:.4f}, {:.4f}, {:.4f}'.format(pair_loss.item(), regular_term.item(), self.beta * U_cnn.mul(noise).mean().item()))
            loss = pair_loss + self.eta * regular_term - self.beta * U_cnn.mul(noise).mean()
        else:
            loss = pair_loss + self.eta * regular_term

        return loss


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
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

def train(train_dataloader, query_dataloader, retrieval_dataloader, arch, code_length, device, lr, max_iter,
              eta, beta, b_reg, topk, evaluate_interval, pre_train):
    ### create model
    model = load_model(arch, code_length, pre_train).to(device)
    criterion = DPSHLoss(eta, beta)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, max_iter, 1e-7)

    # Initialization
    N = len(train_dataloader.dataset)
    U = torch.zeros(N, code_length).to(device)
    train_targets = train_dataloader.dataset.get_onehot_targets().to(device)

    map_history = []
    best_map = 0
    checkpoint = dict()
    time_history = list()
    mAP = 0.0
    training_time = 0
    iteration = tqdm(range(max_iter))
    for epoch in iteration:
        tic = time.time()
        running_loss = list()
        model.train()
        for data, targets, index in train_dataloader:
            batch_size_ = targets.size(0)
            data, targets = data.to(device), targets.to(device).float()
            optimizer.zero_grad()
            S = (targets @ train_targets.t() > 0).float()
            U_cnn = model(data)
            U[index, :] = U_cnn.data
            if b_reg:
                s_vector = util.rand_unit_rect(batch_size_, code_length)
                noises = gene_noise(U_cnn.cpu().data.numpy(), s_vector)
                noises = torch.from_numpy(noises).float().to(device)
            else:
                noises = None

            loss = criterion(U_cnn, U, S, noises)

            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        scheduler.step()
        training_time += time.time() - tic
        iteration.set_description('[iter:{}/{}][loss:{:.2f}][map:{:.2f}][time:{:.2f}]'.format(
            epoch + 1,
            max_iter,
            np.mean(running_loss),
            mAP,
            training_time
        ))

        if epoch % evaluate_interval == evaluate_interval - 1 or epoch == max_iter - 1:
            query_code = generate_code(model, query_dataloader, code_length, device)
            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
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

            if epoch == max_iter - 1:
                Precision, Recall = pr_curve(
                    query_code.to(device),
                    retrieval_code.to(device),
                    query_targets.to(device),
                    retrieval_targets.to(device),
                    device,
                )
                checkpoint.update({'precision': Precision, 'recall': Recall})

            # Save checkpoint
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