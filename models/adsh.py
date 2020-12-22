import pickle
import os
import torch
import torch.nn as nn
import time

import numpy as np
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from networks.model_loader import load_model
from utils.evaluate import pr_curve
import torch.utils.data.sampler as sampler
from utils import util

INERVAL = 10

def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return

def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S

def encode(model, data_loader, num_data, bit):
    model.eval()
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    model.train()
    return B

def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 20, 30, 40, 50]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

#bits='12,24,32,48', gpu='2', arch='alexnet', num_query=1000, num_train=5000,
               # max_iter=150, epochs=3, batch_size=64, num_samples=2000, gamma=200, beta=200, learning_rate=0.001,
                #b_reg=False, pre_train=True
def train(dset_test, dset_database, dset_retrieval, num_samples, arch, code_length, learning_rate, max_iter, epochs, batch_size,
          gpu, gamma, beta, b_reg, pre_train, topK):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(2020)

    '''
    parameter setting
    '''
    weight_decay = 5 * 10 ** -4

    '''
    dataset preprocessing
    '''
    num_database, num_test = len(dset_database), len(dset_test)
    database_labels, test_labels, retrieval_labels = dset_database.get_onehot_targets(), dset_test.get_onehot_targets(), \
                                                     dset_retrieval.get_onehot_targets()

    '''
    model construction
    '''
    model = load_model(arch, code_length, pre_train)
    model.cuda()
    adsh_loss = ADSHLoss(gamma, beta, code_length, num_database)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    V = np.zeros((num_database, code_length))

    model.train()
    iteration = tqdm(range(max_iter))
    best_map = 0
    map_history = list()
    time_history = list()
    mAP = 0.0
    checkpoint = dict()
    training_time = 0.
    for iter in iteration:
        tic = time.time()
        running_loss = list()
        # sampling and construct similarity matrix
        select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
        _sampler = SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=batch_size,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=4)

        # learning deep neural network: feature learning
        sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, database_labels)
        U = np.zeros((num_samples, code_length), dtype=np.float)
        for epoch in range(epochs):
            for i, (train_input, train_label, batch_ind) in enumerate(trainloader):
                batch_size_ = train_label.size(0)
                u_ind = np.linspace(i * batch_size, np.min((num_samples, (i+1)*batch_size)) - 1, batch_size_, dtype=int)
                train_input = Variable(train_input.cuda())

                output = model(train_input)
                S = Sim.index_select(0, torch.from_numpy(u_ind))
                U[u_ind, :] = output.cpu().data.numpy()

                model.zero_grad()
                if b_reg:
                    s_vector = util.rand_unit_rect(batch_size, code_length)
                    noises = gene_noise(U[u_ind, :], s_vector)
                else:
                    noises = None
                loss = adsh_loss(output, V, S, V[batch_ind.cpu().numpy(), :], noises)
                running_loss.append(loss.item())
                loss.backward()
                optimizer.step()
        adjusting_learning_rate(optimizer, iter)

        # learning binary codes: discrete coding
        barU = np.zeros((num_database, code_length))
        barU[select_index, :] = U
        Q = -2*code_length*Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * barU
        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))

        training_time += time.time() - tic
        iteration.set_description('[iter:{}/{}][loss:{:.2f}][map:{:.2f}][time:{:.2f}]'.format(
            iter + 1,
            max_iter,
            np.mean(running_loss),
            mAP,
            training_time
        ))

        if iter % INERVAL == INERVAL - 1 or iter == max_iter - 1:
            testloader = DataLoader(dset_test, batch_size=256,
                                    shuffle=False,
                                    num_workers=4)
            qB = encode(model, testloader, num_test, code_length)
            # rB = V
            rB = encode(model, DataLoader(dset_retrieval, batch_size=256, shuffle=False, num_workers=4),
                        len(dset_retrieval), code_length)
            mAP = calc_topMap(qB, rB, test_labels.numpy(), retrieval_labels.numpy(), topK)
            if iter == max_iter - 1:
                Precision, Recall = pr_curve(
                    torch.from_numpy(qB),
                    torch.from_numpy(rB),
                    test_labels,
                    retrieval_labels,
                    torch.device("cpu"),
                )
                checkpoint.update({'precision': Precision, 'recall': Recall})

            map_history.append(mAP)
            time_history.append(training_time)
            # Checkpoint
            if best_map < mAP:
                best_map = mAP

                checkpoint.update({
                    'model': model.state_dict(),
                    'qB': qB,
                    'rB': rB,
                    'qL': test_labels.numpy(),
                    'rL': retrieval_labels.numpy(),
                    'map': best_map
                })
            else:
                Precision, Recall = pr_curve(
                    torch.from_numpy(qB),
                    torch.from_numpy(rB),
                    test_labels,
                    retrieval_labels,
                    torch.device("cpu"),
                )
                checkpoint.update({'precision': Precision, 'recall': Recall})
    checkpoint.update({'map_history': map_history})
    checkpoint.update({'time_history': time_history})
    return checkpoint

class ADSHLoss(nn.Module):
    def __init__(self, gamma, beta, code_length, num_train):
        super(ADSHLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.code_length = code_length
        self.num_train = num_train

    def forward(self, u, V, S, V_omega, noises=None):
        batch_size = u.size(0)
        V = Variable(torch.from_numpy(V).type(torch.FloatTensor).cuda())
        V_omega = Variable(torch.from_numpy(V_omega).type(torch.FloatTensor).cuda())
        S = Variable(S.cuda())
        square_loss = (u.mm(V.t()) - self.code_length * S) ** 2
        quantization_loss = self.gamma * (V_omega - u) ** 2
        if noises is not None:
            noise_loss = self.beta * u.mul(torch.from_numpy(noises).type(torch.FloatTensor).cuda())
            loss = (square_loss.sum() + quantization_loss.sum() + noise_loss.sum()) / (self.num_train * batch_size)
        else:
            loss = (square_loss.sum() + quantization_loss.sum()) / (self.num_train * batch_size)
        return loss

class SubsetSampler(sampler.Sampler):

    def __init__(self, indices):
        super(SubsetSampler, self).__init__(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.int)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map

def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = np.cast[int](np.dot(queryL[iter, :], retrievalL.transpose()) > 0)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(np.divide(count, tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap

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
