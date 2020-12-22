import torch
import torch.optim as optim
import torch.nn as nn
import time

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
# from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.optimize import linear_sum_assignment
from utils.evaluate import mean_average_precision, pr_curve


def ssdh_loss(output, prediction, Y, wc, alpha, alpha_pre, beta, gamma, eta, noises, multi_label=False):
    if multi_label:
        norm = (prediction - Y) ** 2
        zeros = ((Y == 0.0) & (prediction <= 0.)) | ((Y == 1.0) & (prediction >= 1))
        norm[zeros] = 0.0
        cls_loss = norm.mean()
    else:
        cls_loss = F.cross_entropy(prediction, Y.long(), reduction='mean')
    reg_q = ((output - 0.5) ** 2).mean(axis=-1).mean()
    reg_b = ((output.mean(axis=-1) - 0.5) ** 2).mean()
    reg_wc = (wc ** 2).mean()
    loss = alpha * cls_loss + alpha_pre * reg_wc - beta * reg_q + gamma * reg_b
    # print(loss.item())
    # print(output.max().item())
    # print(output.min().item())
    # print(wc.min().item())
    # print(wc.max().item())
    # print('{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(cls_loss.item(), reg_wc.item(), reg_q.item(), reg_b.item()))
    if noises is not None:
        loss += eta * output.mul(noises).mean()
    return loss


def train(train_dataloader, query_dataloader, retrieval_dataloader, arch, code_length, device, lr, max_iter, alpha, alpha_pre,
    beta, gamma, eta, b_reg, multi_label, topk, evaluate_interval, pre_train):
    # Construct network, optimizer, loss
    model = load_model(arch, code_length, pre_train).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=10 ** -5)

    # scheduler = CosineAnnealingLR(optimizer, max_iter, 1e-7)

    # Initialize
    # N = len(train_dataloader.dataset)
    # B = torch.randn(code_length, N).sign().to(device)
    # U = torch.zeros(code_length, N).to(device)
    num_classes = train_dataloader.dataset.num_classes
    wc = torch.randn(code_length, num_classes, device=device, dtype=torch.float, requires_grad=True)
    bc = torch.zeros(num_classes, device=device, dtype=torch.float, requires_grad=True)

    best_map = 0.

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
        # CNN-step
        for data, targets, index in train_dataloader:
            _batch_size = targets.size(0)
            data, targets = data.to(device), targets.to(device).float()
            optimizer.zero_grad()

            output = model(data)
            prediction = output.mm(wc) + bc
            if b_reg:
                s_vector = rand_unit_rect(_batch_size, code_length)
                noises = gene_noise(output.cpu().data.numpy(), s_vector)
                noises = torch.from_numpy(noises.T).float().to(device)
            else:
                noises = None
            loss = ssdh_loss(output, prediction, targets, wc, alpha, alpha_pre, beta, gamma, eta, noises, multi_label)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        # scheduler.step()
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
                topk
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
            code[index, :] = (((hash_code - 0.5).sign() + 1) / 2).cpu()

    model.train()
    return code

def rand_unit_rect(npoints, ndim):
    '''
    Generates "npoints" number of vectors of size "ndim"
    such that each vectors is a point on an "ndim" dimensional sphere
    that is, so that each vector is of distance 1 from the center
    npoints -- number of feature vectors to generate
    ndim -- how many features per vector
    returns -- np array of shape (npoints, ndim), dtype=float64
    '''
    vec = np.random.randint(0, 2, size=(npoints, ndim))
    return vec

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


def load_model(arch, code_length, pre_train=True):
    """
    Load cnn model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    #'E:Research\\data_codes\\w-hash-experiments\\networks\\vgg16-397923af.pth'
    if arch == 'alexnet':
        model = AlexNet(code_length)
        if pre_train:
            # state_dict = load_state_dict_from_url('alexnet-owt-4df8aa71.pth',
            #                                       'E:Research\\data_codes\\w-hash-experiments\\networks')
            model.load_state_dict(
                torch.load('/data00/zq/dsh/networks/alexnet-owt-4df8aa71.pth'),
                strict=False)
    elif arch == 'vgg16':
        model = VGG(make_layers(cfgs['D'], batch_norm=False), code_length)
        if pre_train:
            model.load_state_dict(
                torch.load('/data00/zq/dsh/networks/vgg16-397923af.pth'), strict=False)
    else:
        raise ValueError('Invalid model name!')

    return model


class AlexNet(nn.Module):

    def __init__(self, code_length):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096 ,1000),
        )

        self.classifier = self.classifier[:-1]
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, code_length),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.hash_layer(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, code_length):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )
        self.classifier = self.classifier[:-1]

        self.hash_layer = nn.Sequential(
            nn.Linear(4096, code_length),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.hash_layer(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
