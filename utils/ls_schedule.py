

def step_lr_scheduler(param_lr, optimizer, iter_num, gamma, step, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (gamma ** (iter_num // step))

    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1

    return optimizer