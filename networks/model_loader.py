import networks.alexnet as alexnet
import networks.vgg16 as vgg16

def load_model(arch, code_length, pre_train=True):
    """
    Load cnn model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = alexnet.load_model(code_length, pre_train)
    elif arch == 'vgg16':
        model = vgg16.load_model(code_length, pre_train)
    else:
        raise ValueError('Invalid model name!')

    return model

