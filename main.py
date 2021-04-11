
import os
import sys

from mmodel import get_basic_params, get_model

if __name__ == '__main__':
    model = 'ZZZ_model'
    os.environ['TARGET_MODEL'] = model

    params = get_basic_params()

    assert 'torch' not in sys.modules
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
    #print(params.gpu)
    #print()
    import torch
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(000000)
    torch.cuda.manual_seed_all(000000)

    model = get_model()
    #print("qwyeoqywoeyqwoeyqwoeyoyqwe")
    model.train_model()