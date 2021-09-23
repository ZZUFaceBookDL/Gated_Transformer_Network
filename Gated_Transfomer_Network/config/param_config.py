# @Time    : 2021/07/20 19:47
# @Author  : SY.M
# @FileName: param_config.py

from config.path_config import UCR, UEA, MTS


class Config:
    archive = UEA            # select archive
    assert archive in [UCR, UEA, MTS], 'Select archive Error!'
    k_fold = 5               # fold number of K_fold Cross Validation

    # Hyper-Parameters
    EPOCH = 200              # Epoch
    BATCH_SIZE = 16          # Batch Size
    LR = 1e-4                # Learning rate

    d_model = 512            # size of embedding output
    d_feature = 256            # channel-expand dimension for univariate TS
    d_hidden = 2048          # hidden layer size of position-feedforward
    q = 8                    # Query size of attention
    v = 8                    # Value size of attention
    h = 8                    # Head number of Multi-head attention
    N = 8                    # size of Both wise Encoder stack
    dropout = 0            # dropout
    optimizer_name = 'Adam'  # optimizer

    test_interval = 1        # interval of test or validate
    draw_key = 10            # draw result plot when EPOCH >= draw_key
