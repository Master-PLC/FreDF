from . import LSTM, MICN, TCN, Autoformer, Crossformer, DLinear, \
    ETSformer, FEDformer, FiLM, FreTS, Informer, Koopa, LightTS, \
    Nonstationary_Transformer, PatchTST, Pyraformer, Reformer, \
    TiDE, TimesNet, Transformer, iTransformer

MODEL_DICT = {
    'Autoformer': Autoformer,
    'Crossformer': Crossformer,
    'DLinear': DLinear,
    'ETSformer': ETSformer,
    'FEDformer': FEDformer,
    'FiLM': FiLM,
    'FreTS': FreTS,
    'Informer': Informer,
    'iTransformer': iTransformer,
    'Koopa': Koopa,
    'LightTS': LightTS,
    'LSTM': LSTM,
    'MICN': MICN,
    'Nonstationary_Transformer': Nonstationary_Transformer,
    'PatchTST': PatchTST,
    'Pyraformer': Pyraformer,
    'Reformer': Reformer,
    'TCN': TCN,
    'TiDE': TiDE,
    'TimesNet': TimesNet,
    'Transformer': Transformer,
}