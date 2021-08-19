hparams = {
    'batch_size': 32,
    'epochs': 8,
    'lr': 0.0001,
    'name': 'mind_news',
    'loss': 'cross_entropy_loss',
    'optimizer': 'adam',
    'version': 'v3',
    'description': 'NRMS lr=5e-4, with weight_decay',
    'pretrained_model': './data/utils/embedding.npy',
    'model': {
        'dct_size': 'auto',
        'nhead': 20,
        'embed_size': 300, #self_attn_size
        # 'self_attn_size': 400,
        'encoder_size': 300, #
        'v_size': 200
    },
    'data': {
        "title_size": 30,
        "his_size": 50,
        "data_format": 'news',
        "npratio": 4,
        'pos_k': 50,
        'neg_k': 4,
        'maxlen': 15
    }
}
