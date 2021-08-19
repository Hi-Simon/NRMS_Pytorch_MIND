import os
import pytorch_lightning as pl
from light_module import LTModel
from utils.config import hparams
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

def replace_test(modify_news_id=None,
                 target_news_id=None,
                 modify_index=None,
                 target_index=None,
                 ):
    hparams['modify_news_id'] = modify_news_id
    hparams['target_news_id'] = target_news_id
    hparams['modify_index'] = modify_index
    hparams['target_index'] = target_index

    pl.seed_everything(42, workers=True)
    checkpoint_callback = ModelCheckpoint(
        filename=f'lightning_logs/{hparams["name"]}/{hparams["version"]}/' + '{epoch}-{valid_loss:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='valid_loss',
        # monitor='auroc',
        mode='min',
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor='train_loss',
        # min_delta=0.001,
        patience=5000,
        strict=False,
        verbose=True,
        mode='min'
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    model = LTModel(hparams)

    # wandb_logger = WandbLogger(save_dir='lightning_logs', project='MIND', log_model='all', version=hparams['version'])
    # wandb_logger.watch(model)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=hparams['epochs'],
                         progress_bar_refresh_rate=1,
                         log_every_n_steps=1,
                         # resume_from_checkpoint='lightning_logs/version_62',
                         callbacks=[early_stop, checkpoint_callback],
                         # logger=wandb_logger,
                         )
    print(trainer.fit(model))
    metric_list = trainer.test(model)
    return metric_list[0]['mrr']
    # print(trainer.logger.experiment.)


if __name__ == "__main__":
    print(replace_test())