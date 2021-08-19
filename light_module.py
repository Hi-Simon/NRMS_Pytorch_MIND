import os
import torch
import numpy as np

from model.TEST import TESTModel
from model.NRMS import NRMSModel
from data.DUMMY import DummyDataset
from data.MIND import MINDataset
from utils.metric import ndcg_score, mrr_score

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from torchmetrics.functional.classification.auroc import auroc
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT

class LTModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        # self.model = TESTModel()
        w2v = np.load(hparams['pretrained_model'])
        if hparams['model']['dct_size'] == 'auto':
            hparams['model']['dct_size'] = len(w2v)
        self.model = NRMSModel(self.hparams.model, torch.tensor(w2v, dtype=torch.float32))

    def prepare_data(self) -> None:
        """prepare_data

        load dataset
        """
        data_path = r'./data/'

        train_news_file = os.path.join(data_path, 'train', r'news.tsv')
        train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
        test_news_file = os.path.join(data_path, 'valid', r'news.tsv')
        test_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
        wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
        userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
        wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
        yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')
        # d = self.hparams1['data']
        train_dataset = MINDataset(self.hparams, train_news_file, train_behaviors_file, wordDict_file, userDict_file,
                                   npratio=self.hparams.data['npratio'], modify_news_id=self.hparams.modify_news_id,
                                   target_news_id=self.hparams.target_news_id, modify_index=self.hparams.modify_index,
                                   target_index=self.hparams.target_index, target_news_file=test_news_file,
                                   )
        train_dataset.load_data_from_file()
        train_length = int(0.9*len(train_dataset))
        split_length = [train_length, len(train_dataset)-train_length]
        self.train_dataset, self.val_dataset = random_split(train_dataset, split_length)

        self.test_dataset = MINDataset(self.hparams, test_news_file, test_behaviors_file, wordDict_file, userDict_file,
                                       target_news_id=self.hparams.target_news_id)
        self.test_dataset.load_data_from_file()



    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False)
        return val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False)
        return test_dataloader

    def training_step(self, batch, batch_idx):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        loss, score = self.model(*batch)
        logits = score.detach().cpu()
        labels = batch[0].detach().cpu().numpy()
        l = np.argmax(F.softmax(logits, dim=-1).numpy(), axis=1)
        acc = np.mean([labels==l])
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss, 'acc':acc}

    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx: int, dataloader_idx: int) -> None:
    #     loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
    #     acc_mean = np.stack([x['acc']for x in outputs]).mean()
    #     self.log('batch_acc', acc_mean)
    #     self.log('batch_loss', loss_mean)

    def validation_step(self, batch, batch_idx):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        labels, imp_indexes, user_indexes, candidate_title_index_batch, click_title_index_batch = batch
        _, logits = self.model(labels, imp_indexes, user_indexes, candidate_title_index_batch, click_title_index_batch)
        self.log('valid_loss', _)
        mrr = 0.
        auc = 0.
        ndcg5, ndcg10 = 0., 0.

        for score, label in zip(logits, labels):
            label = F.one_hot(label, len(score))
            auc += auroc(score, label)
            score = score.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            mrr += mrr_score(label, score)
            ndcg5 += ndcg_score(label, score, 5)
            ndcg10 += ndcg_score(label, score, 10)
        self.log('mrr',(mrr / logits.shape[0]).item(), prog_bar=True)
        return {'auroc': (auc / logits.shape[0]).item(), 'mrr': (mrr / logits.shape[0]).item(), 'ndcg5': (ndcg5 / logits.shape[0]).item(), 'ndcg10': (ndcg10 / logits.shape[0]).item()}

    def on_validation_batch_end( self, outputs, batch, batch_idx: int, dataloader_idx: int ) -> None:
        mrr = outputs['mrr']
        auroc = outputs['auroc']
        ndcg5 = outputs['ndcg5']
        ndcg10 = outputs['ndcg10']

        self.log('mrr', mrr)
        self.log('auroc', auroc)
        self.log('ndcg5', ndcg5)
        self.log('ndcg10', ndcg10)

    def test_step(self, batch, batch_idx):
        labels, imp_indexes, user_indexes, candidate_title_index_batch, click_title_index_batch = batch
        preds = self.model(None, imp_indexes, user_indexes, candidate_title_index_batch, click_title_index_batch)
        return preds,labels,imp_indexes

    def test_epoch_end(self, outputs):
        preds = []
        labels = []
        imp_indexes = []
        for step_preds, step_labels, step_imp_indexes in outputs:
            preds.extend(np.reshape(step_preds.detach().cpu().numpy(), -1))
            labels.extend(np.reshape(step_labels.detach().cpu().numpy(), -1))
            imp_indexes.extend(np.reshape(step_imp_indexes.detach().cpu().numpy(), -1))

        group_impr_indexes, group_labels, group_preds = self.group_labels(
            labels, preds, imp_indexes
        )
        mrr = []
        for label, pred in zip(group_labels, group_preds):
            mrr.append(mrr_score(label, pred))
        mrr = np.mean(mrr)
        print(mrr)
        self.log('mrr', mrr)
        # self.logger.log
        return {'progress_bar': mrr}

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            list, list, list:
            - Keys after group.
            - Labels after group.
            - Preds after group.

        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
