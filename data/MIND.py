from torch.utils.data import Dataset
import pickle
import numpy as np
import csv
import random
import re
from tqdm.auto import tqdm


class MINDataset(Dataset):
    def __init__(self,
                 hparams,
                 news_file,
                 behaviors_file,
                 wordDict_file,
                 userDict_file,
                 col_spliter='\t',
                 ID_spliter='%',
                 npratio=-1,
                 modify_news_id=None,
                 target_news_id=None,
                 modify_index=None,
                 target_index=None,
                 target_news_file=None,

    ):
        self.modify_news_id=modify_news_id
        self.target_news_id=target_news_id
        self.modify_index=modify_index
        self.target_index=target_index
        self.target_news_file=target_news_file

        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['data']['title_size']
        self.his_size = hparams['data']['his_size']
        self.npratio = npratio

        self.word_dict = self.load_dict(wordDict_file)
        self.uid2index = self.load_dict(userDict_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_dict(self, file_path):
        """load pickle file

        Args:
            file path (str): file path

        Returns:
            object: pickle loaded object
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def word_tokenize(self, sent):
        """ Split sentence into word list using regex.
        Args:
            sent (str): Input sentence

        Return:
            list: word list
        """
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []

    def init_news(self, news_file):
        """init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """
        target_title_token = []
        if self.modify_news_id != None:
            with open(self.target_news_file, "r", encoding='utf-8') as f:
                rd = f.readlines()
                for line in rd:
                    nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(self.col_spliter)
                    if nid == self.target_news_id:
                        title=self.word_tokenize(title)
                        target_title_token.extend(title)
                        break


        self.nid2index = {}
        news_title = [""]

        with open(news_file, "r",  encoding='utf-8') as f:
            rd = f.readlines()
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(self.col_spliter)

                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                title = self.word_tokenize(title)
                if self.modify_news_id != None and nid == self.modify_news_id:
                    title[self.modify_index] = target_title_token[self.target_index]
                news_title.append(title)

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        for news_index in tqdm(range(len(news_title)), desc='init news'):
            title = news_title[news_index]
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index] in self.word_dict:
                    self.news_title_index[news_index, word_index] = self.word_dict[
                        title[word_index].lower()
                    ]

    def init_behaviors(self, behaviors_file):
        """init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []

        with open(behaviors_file, "r", encoding='utf-8') as f:
            rd = f.readlines()
            impr_index = 0
            for line in tqdm(rd, desc='init behaviors'):
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[: self.his_size]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def _convert_data(
            self,
            label_list,
            imp_indexes,
            user_indexes,
            candidate_title_indexes,
            click_title_indexes,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            click_title_indexes (list): words indices for user's clicked news titles.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        labels = np.asarray(label_list, dtype=np.int64)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int64
        )
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        return (
            labels,
            imp_indexes,
            user_indexes,
            candidate_title_index_batch,
            click_title_index_batch,
        )

    def newsample(self, news, ratio):
        """ Sample ratio samples from news list.
        If length of news is less than ratio, pad zeros.

        Args:
            news (list): input news list
            ratio (int): sample number

        Returns:
            list: output of sample list.
        """
        if ratio > len(news):
            return news + [0] * (ratio - len(news))
        else:
            return random.sample(news, ratio)

    def init_data(self, line):
        """Parse one behavior sample into feature values.
        if npratio is larger than 0, return negtive sampled result.

        Args:
            line (int): sample index.

        Yields:
            list: Parsed results including label, impression id , user id,
            candidate_title_index, clicked_title_index.
        """
        if self.npratio > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                candidate_title_index = []
                impr_index = []
                user_index = []
                # label = [1] + [0] * self.npratio
                label = 0

                n = self.newsample(negs, self.npratio)
                candidate_title_index = self.news_title_index[[p] + n]
                click_title_index = self.news_title_index[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                self.data.append(
                    self._convert_data(
                        label,
                        impr_index,
                        user_index,
                        candidate_title_index,
                        click_title_index,
                    )
                )


        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]
            if self.target_news_id != None and self.nid2index[self.target_news_id] not in impr:
                return


            for news, label in zip(impr, impr_label):
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [label]

                candidate_title_index.append(self.news_title_index[news])
                click_title_index = self.news_title_index[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])
                self.data.append(
                    self._convert_data(
                        label,
                        impr_index,
                        user_index,
                        candidate_title_index,
                        click_title_index,
                    )
                )


    def load_data_from_file(self):
        """Read and parse data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Yields:
            object: An iterator that yields parsed results, in the format of dict.
        """
        # 第i条新闻的第j个单词在词表中是第几个词
        if not hasattr(self, "news_title_index"):
            self.init_news(self.news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(self.behaviors_file)

        if not hasattr(self, "data"):
            self.data = []
            indexes = np.arange(len(self.labels))
            if self.npratio > 0:
                np.random.shuffle(indexes)
            for index in tqdm(indexes, desc='init data'):
                self.init_data(index)
