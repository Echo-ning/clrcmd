import csv
import logging
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, default_data_collator

logger = logging.getLogger(__name__)


class STSBenchmarkDataset(Dataset):
    def __init__(
        self, examples: List[Tuple[Tuple[str, str], float]], tokenizer: PreTrainedTokenizerBase
    ):
        # 初始化方法，输入参数包括 examples 和 tokenizer
        self.examples = examples# examples 是一个包含 (text1, text2, score) 的列表
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]:
        # 获取第 idx 个样本
        (text1, text2), score = self.examples[idx]  # 获取文本对和分数
        text1 = self.tokenizer(
            text1, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )   # 对文本进行编码
        text2 = self.tokenizer(
            text2, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )   # 对文本进行编码
        text1 = {k: v[0] for k, v in text1.items()} # 获取编码后的文本对应的词嵌入
        text2 = {k: v[0] for k, v in text2.items()} # 获取编码后的文本对应的词嵌入
        return {"inputs1": text1, "inputs2": text2, "label": torch.tensor(score)}
        # 返回包含 inputs1、inputs2 和 label 的字典，其中 inputs1 和 inputs2 表示文本对应的词嵌入，label 表示文本对的相似度得分
    def __len__(self):
        # 获取数据集的大小
        return len(self.examples)


class NLIContrastiveLearningDataset(Dataset):
    '''
    定义一个名为NLIContrastiveLearningDataset的类，继承自torch.utils.data.Dataset
    '''
    # 定义构造函数，接收文件路径和tokenizer作为参数
    def __init__(self, filepath: str, tokenizer: PreTrainedTokenizerBase):
        # 以只读方式打开文件
        with open(filepath) as f:
            self.examples = [
                (row["sent0"], row["sent1"], row["hard_neg"]) for row in csv.DictReader(f)
            ]
        # 将传入的tokenizer赋值给self.tokenizer
        self.tokenizer = tokenizer

    # 定义__getitem__方法，用于从数据集中获取单个样本，返回一个包含3个字典类型的元组
    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        # 从self.examples中取出第idx个元素，将其中的文本和负例文本分别赋值给text1和text2
        # 这里的hard_neg指的是一个hard negative例子，即两个看似不相关的句子实际上有些相关性，需要通过对抗性学习方法进行训练
        text1, text2, text_neg = self.examples[idx]

        # 将text1、text2和text_neg分别通过tokenizer转换成模型所需要的格式
        text1 = self.tokenizer(
            text1, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
        )
        text2 = self.tokenizer(
            text2, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
        )
        text_neg = self.tokenizer(
            text_neg, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
        )

        # 从text1、text2和text_neg的返回结果中取出第一个元素（因为这里只处理一个样本）
        text1 = {k: v[0] for k, v in text1.items()}
        text2 = {k: v[0] for k, v in text2.items()}
        text_neg = {k: v[0] for k, v in text_neg.items()}
        # 将text1、text2和text_neg打包成一个字典，返回结果
        return {"inputs1": text1, "inputs2": text2, "inputs_neg": text_neg}

    # 定义__len__方法，用于获取数据集的长度
    def __len__(self) -> int:
        return len(self.examples)


class ContrastiveLearningCollator:
    '''
    对对比学习数据进行批处理
    '''
    # 这个类的构造函数定义了一个__call__方法，用于对传入的数据进行批处理。
    # 该方法接受一个列表，其中包含元组，每个元组都包含三个字典，每个字典代表一个样本，包含输入和标签数据。
    def __call__(
        self, features: List[Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        # 这里创建了一个空字典result，用于存储批处理后的结果。
        result = {}
        # 遍历样本中的第一个字典的所有键。
        for k in features[0]:
            # 如果键是inputs1，则将该键对应的值收集到一个列表中，并通过default_data_collator函数对其进行标准化和堆叠，然后将其存储在result字典中。
            if k == "inputs1":
                result["inputs1"] = default_data_collator([x["inputs1"] for x in features])
            elif k == "inputs2":
                result["inputs2"] = default_data_collator([x["inputs2"] for x in features])
            elif k == "inputs_neg":
                result["inputs_neg"] = default_data_collator([x["inputs_neg"] for x in features])
            elif k == "label":
                result["label"] = torch.stack([x["label"] for x in features])
        # 返回处理后的结果。
        # 由于inputs1、inputs2和inputs_neg是用相同的方式处理的，因此返回的结果是一个包含三个字典的元组。
        return result
