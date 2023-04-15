import argparse
import logging
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from clrcmd.data.dataset import STSBenchmarkDataset
from clrcmd.data.sts import load_sts_benchmark
from clrcmd.models import create_contrastive_learning, create_tokenizer

logger = logging.getLogger(__name__)

# 创建一个命令行参数解析器的类。
# formatter_class：用于设置帮助信息的格式。1
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# fmt: off
# add_argument：添加一个命令行参数。
parser.add_argument("--model", type=str, help="Model", default="bert-cls",
                    choices=["bert-cls", "bert-avg", "bert-rcmd", "roberta-cls", "roberta-avg", "roberta-rcmd"])
parser.add_argument("--checkpoint", type=str, help="Checkpoint path", default=None)
parser.add_argument("--data-dir", type=str, help="data dir", default="data")
# fmt: on


# 这段代码的作用是使用预训练的对比学习模型来评估句子相似性。
# 具体来说，该代码读取了一个已经训练好的对比学习模型和一个数据集，然后使用该模型在数据集上进行预测，计算模型的性能指标并将结果记录在一个日志文件中。
def main():
    # 设置随机数种子，确保结果可重复。
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # 创建一个名为"log"的文件夹，用于存储日志文件。
    os.makedirs("log", exist_ok=True)
    # 检查是否有GPU可用，如果有则使用GPU，否则使用CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 解析命令行参数。
    args = parser.parse_args()
    # 设置日志记录器的配置，包括日志级别、日志格式和日志文件名。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filename=f"log/evaluate-{args.model}.log",
    )
    # 将命令行参数记录在日志文件中。
    logger.info("** Command Line Arguments **")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Create tokenizer and model
    # 创建一个分词器(tokenizer)和对比学习模型(model)。
    tokenizer = create_tokenizer(args.model)
    model = create_contrastive_learning(args.model).to(device)

    # Load method
    # 加载已经训练好的对比学习模型的权重。
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint, "pytorch_model.bin")))
    model = model.model

    # Load dataset
    # 加载数据集。
    sources = load_sts_benchmark(args.data_dir)
    loaders = {
        name: {
            k: DataLoader(STSBenchmarkDataset(v, tokenizer), batch_size=32)
            for k, v in testset.items()
        }
        for name, testset in sources.items()
    }

    # Evaluate
    # 在数据集上进行预测并计算性能指标。
    result = {}
    model.eval()
    # 通过使用 torch.no_grad() 上下文管理器来禁用梯度计算，避免在评估时浪费计算资源；
    with torch.no_grad():
        # 遍历所有数据集，使用数据加载器加载数据，对每个数据集进行评估；
        for source_name, source in loaders.items():
            # 对于每个数据集，遍历其数据加载器，获取输入数据和标签数据，通过模型计算预测分数，并将分数和标签数据分别存储在 scores 和 labels 数组中；
            logger.info(f"Evaluate {source_name}")
            scores_all, labels_all = [], []
            for _, loader in source.items():
                scores, labels = [], []
                for examples in tqdm(loader, desc=f"Evaluate {source_name}"):
                    inputs1 = {k: v.to(device) for k, v in examples["inputs1"].items()}
                    inputs2 = {k: v.to(device) for k, v in examples["inputs2"].items()}
                    scores.append(model(inputs1, inputs2).cpu().numpy())
                    labels.append(examples["label"].numpy())
                # 将 scores 和 labels 数组连接成一个大数组，分别存储在 scores_all 和 labels_all 数组中；
                scores, labels = np.concatenate(scores), np.concatenate(labels)
                scores_all.append(scores)
                labels_all.append(labels)
            scores_all, labels_all = np.concatenate(scores_all), np.concatenate(labels_all)
            # 对 scores_all 和 labels_all 数组使用 spearmanr 函数计算它们的斯皮尔曼等级相关系数，将结果存储在 result 字典中。
            # spearmanr 函数计算的是预测分数和标签数据之间的相关性，用来评估模型预测的质量。
            result[source_name] = spearmanr(scores_all, labels_all)[0]

    # 将性能指标记录在日志文件中，并计算平均性能指标。
    logger.info("** Result **")
    for metric_name, metric_value in result.items():
        logger.info(f"{metric_name} = {metric_value:.4f}")
    score_avg = np.average(list(result.values()))
    logger.info(f"avg = {score_avg:.4f}")


if __name__ == "__main__":
    main()
