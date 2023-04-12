import argparse
import logging
import os
import uuid

from transformers import TrainingArguments, set_seed


'''
导入需要用到的自定义模块，包括
数据集处理（ContrastiveLearningCollator、NLIContrastiveLearningDataset、STSBenchmarkDataset）、
模型创建（create_contrastive_learning、create_tokenizer）、
训练器（STSTrainer）
评估指标（compute_metrics）
'''
from clrcmd.data.dataset import (
    ContrastiveLearningCollator,
    NLIContrastiveLearningDataset,
    STSBenchmarkDataset,
)
from clrcmd.data.sts import load_stsb_dev
from clrcmd.models import create_contrastive_learning, create_tokenizer
from clrcmd.trainer import STSTrainer, compute_metrics

logger = logging.getLogger(__name__)

# 定义命令行参数parser，包括数据路径（data-dir）、模型名称（model）、输出路径（output-dir）、softmax温度（temp）、随机种子（seed）等；
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# fmt: off
parser.add_argument("--data-dir", type=str, help="Data directory", default="data")
parser.add_argument("--model", type=str, help="Model", default="bert-cls",
                    choices=["bert-cls", "bert-avg", "bert-rcmd", "roberta-cls", "roberta-avg", "roberta-rcmd"])
parser.add_argument("--output-dir", type=str, help="Output directory", default="ckpt")
parser.add_argument("--temp", type=float, help="Softmax temperature", default=0.05)
parser.add_argument("--seed", type=int, help="Seed", default=0)
# fmt: on


def main():
    # 解析命令行参数
    args = parser.parse_args()
    # 生成唯一的实验名称experiment_name
    experiment_name = f"{args.model}-{uuid.uuid4()[:6]}"
    # 指定训练过程的各种参数，包括batch size、learning rate、epoch数、logging和evaluation策略、保存模型等
    training_args = TrainingArguments(
        os.path.join(args.output_dir, experiment_name),
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=True,
        logging_strategy="steps",
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        metric_for_best_model="eval_spearman",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
    )
    # 如果是单机训练或第一个进程，设置logging的level、format和输出文件
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            filename=f"log/train-{experiment_name}.log",
        )
    # 输出各个超参数，以及进程的相关信息
    logger.info("Hyperparameters")
    for k, v in vars(args).items():
        logger.info(f"{k} = {v}")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16} "
    )

    # Set seed before initializing model.
    # 设置随机数种子
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    # 根据模型名称创建tokenizer和模型
    tokenizer = create_tokenizer(args.model)
    model = create_contrastive_learning(args.model, args.temp)
    model.train()

    # 加载训练集和评估集数据，其中训练集数据从csv文件中加载，而评估集数据则从文件夹中的文件中加载
    train_dataset = NLIContrastiveLearningDataset(
        os.path.join(args.data_dir, "nli_for_simcse.csv"), tokenizer
    )
    eval_dataset = STSBenchmarkDataset(
        load_stsb_dev(os.path.join(args.data_dir, "STS", "STSBenchmark"))["dev"], tokenizer
    )

    # 创建STSTrainer对象，指定训练参数、训练集和评估集数据、tokenizer、回调函数等
    trainer = STSTrainer(
        model=model,
        data_collator=ContrastiveLearningCollator(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbackes=[],
    )
    # 开始训练，并输出训练结果
    train_result = trainer.train()
    logger.info(train_result)
    # 保存最好的模型
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-best"))


if __name__ == "__main__":
    main()
