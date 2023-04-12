import argparse
import json
import logging
import os
from typing import Tuple

import torch
from transformers import AutoTokenizer

from clrcmd.evaluation.ists import inference, load_examples, preprocess, save
from clrcmd.models import create_contrastive_learning

# fmt: off
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-dir", type=str, required=True, help="data dir")
parser.add_argument("--source", type=str, required=True, choices=["images", "headlines", "answers-students"], help="source")
parser.add_argument("--checkpoint-dir", type=str, required=True, help="checkpoint directory")
# fmt: on


# 定义一个create_filepaths函数，用于创建文件路径。
def create_filepaths(data_dir: str, source: str) -> Tuple[str, str, str, str]:
    return (
        os.path.join(data_dir, f"STSint.testinput.{source}.sent1.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent2.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent1.chunk.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent2.chunk.txt"),
    )


def main():
    # 使用argparse库解析命令行参数，包括data-dir、source和checkpoint-dir等。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(message)s")
    args = parser.parse_args()

    # 使用load_examples函数加载数据，并通过preprocess函数进行预处理（例如，将原始文本转化为token表示等）。
    examples = load_examples(*create_filepaths(args.data_dir, args.source))
    logging.info(f"Loading iSTS example (source = {args.source})")
    logging.info(f"{examples[0] = }")

    # 读取了一个json文件，其中包含了一个模型的配置
    with open(os.path.join(args.checkpoint_dir, "model_args.json")) as f:
        model_args = json.load(f)
    logging.info("Load model configuration")
    logging.info(f"{model_args = }")

    # 通过AutoTokenizer类加载了预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained(model_args["huggingface_model_name"], use_fast=False)
    logging.info(f"Loading tokenizer (model = {model_args['huggingface_model_name']})")
    # 对数据进行了预处理
    examples = preprocess(tokenizer=tokenizer, examples=examples)
    logging.info("Preprocess examples (Tokenize examples)")
    logging.info(f"{examples[0] = }")

    # 创建了一个对比学习的模型
    module = create_contrastive_learning(
        model_name=model_args["model_name"], temp=model_args["temp"], dense_rwmd=False
    )
    # 如果在检查点目录下发现了"pytorch_model.bin"文件，则加载该文件中保存的模型参数
    if os.path.exists(os.path.join(args.checkpoint_dir, "pytorch_model.bin")):
        module.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "pytorch_model.bin")))
        logging.info("Load model")

    # 接着使用torch.device()方法设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 对数据进行了推断
    examples = inference(model=module.model, prep_examples=examples, device=device)
    logging.info("Perform inference")
    logging.info(f"{examples[0] = }")
    # 保存了推断结果
    outfile = f"{args.source}.wa" if args.checkpoint_dir else f"{args.source}.wa.untrained"
    save(examples, os.path.join(args.checkpoint_dir, outfile))
    logging.info("Complete saving examples")
    exit()


if __name__ == "__main__":
    main()
