import argparse
import json
import random

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sentsim.config import ModelArguments
from sentsim.data.sts import load_sts13
from sentsim.models.models import create_contrastive_learning
from transformers import AutoTokenizer

# 定义画热力图的函数
def plot_heatmap(s1, s2, data, score, fpath):
    l1, l2 = len(s1), len(s2)

    # 用 matplotlib 画图
    fig, ax = plt.subplots(figsize=(int(0.5 * len(s2)), int(0.4 * len(s1))), facecolor="white")

    # 将坐标轴倒转
    plt.gca().invert_yaxis()
    # 设置标题和标签
    ax.set_title(f"score: {score:.3f}")
    ax.set_yticks([y + 0.5 for y in range(l1)])
    ax.set_yticklabels(s1)
    ax.set_xticks([x + 0.5 for x in range(l2)])
    ax.set_xticklabels(s2, rotation=90)
    ax.xaxis.set_ticks_position("top")
    # 绘制热力图
    im = ax.pcolormesh(data, edgecolors="k", linewidths=1, cmap=plt.get_cmap("Blues"))

    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig(fpath, dpi=400)

# 定义命令行参数
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 第一个句子的路径
parser.add_argument("--sent1-path", type=str)
# 第二个句子的路径
parser.add_argument("--sent2-path", type=str)
# 模型参数的路径，超参
parser.add_argument("--model-args-path", type=str)
# 模型参数的路径
parser.add_argument("--ckpt-path", type=str)


def main():
    # 解析命令行参数
    args = parser.parse_args()
    # 加载 STS13 数据集
    sts13 = load_sts13("/nas/home/sh0416/data/STS/STS13-en-test/")

    # 加载模型参数和 tokenizer
    with open(args.model_args_path) as f:
        model_args = ModelArguments(**json.load(f))
    # 加载预训练模型的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # 创建对比学习模型
    module = create_contrastive_learning(model_args)
    # 加载模型权重
    module.load_state_dict(torch.load(args.ckpt_path))
    model = module.model
    # 设置模型为评估模式
    model.eval()
    # 初始化计数器
    step = 0
    with torch.no_grad():
        # 将所有的STS-13测试样本合并到一个列表中
        examples = [x for examples in sts13.values() for x in examples]
        # 随机抽取60个样本和最后30个样本，共计90个样本
        random.seed(0)
        examples = random.sample(examples, k=60) + sorted(examples, key=lambda x: x[1])[-30:]
        # 根据样本的相似度得分从小到大排序
        examples = sorted(examples, key=lambda x: x[1])
        # 仅保留得分最小和最大的30个样本
        # examples = examples[:30] + examples[-30:]
        # 遍历所有样本
        for (s1, s2), score in examples:
            # 将两个句子进行分词
            t1 = tokenizer.convert_ids_to_tokens(tokenizer(s1)["input_ids"])
            t2 = tokenizer.convert_ids_to_tokens(tokenizer(s2)["input_ids"])
            # 如果句子长度超过12，就跳过该样本
            if len(t1) > 12 or len(t2) > 12:
                continue
            # 将两个句子转换为PyTorch张量
            x1 = tokenizer(s1, padding=True, return_tensors="pt")
            x2 = tokenizer(s2, padding=True, return_tensors="pt")
            # 计算两个句子的注意力矩阵，并将其转换为NumPy数组
            heatmap = model.compute_heatmap(x1, x2)[0].numpy()
            # 将热力图可视化并保存到文件中
            plot_heatmap(t1, t2, heatmap, score, f"case_{step}_avg.png")
            # 更新计数器
            step += 1


if __name__ == "__main__":
    main()
