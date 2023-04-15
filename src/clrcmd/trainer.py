from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from transformers import EvalPrediction, Trainer
from transformers.modeling_utils import unwrap_model
from transformers.utils import logging

logger = logging.get_logger(__name__)


def compute_metrics(x: EvalPrediction) -> Dict[str, float]:
    return {"spearman": spearmanr(x.predictions, x.label_ids).correlation}


class STSTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 将模型设为评估模式。
        model.eval()
        # 使用 with 语句和 torch.no_grad() 上下文管理器，确保在该上下文中计算的所有 torch.Tensor 对象都不会跟踪梯度。
        with torch.no_grad():
            # 将输入数据 inputs 中键为 inputs1 的值赋值给变量 inputs1，并使用 _prepare_inputs 方法对数据进行预处理。
            inputs1 = self._prepare_inputs(inputs["inputs1"])
            inputs2 = self._prepare_inputs(inputs["inputs2"])
            label = self._prepare_inputs(inputs["label"])
            # make sure the model is unwrapped from distributed modules
            # 执行前向传播计算，并返回评估结果。unwrap_model()函数将模型从分布式模块中解包，model()方法将输入数据输入到模型中，并返回评估结果。
            score = unwrap_model(model).model(inputs1, inputs2)
        # 将模型设置为训练模式，以便在下一次训练前再次计算梯度。
        model.train()
        return (None, score, label)
