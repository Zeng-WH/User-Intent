#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from textwrap import indent
from typing import Optional
from unittest import result
from torch import nn

import datasets
from datasets import utils
import numpy as np
from datasets import load_dataset, load_metric
import torch
import copy
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.file_utils import is_apex_available
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
#from model.modeling_bert_prompt import BertPromptForSequenceClassification
from transformers.trainer_utils import get_last_checkpoint
from transformers import  BertTokenizer
#from transformers import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification
from model.modeling_bert import BertForSequenceClassification
from metric.accuracy import compute_accuracy
from metric.matthews_correlation import compute_matthews_correlation
#from transformers.utils import check_min_version, send_example_telemetry
#from transformers.utils.versions import require_version
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import math
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
import json
if is_apex_available():
    from apex import amp
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.21.0.dev0")

#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


user_intent_set = [
    "求助-查询",
    "求助-故障",
    "提供信息",
    "投诉反馈",
    "取消",
    "询问",
    "请求重复",
    "主动确认",
    "被动确认",
    "否认",
    "问候",
    "再见",
    "客套",
    "其他"
]
intent2id = {
    "求助-查询" : 0,
    "求助-故障" : 1,
    "提供信息" : 2,
    "投诉反馈": 3,
    "取消": 4,
    "询问": 5,
    "请求重复": 6,
    "主动确认": 7,
    "被动确认": 8,
    "否认": 9,
    "问候": 10,
    "再见": 11,
    "客套": 12,
    "其他": 13,
}
id2intent = {
    0:"求助-查询",
    1:"求助-故障",
    2:"提供信息",
    3:"投诉反馈",
    4:"取消",
    5:"询问",
    6:"请求重复",
    7:"主动确认",
    8:"被动确认",
    9:"否认",
    10:"问候",
    11:"再见",
    12:"客套",
    13:"其他",
}
user_intent_dic = {
    "求助-查询" : 0,
    "求助-故障" : 1,
    "提供信息" : 2,
    "投诉反馈": 3,
    "取消": 4,
    "询问": 5,
    "请求重复": 6,
    "主动确认": 7,
    "被动确认": 8,
    "否认": 9,
    "问候": 10,
    "再见": 11,
    "客套": 12,
    "其他": 13,
}

class FGM():
    def __init__(self, model, eplsilon):
        self.eplsilon = eplsilon
        self.model = model
        self.backup = {}
    
    def attack(self, epsilon=1., emb_name='wte'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.eplsilon* np.random.random()*param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='wte'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def logits2label(output, probe_gate):
    preds = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    preds_si = torch.sigmoid(torch.tensor(preds))
    #print(preds_si.shape)

    temp_gate = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    #mask_si = preds_si.ge(probe_gate)

    mask_si = preds_si.ge(temp_gate)
    #print(mask_si[0:5])
    #print(preds_si[0:5])

    index_list = []
    for index, mask_item in enumerate(mask_si):
        mask_v = torch.masked_select(preds_si[index], mask_item)
        index_temp = []
        for item in mask_v:
            temp1 = (preds_si[index] == item).nonzero(as_tuple=True)[0]    
            index_temp.extend(temp1.tolist())
        if len(index_temp) == 0:
            index_temp.append(int(np.argmax(preds_si[index])))
        index_list.append(index_temp)
    p_label_list = []
    for index_i in index_list:
        temp_label = []
        for item in index_i:
            temp_label.append(id2intent[item])
        temp_label = ",".join(temp_label)
        p_label_list.append(temp_label)

    true_label_ids = output.label_ids
    true_label_list = []
    for item in true_label_ids:
        temp_label = []
        temp2 = (torch.tensor(item) == 1.).nonzero(as_tuple=True)[0] 
        temp2 = temp2.tolist()
        for i in temp2:
            temp_label.append(id2intent[i])
        temp_label = ",".join(temp_label)
        true_label_list.append(temp_label)
    
    return p_label_list, true_label_list



class Intent_Trainer(Trainer):
    def init_fgm(self, model, eplsilon):
        self.fgm = FGM(model, eplsilon)
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)


        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
            self.fgm.attack() #在prompt上添加扰动
        with self.autocast_smart_context_manager():
            loss_adv = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss_adv = loss_adv.mean()
        loss_adv.backward()
        self.fgm.restore() #恢复embedding

        return loss.detach()

    def evaluate_result(
        self,
        eval_dataset: Optional[Dataset] = None,
        probe_gate: Optional[float] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        p_label, true_label = logits2label(output, probe_gate)

        preds_si = torch.sigmoid(torch.tensor(output.predictions))
        #print(preds_si[0:20])
        #print(output.predictions.shape)
        #output.predictions = preds_si
        return output, p_label, true_label 


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    pre_seq_len: Optional[int] = field(
        default=200,
        metadata={
            "help": "length of prefix"

        }
    )
    prefix_drop: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "prefix dropout rate"

        }
    )
    prob_gate: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "prob gate"

        }
    )
    adver_eplsilon: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "prefix dropout rate"

        }
    )

def get_intent_dict(intent):
    intent_dict = {}
    for act in intent.split(','):
        intent_dict[act]=None
    intent_dict2=copy.deepcopy(intent_dict)
    for key, entry in intent_dict2.items():
        if isinstance(entry, list):
            new_entry=[]
            for item in entry:
                if isinstance(item, list):
                    new_entry.extend(item)
                else:
                    new_entry.append(item)
            intent_dict2[key]=new_entry
    return intent_dict, intent_dict2

def eval_user_intent(pred_l, true_l):
    tp_u, fp_u, fn_u = 0, 0, 0
    for pred_i, true_i in zip(pred_l, true_l):
        gen_ui, _=get_intent_dict(pred_i.replace('求助-查询', '询问'))
        true_ui, true_ui2=get_intent_dict(true_i.replace('求助-查询', '询问'))
        for ui, info in gen_ui.items():
            if info is None:
                if ui in true_ui:
                    tp_u+=1
                else:
                    fp_u+=1
            elif isinstance(info, list):
                if true_ui.get(ui, None) is None:
                    fp_u+=1
                else:
                    flag=1
                    for e in info:
                        if e not in true_ui2[ui]:
                            flag=0
                            break
                    if flag:
                        tp_u+=1
                    else:
                        fp_u+=1
            else:
                logging.info('Unknown format of user intent')
        for ui, info in true_ui.items():
            if info is None:
                if ui not in gen_ui:
                    fn_u+=1
            elif isinstance(info, list):
                if gen_ui.get(ui, None) is None:
                    fn_u+=1
                else:
                    flag=0
                    for e in info:
                        if isinstance(e, list):
                            if not any([t in gen_ui[ui] for t in e]):
                                flag=1
                                break
                        else:
                            if e not in gen_ui[ui]:
                                flag=1
                                break
                    if flag:
                        fn_u+=1
    P_u = tp_u/(tp_u+fp_u)
    R_u = tp_u/(tp_u+fn_u)
    F1_u = 2*P_u*R_u/(P_u+R_u)
    eval_result = {
        'P for user intent': P_u,
        'R for user intent': R_u,
        'F1 for user intent': F1_u,
    }
    return eval_result


def main():
    os.environ["WANDB_DISABLED"] = "true"
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    #send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels

    '''
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    '''
    num_labels = len(user_intent_set)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = GPT2ForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    config.pad_token_id = 0

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    '''
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    '''
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        #print(result)

        # Map labels to IDs (not necessary for GLUE tasks)
        #if label_to_id is not None and "label" in examples:

        if "label" in examples:
            label_list = []
            for label_item in examples["label"]:
                label_item = label_item.split(",")
                temp_label = [0 for u_i in user_intent_set]
                for item in label_item:
                    temp_label[intent2id[item]] = 1
                label_list.append(temp_label)

            result["label"] = label_list
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    '''
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")
    '''


    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        #preds = np.squeeze(preds)
        preds_si = torch.sigmoid(torch.tensor(preds))

        #temp_gate = torch.tensor([0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.5, 0.5, 0.5, 0.3])
        #temp_gate = torch.tensor([0.1000, 0.3200, 0.4300, 0.0900, 0.4500, 0.3500, 0.3000, 0.4500, 0.3800,
        #0.6600, 0.2100, 0.2700, 0.2700, 0.4000])
        #temp_gate = torch.tensor([0.0500, 0.7300, 0.2900, 0.1200, 0.4300, 0.3300, 0.7700, 0.5800, 0.4200,
        #0.5000, 0.4100, 0.2600, 0.5000, 0.3500])
        temp_gate = torch.tensor([0.0900, 0.5600, 0.3700, 0.1600, 0.5000, 0.3000, 0.5000, 0.5100, 0.3900,
        0.5900, 0.3500, 0.3000, 0.5000, 0.3200])
        #mask_si = preds_si.ge(model_args.prob_gate)
        #print(preds_si[0:5])
        mask_si = preds_si.ge(temp_gate)
        #print(mask_si[0:5])

        index_list = []
        for index, mask_item in enumerate(mask_si):
            mask_v = torch.masked_select(preds_si[index], mask_item)
            index_temp = []
            for item in mask_v:
                temp1 = (preds_si[index] == item).nonzero(as_tuple=True)[0]    
                index_temp.extend(temp1.tolist())
            if len(index_temp) == 0:
                index_temp.append(int(np.argmax(preds_si[index])))
            index_list.append(index_temp)
        p_label_list = []
        for index_i in index_list:
            temp_label = []
            for item in index_i:
                temp_label.append(id2intent[item])
            temp_label = ",".join(temp_label)
            p_label_list.append(temp_label)

        true_label_ids = p.label_ids
        true_label_list = []
        for item in true_label_ids:
            temp_label = []
            temp2 = (torch.tensor(item) == 1.).nonzero(as_tuple=True)[0] 
            temp2 = temp2.tolist()
            for i in temp2:
                temp_label.append(id2intent[i])
            temp_label = ",".join(temp_label)
            true_label_list.append(temp_label)

        eval_result = eval_user_intent(p_label_list, true_label_list)
        #print(eval_result)
        return eval_result
    
    def cal_gate(p, preds_si, temp_gate):
        mask_si = preds_si.ge(temp_gate)
        index_list = []
        for index, mask_item in enumerate(mask_si):
            mask_v = torch.masked_select(preds_si[index], mask_item)
            index_temp = []
            for item in mask_v:
                temp1 = (preds_si[index] == item).nonzero(as_tuple=True)[0]    
                index_temp.extend(temp1.tolist())
            if len(index_temp) == 0:
                index_temp.append(int(np.argmax(preds_si[index])))
            index_list.append(index_temp)        
        p_label_list = []
        for index_i in index_list:
            temp_label = []
            for item in index_i:
                temp_label.append(id2intent[item])
            temp_label = ",".join(temp_label)
            p_label_list.append(temp_label)
        true_label_ids = p.label_ids
        true_label_list = []
        for item in true_label_ids:
            temp_label = []
            temp2 = (torch.tensor(item) == 1.).nonzero(as_tuple=True)[0] 
            temp2 = temp2.tolist()
            for i in temp2:
                temp_label.append(id2intent[i])
            temp_label = ",".join(temp_label)
            true_label_list.append(temp_label)

        eval_result = eval_user_intent(p_label_list, true_label_list)
        return eval_result

    def compute_metrics_search(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds_si = torch.sigmoid(torch.tensor(preds))


        gate_init = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


        best_F1 = 0
        best_gate = copy.deepcopy(gate_init)
        for gate_index, gate_value in enumerate(gate_init):
            #print(gate_index)
            gate_range = torch.arange(0, 1, 0.01)
            for item in gate_range:
                #print(item)
                gate_init[gate_index] = item
                #print(gate_init)
                #print(temp_eval_result['F1 for user intent'])
                temp_eval_result = cal_gate(p, preds_si, gate_init)
                #print(temp_eval_result['F1 for user intent'])
                if temp_eval_result['F1 for user intent'] > best_F1:
                    print(temp_eval_result['F1 for user intent'])
                    best_F1 = copy.deepcopy(temp_eval_result['F1 for user intent'])
                    best_gate = copy.deepcopy(gate_init)
            gate_init = copy.deepcopy(best_gate)
        print(best_gate)

        eval_result = cal_gate(p, preds_si, best_gate)

        print(eval_result)
        

        return eval_result



    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Intent_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.init_fgm(model, model_args.adver_eplsilon)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            #metrics = trainer.evaluate(eval_dataset=eval_dataset)

            outputs, p_label, true_label = trainer.evaluate_result(eval_dataset=eval_dataset, probe_gate=model_args.prob_gate)
            metrics = outputs.metrics

            predictions = torch.sigmoid(torch.tensor(outputs.predictions))
            label_ids = outputs.label_ids

            
            '''
            with open("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD2022-main/Track2/intent_class/data/datav1/dev.json", "r") as r:
                test_lines = r.readlines()
            test_sample = []
            for item in test_lines:
                test_sample.append(json.loads(item))
            for index, item in enumerate(test_sample):
                item["id_label"] = true_label[index]
                item["p_label"] = p_label[index]
                item["predictions"] = predictions[index].tolist()
                item["label_ids"] = label_ids[index].tolist()
            #print(test_sample[0:10])
            
            
            with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD/SereTOD2022-main/Track2/intent_class/checkpoint/gpt2-1/eval_case.json", "w") as w:
                for item in test_sample:
                    w.write(json.dumps(item))
                    w.write("\n")
            '''

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)
    is_regression = False
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    '''
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    '''


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()