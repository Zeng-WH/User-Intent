# Multi-label Intent Classification

此代码包括multi-label意图识别分类的代码。BackBone是GPT-2.

预训练模型来自：https://huggingface.co/uer/gpt2-chinese-cluecorpussmall



## 训练模型：

```shell
bash train_intent_gpt_fgm.sh
```

需要修改数据路径等，在OOM的情况下需要灵活修改batch size以及learning rate，极端情况下可能需要修改train epoch.

目前主要调节的超参为对抗的扰动程度，即为`adver_eplsilon`

## 测试模型：

```
bash test_intent_gpt.sh
```

**注：**

1. 训练模型的过程中使用训练数据训练，验证集数据搜索最佳概率阈值。这个阈值得去log里看。

然后使用得到的阈值在测试集上测结果（切记！）

2. 在训练模型过程中，在`run_intent_gpt_search_fgm.py`中使用的是`compute_metrics_search`函数，而在测试的过程中使用的应是`compute_metrics`函数，同时修改该函数中`temp_gate`为验证集上的最佳概率阈值。(将在后续版本中优化该设置)

函数切换代码在：

```python
    trainer = Intent_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics_search,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
```









