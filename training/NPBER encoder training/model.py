import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    DataCollatorForLanguageModeling
)

# Initialise model
class Bert(pl.LightningModule):

    def __init__(self, use_gpu, config):
        super().__init__()
        self.use_gpu = use_gpu
        self.config = config
        if self.use_gpu:
            self.bert = BertForMaskedLM(config=self.config).to('cuda:0')
        else:
            self.bert = BertForMaskedLM(config=self.config)

    def forward(self, input_ids, labels):
        return self.bert(input_ids=input_ids,labels=labels)

    def training_step(self, batch, batch_idx):
        if self.use_gpu:
            input_ids = batch["input_ids"].to('cuda:0')
            labels = batch["labels"].to('cuda:0')
        else:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
       
        outputs = self(input_ids=input_ids, labels=labels)
        # print(outputs.loss)
        return outputs

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)