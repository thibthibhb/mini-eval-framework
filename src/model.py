import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F


class DistilBERTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = 2

        self.bert = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'predictions': torch.argmax(logits, dim=-1)
        }

    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['predictions']

    def predict_proba(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return F.softmax(outputs['logits'], dim=-1)


def create_model(config):
    model = DistilBERTClassifier(config)
    return model