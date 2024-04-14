import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import pickle
import numpy as np
import streamlit as st


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def prepare_input(text):
    inputs = cls_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        truncation=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor([v], dtype=torch.long)
    return inputs


class CustomModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.config = AutoConfig.from_pretrained(name)
        self.fc = nn.Linear(self.config.hidden_size, 11)

    def feature_me5(self, inputs):
        outputs = self.model(**inputs)
        feature = average_pool(outputs.last_hidden_state,
                               inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature_me5(inputs)
        output = self.fc(feature)

        return output


@st.cache_data
def setup():
    t5_tokenizer = T5Tokenizer.from_pretrained("checkpoints/t5_model")
    t5_model = T5ForConditionalGeneration.from_pretrained(
        "checkpoints/t5_model").eval()

    cls_model = CustomModel("intfloat/multilingual-e5-large")
    state = torch.load(
        "checkpoints/cls_model/intfloat-multilingual-e5-large_fold0_best.pth",
        map_location=torch.device("cpu"),
    )
    cls_model.load_state_dict(state["model"])
    cls_model.eval()
    cls_tokenizer = AutoTokenizer.from_pretrained(
        "intfloat/multilingual-e5-large")

    return t5_tokenizer, t5_model, cls_model, cls_tokenizer


t5_tokenizer, t5_model, cls_model, cls_tokenizer = setup()
with open("checkpoints/cls_model/executor_le.pkl", "rb") as f:
    class_le = pickle.load(f)
en2ru = {
    "proxy": "Доверенность",
    "contract": "Договор",
    "act": "Акт",
    "application": "Заявление",
    "order": "Приказ",
    "invoice": "Счет",
    "bill": "Приложение",
    "arrangement": "Соглашение",
    "contract offer": "Договор оферты",
    "statute": "Устав",
    "determination": "Решение",
}


def clear_text(text):
    text = " ".join(re.findall(r"[а-яА-Я0-9 ёЁ\-\.,?!+a-zA-Z]+", text))
    return text


def get_summary(text):
    prefix = "Напиши заголовок для документа: "
    inputs = prefix + text
    inputs = t5_tokenizer(inputs, return_tensors="pt",
                          max_length=512, truncation=True)
    outputs = t5_model.generate(
        **inputs, max_new_tokens=150, repetition_penalty=4.0)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def get_class(text):
    inputs = prepare_input(text)
    pred = cls_model(inputs)
    label = np.argmax(pred.detach())
    label = class_le.inverse_transform([label])[0]

    return en2ru[label]


def get_predicts(text):
    text = clear_text(text)
    class_pred = get_class(text)
    summary_pred = get_summary(text)

    return class_pred, summary_pred
