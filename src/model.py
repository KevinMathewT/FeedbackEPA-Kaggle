import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from config import config


class MultiDropout(nn.Module):
    def __init__(self, model_config):
        super(MultiDropout, self).__init__()
        if config['pooler'] == 'mean_max':
            self.fc = nn.Linear(2 * model_config.hidden_size, config["num_classes"])
        else:
            self.fc = nn.Linear(model_config.hidden_size, config["num_classes"])

        # Multi Dropouts
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

    def forward(self, out):
        logits1 = self.fc(self.dropout1(out))
        logits2 = self.fc(self.dropout2(out))
        logits3 = self.fc(self.dropout3(out))
        logits4 = self.fc(self.dropout4(out))
        logits5 = self.fc(self.dropout5(out))

        out = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

class MeanPooling(nn.Module):
    def __init__(self, model_config):
        super(MeanPooling, self).__init__()
        self.drop = nn.Dropout(p=0.1)
        self.fc = nn.Linear(model_config.hidden_size, config["num_classes"])
        if config['multi_drop']:
            self.md = MultiDropout(model_config=model_config)

    def forward(self, model_out, attention_mask):
        last_hidden_state = model_out.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        if not config['multi_drop']:
            out = self.drop(mean_embeddings)
            outputs = self.fc(out)
        if config['multi_drop']:
            outputs = self.md(mean_embeddings)

        return outputs


class MaxPooling(nn.Module):
    def __init__(self, model_config):
        super(MaxPooling, self).__init__()
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(model_config.hidden_size, config["num_classes"])
        if config['multi_drop']:
            self.md = MultiDropout(model_config=model_config)

    def forward(self, model_out, attention_mask):
        last_hidden_state = model_out.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        last_hidden_state[
            input_mask_expanded == 0
        ] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(last_hidden_state, 1)[0]

        if not config['multi_drop']:
            out = self.drop(max_embeddings)
            outputs = self.fc(out)
        if config['multi_drop']:
            outputs = self.md(max_embeddings)

        return outputs


class MeanMaxPooling(nn.Module):
    def __init__(self, model_config):
        super(MeanMaxPooling, self).__init__()
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(2 * model_config.hidden_size, config["num_classes"])
        if config['multi_drop']:
            self.md = MultiDropout(model_config=model_config)

    def forward(self, model_out, attention_mask):
        last_hidden_state = model_out.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Max Pooling
        last_hidden_state[
            input_mask_expanded == 0
        ] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(last_hidden_state, 1)[0]

        # Mean Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)

        if not config['multi_drop']:
            out = self.drop(mean_max_embeddings)
            outputs = self.fc(out)
        if config['multi_drop']:
            outputs = self.md(mean_max_embeddings)

        return outputs


class Conv1DPooling(nn.Module):
    def __init__(self, model_config):
        super(Conv1DPooling, self).__init__()
        self.cnn1 = nn.Conv1d(768, 256, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(256, config["num_classes"], kernel_size=2, padding=1)
        if config['multi_drop']:
            self.md = MultiDropout(model_config=model_config)

    def forward(self, model_out, attention_mask):
        last_hidden_state = model_out.last_hidden_state
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        cnn_embeddings = F.relu(self.cnn1(last_hidden_state))
        cnn_embeddings = self.cnn2(cnn_embeddings)
        logits, _ = torch.max(cnn_embeddings, 2)
        return logits


class AttentionPooling(nn.Module):
    def __init__(self, model_config):
        super(AttentionPooling, self).__init__()
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(model_config.hidden_size, config["num_classes"])
        self.attention = nn.Sequential(
            nn.Linear(768, 512), nn.Tanh(), nn.Linear(512, 1), nn.Softmax(dim=1)
        )
        self.regressor = nn.Sequential(nn.Linear(768, 3))
        if config['multi_drop']:
            self.md = MultiDropout(model_config=model_config)

    def forward(self, model_out, attention_mask):
        last_hidden_state = model_out.last_hidden_state
        weights = self.attention(last_hidden_state)
        context_vector = torch.sum(weights * last_hidden_state, dim=1)

        if not config['multi_drop']:
            outputs = self.regressor(context_vector)
        if config['multi_drop']:
            outputs = self.md(context_vector)

        return outputs


class DefaultPooling(nn.Module):
    def __init__(self, model_config):
        super(DefaultPooling, self).__init__()
        self.fc = nn.Linear(model_config.hidden_size, config["num_classes"])
        if config['multi_drop']:
            self.md = MultiDropout(model_config=model_config)

    def forward(self, model_out, attention_mask):
        pooler_output = model_out.pooler_output
        logits = self.fc(pooler_output)
        return logits


models_dict = {
    "max": MaxPooling,
    "mean": MeanPooling,
    "mean_max": MeanMaxPooling,
    "conv1d": Conv1DPooling,
    "attention": AttentionPooling,
    "default": DefaultPooling,
}

class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({"output_hidden_states": True})
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.pooler = models_dict[config['pooler']](model_config=self.config)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask)
        out = self.pooler(out, mask)
        return out



def get_model():
    return FeedBackModel(config["model_name"])
