import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule, Trainer


class LSTMModel(LightningModule):
    """LSTM sequence-to-sequence model for testing TBPTT with automatic optimization."""

    def __init__(self, truncated_bptt_steps=2, input_size=50, hidden_size=8):
        super().__init__()
        torch.manual_seed(42)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size * 2, 50)
        self.truncated_bptt_steps = truncated_bptt_steps
        self.automatic_optimization = True

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx, hiddens):
        x, y = batch
        if hiddens is not None:
            hiddens1 = hiddens
        else:
            hiddens1 = None
            hiddens2 = None
        pred1, hiddens1 = self.lstm(x, hiddens1)
        pred2, hiddens2 = self.lstm2(x, hiddens2)
        logits = torch.concat([pred1, pred2], dim=-1)
        linear = self.linear(logits)
        loss = F.mse_loss(linear, y)
        return {"loss": loss, "hiddens": hiddens1}

    def train_dataloader(self):
        dataset = TensorDataset(torch.rand(50, 2000, self.input_size), torch.rand(50, 2000, self.input_size))
        return DataLoader(dataset=dataset, batch_size=4)


model = LSTMModel(truncated_bptt_steps=100)
trainer = Trainer(
    default_root_dir="./",
    max_epochs=2,
    log_every_n_steps=2,
    enable_model_summary=False,
    enable_checkpointing=False,
)
trainer.fit(model)
