from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pig.models.base import Detector


class TransformerDetector(Detector):
    name = "M3_transformer"

    def __init__(self, model_dir: str | Path | None = None, max_length: int = 256, batch_size: int = 32, device: str | None = None):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.tokenizer = None
        if model_dir:
            self._load_dir(model_dir)

    def _load_dir(self, model_dir):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        meta = Path(model_dir) / "pig_meta.json"
        if meta.exists():
            self.threshold = json.loads(meta.read_text())["threshold"]

    @staticmethod
    def train(train_texts, train_labels, val_texts, val_labels, base: str, out_dir: str | Path,
              epochs: int = 4, lr: float = 5e-5, batch_size: int = 32, max_length: int = 256,
              weight_decay: float = 0.01, seed: int = 42):
        import torch
        from datasets import Dataset
        from sklearn.metrics import f1_score
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                                  EarlyStoppingCallback, Trainer, TrainingArguments)

        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)

        def enc(texts, labels):
            ds = Dataset.from_dict({"text": list(texts), "label": [int(x) for x in labels]})
            return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])

        y = np.asarray(train_labels)
        w = torch.tensor([len(y) / (2 * (y == 0).sum()), len(y) / (2 * (y == 1).sum())], dtype=torch.float)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                out = model(**inputs)
                loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w.to(out.logits.device))
                return (loss, out) if return_outputs else loss

        def metrics(p):
            return {"f1": f1_score(p.label_ids, p.predictions.argmax(-1))}

        args = TrainingArguments(
            output_dir=str(Path(out_dir) / "checkpoints"), num_train_epochs=epochs, learning_rate=lr,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
            weight_decay=weight_decay, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
            warmup_ratio=0.1, logging_steps=20, seed=seed, report_to="none",
            fp16=torch.cuda.is_available(),
        )
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=enc(train_texts, train_labels),
            eval_dataset=enc(val_texts, val_labels), data_collator=DataCollatorWithPadding(tok),
            compute_metrics=metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        trainer.save_model(str(out_dir))
        tok.save_pretrained(str(out_dir))
        return trainer.state.log_history

    def predict_proba(self, texts):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(list(texts[i : i + self.batch_size]), truncation=True,
                                       max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.model(**batch).logits, -1)[:, 1].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def save_threshold(self, model_dir) -> None:
        (Path(model_dir) / "pig_meta.json").write_text(json.dumps({"threshold": float(self.threshold)}))

    def __getstate__(self):
        raise TypeError("TransformerDetector is stored as a HF model directory, use TransformerDetector(model_dir)")
