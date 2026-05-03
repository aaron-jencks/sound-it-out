import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from sklearn.metrics import fbeta_score
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# Core Model Components

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

# -----------------------------------------------------------------------------
# Batched Attention

class CausalSelfAttentionBatched(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim).bfloat16())
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, ve: Tensor | None, sa_lambdas: Tensor):
        B, T, D = x.shape
        qkv = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        v = norm(v)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w[3])
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.proj_w = nn.Parameter(torch.zeros(dim, hdim).bfloat16())
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor):
        x = F.linear(x, self.fc_w)
        x = F.relu(x).square()
        x = F.linear(x, self.proj_w)
        return x

class BlockBatched(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttentionBatched(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor):
        x = (lambdas[0] * x + lambdas[1] * x0).type_as(x)
        if self.attn is not None:
            x = x + self.attn(x, ve, sa_lambdas)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# Batched GPT Model

class GPTBatched(nn.Module):
    """MEDIUM MODEL: 16 layers, 8 heads, 1024 dimensions"""
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, model_dim) for _ in range(3)
        ])
        self.blocks = nn.ModuleList([
            BlockBatched(model_dim, num_heads, max_seq_len, i)
            for i in range(num_layers)
        ])
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))

    @staticmethod
    def from_pretrained_gpt(original_gpt):
        model_dim = original_gpt.embed.embedding_dim
        vocab_size = original_gpt.embed.num_embeddings
        num_layers = len(original_gpt.blocks)
        first_attn = next(block.attn for block in original_gpt.blocks if block.attn is not None)
        num_heads = first_attn.num_heads
        max_seq_len = first_attn.rotary.cos.size(0)
        batched_model = GPTBatched(vocab_size, num_layers, num_heads, model_dim, max_seq_len)
        batched_model.embed.weight.data.copy_(original_gpt.embed.weight.data)
        for i, (old_ve, new_ve) in enumerate(zip(original_gpt.value_embeds, batched_model.value_embeds)):
            new_ve.weight.data.copy_(old_ve.weight.data)
        for i, (old_block, new_block) in enumerate(zip(original_gpt.blocks, batched_model.blocks)):
            if old_block.attn is not None and new_block.attn is not None:
                new_block.attn.qkvo_w.data.copy_(old_block.attn.qkvo_w.data)
            new_block.mlp.fc_w.data.copy_(old_block.mlp.fc_w.data)
            new_block.mlp.proj_w.data.copy_(old_block.mlp.proj_w.data)
        min_size = min(batched_model.scalars.size(0), original_gpt.scalars.size(0))
        batched_model.scalars.data[:min_size].copy_(original_gpt.scalars.data[:min_size])
        return batched_model

    def forward_features(self, input_seq: Tensor):
        B, T = input_seq.shape
        x = x0 = norm(self.embed(input_seq))
        ve_raw = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve_raw[0], ve_raw[1], ve_raw[2]] + [None] * (len(self.blocks) - 6) + [ve_raw[0], ve_raw[1], ve_raw[2]]
        assert len(ve) == len(self.blocks)
        skip_connections = []
        skip_map = {9: 6, 10: 4, 11: 2}
        skip_weights = self.scalars[:len(self.blocks)]
        lambdas = self.scalars[len(self.blocks):3*len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3*len(self.blocks):5*len(self.blocks)].view(-1, 2)
        for i, block in enumerate(self.blocks):
            if i in skip_map:
                x = (x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]).type_as(x)
            x = block(x, ve[i], x0, lambdas[i], sa_lambdas[i])
            skip_connections.append(x)
        x = norm(x)
        return x

# -----------------------------------------------------------------------------
# Dataset class — proper DataLoader pipeline

class XNLIDataset(Dataset):
    """
    PyTorch Dataset that reads pretokenized .bin files.
    Enables DataLoader with background workers, prefetching, and pin_memory.

    Binary file format:
      - int32: num_examples
      - int32[num_examples * context_window]: token ids
      - int64[num_examples]: labels
    """
    def __init__(self, bin_path: Path, context_window: int):
        super().__init__()
        with open(bin_path, 'rb') as f:
            num_examples = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            self.data = np.fromfile(
                f, dtype=np.int32, count=num_examples * context_window
            ).reshape(num_examples, context_window).copy()
            self.labels = np.fromfile(f, dtype=np.int64, count=num_examples).copy()
        print(f"Loaded {len(self.data):,} examples from {bin_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]).to(torch.int32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

# -----------------------------------------------------------------------------
# Classification Model Wrapper

class GPTClassification(nn.Module):
    """Wrapper around pretrained GPT for classification — MEDIUM MODEL
    MEDIUM MODEL: 16 layers, 8 heads, 1024 dimensions
    """
    def __init__(self, device, vocab_file, merges_file, data_dir, num_classes=2,
                 num_embed=1024, dropout=0.1, context_size=1024, batch_size=16, ipa=False,
                 text_column='sentence', label_column='label'):
        super().__init__()

        if context_size % 128 != 0:
            print(f"Warning: context_size {context_size} is not a multiple of 128. Rounding up.")
            context_size = ((context_size // 128) + 1) * 128

        self.device = device
        self.num_classes = num_classes
        self.context_window = context_size
        self.batch_size = batch_size
        self.dropout_rate = dropout

        self.text_column = text_column if isinstance(text_column, list) else [text_column]
        self.label_column = label_column

        # Training hyperparameters
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.grad_clip = 1.0
        self.warmup_iter_ratio = 0.1
        self.lr_decay_iter_ratio = 0.9
        self.min_lr = 1e-5

        self.tokenizer = Tokenizer.from_file(str(vocab_file))
        self.data_dir = Path(data_dir)
        self.train_data_path = self.data_dir / "train.bin"
        self.val_data_path = self.data_dir / "val.bin"

        self.pad_token_id = 0
        self.pretrained_model = None
        self.class_weights = None

        # Datasets — populated after prepare_if_needed()
        self.train_dataset = None
        self.val_dataset = None

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_embed, num_embed // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_embed // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        features = self.pretrained_model.forward_features(input_ids)  # [B, T, D]

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        batch_size = input_ids.shape[0]
        token_indices = torch.arange(input_ids.shape[1], device=input_ids.device)
        last_non_pad_indices = (token_indices * attention_mask).argmax(dim=1)
        pooled = features[torch.arange(batch_size, device=features.device), last_non_pad_indices]

        logits = self.classifier(pooled)

        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            return logits, loss
        return logits

    def prepare_if_needed(self, train_dataset, val_dataset, force_tokenization=False):
        """Tokenize if needed, then build XNLIDataset objects for DataLoader use"""
        if force_tokenization or not self.train_data_path.exists():
            print("Tokenizing training data...")
            self._tokenize_and_save(train_dataset, self.train_data_path)

        if force_tokenization or not self.val_data_path.exists():
            print("Tokenizing validation data...")
            self._tokenize_and_save(val_dataset, self.val_data_path)

        # Build Dataset objects (fast — just memory-maps the .bin files)
        self.train_dataset = XNLIDataset(self.train_data_path, self.context_window)
        self.val_dataset   = XNLIDataset(self.val_data_path,   self.context_window)

        self._compute_class_weights()

    def build_dataloaders(self, batch_size: int, num_workers: int = 4):
        """
        Build DataLoaders with background workers and prefetching.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=False,
        )
        return train_loader, val_loader

    def _compute_class_weights(self):
        labels = self.train_dataset.labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nClass distribution in training data:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count:,} samples ({100.0 * count / len(labels):.2f}%)")

        total_samples = len(labels)
        weights = np.zeros(self.num_classes, dtype=np.float32)
        for label, count in zip(unique_labels, counts):
            weights[label] = total_samples / (self.num_classes * count)

        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        print(f"\nComputed class weights:")
        for i, w in enumerate(weights):
            print(f"  Class {i}: {w:.4f}")
        print()

    def _tokenize_and_save(self, dataset, save_path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        all_input_ids = []
        all_labels = []

        for example in dataset:
            if len(self.text_column) == 1:
                text = str(example[self.text_column[0]])
            else:
                text = ' <ENDOFTEXT> '.join([str(example[col]) for col in self.text_column])

            encoded = self.tokenizer.encode(text)
            input_ids = encoded.ids[:self.context_window]
            if len(input_ids) < self.context_window:
                input_ids = input_ids + [0] * (self.context_window - len(input_ids))

            all_input_ids.append(input_ids)
            all_labels.append(example[self.label_column])

        input_ids_array = np.array(all_input_ids, dtype=np.int32)
        labels_array    = np.array(all_labels,    dtype=np.int64)

        with open(save_path, 'wb') as f:
            np.array([len(all_input_ids)], dtype=np.int32).tofile(f)
            input_ids_array.tofile(f)
            labels_array.tofile(f)

        print(f"Saved {len(all_input_ids):,} examples to {save_path}")
        print(f"Text columns used: {self.text_column}")

    def _calculate_f2(self, preds, labels, beta=2):
        preds_np  = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        try:
            return float(fbeta_score(labels_np, preds_np, beta=beta, average='macro', zero_division=0))
        except Exception:
            return 0.0

    def get_token_count(self):
        return len(self.train_dataset) * self.context_window

    def get_metadata(self):
        return {
            'vocab_size': self.tokenizer.get_vocab_size(),
            'max_length': self.context_window
        }

    def rebuild_classifier(self, model_dim):
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(model_dim // 2, self.num_classes),
        ).to(self.device)


# -----------------------------------------------------------------------------
# Helper classes to load training script checkpoints

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x: Tensor):
        return F.linear(x, self.weight)

class TrainingScriptAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = 128
        hdim = num_heads * 128
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim))
        self.rotary  = Rotary(128, max_seq_len)

class TrainingScriptMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w   = nn.Parameter(torch.empty(hdim, dim))
        self.proj_w = nn.Parameter(torch.empty(dim, hdim))

class TrainingScriptBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = TrainingScriptAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp  = TrainingScriptMLP(dim)

class GPTFlexAttention(nn.Module):
    """Matches ACTUAL checkpoint format — MEDIUM MODEL (16L, 8H, 1024D)"""
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed        = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks       = nn.ModuleList([
            TrainingScriptBlock(model_dim, num_heads, max_seq_len, i)
            for i in range(num_layers)
        ])
        self.lm_head_w = nn.Parameter(torch.empty(vocab_size, model_dim))
        self.scalars   = nn.Parameter(torch.zeros(5 * num_layers))
