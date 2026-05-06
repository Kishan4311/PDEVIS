# README — Apply ALD (Adaptive Layer-parallel Drafting) Patch

Apply the following changes in BOTH files:

```bash id="6j1ed4"
transformers/models/qwen3/modeling_qwen3.py
transformers/models/llama/modeling_llama.py
```

---

# 1. Add Import

## BEFORE

```python id="pz4h9l"
from .configuration_qwen3 import Qwen3Config
```

## AFTER

```python id="q2m7kh"
from .configuration_qwen3 import Qwen3Config
import torch.nn.functional as F
```

---

# 2. Add Reuse Variables in `__init__`

Inside `class Qwen3Model(...)` / `class LlamaModel(...)`

## BEFORE

```python id="hf5uvh"
self.padding_idx = config.pad_token_id
self.vocab_size = config.vocab_size
```

## AFTER

```python id="u3hqf5"
self.padding_idx = config.pad_token_id
self.vocab_size = config.vocab_size

# === ALD changes ===
self.enable_reuse = False
self._reuse_skip_accum = 0
self._reuse_token_count = 0
```

---

# 3. Add `pop_mean_skip()` Function

Inside model class (`Qwen3Model` / `LlamaModel`).

Find this section:

## BEFORE

```python id="h6pv2g"
# Initialize weights and apply final processing
self.post_init()

@check_model_inputs
@auto_docstring
def forward(
```

---

## AFTER

```python id="gql8rq"
# Initialize weights and apply final processing
self.post_init()

# === ALD changes ===
def pop_mean_skip(self):
    if self._reuse_token_count == 0:
        return 0.0

    mean = self._reuse_skip_accum / self._reuse_token_count

    self._reuse_skip_accum = 0
    self._reuse_token_count = 0

    return mean

@check_model_inputs
@auto_docstring
def forward(
```

---

# 4. Initialize Reuse Variables Before Decoder Loop

Find:

```python id="tcrr5e"
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

Add BELOW it.

## AFTER

```python id="4x39y7"
position_embeddings = self.rotary_emb(hidden_states, position_ids)

# === ALD changes ===
if self.enable_reuse:
    past_last_hidden = None
    last_idx = -1
    skip_count = 0
```

---

# 5. Add Reuse Logic Inside Decoder Loop

Find decoder loop:

## BEFORE

```python id="65kqec"
for decoder_layer in self.layers[: self.config.num_hidden_layers]:
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask_mapping[decoder_layer.attention_type],
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
```

---

## AFTER

```python id="79pqhm"
for decoder_layer in self.layers[: self.config.num_hidden_layers]:
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask_mapping[decoder_layer.attention_type],
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    # === ALD changes ===
    if self.enable_reuse:

        current_last = hidden_states[:, last_idx, :]

        if past_last_hidden is not None:

            past_last_hidden = past_last_hidden.to(
                device=current_last.device,
                dtype=current_last.dtype,
            )

            sim = F.cosine_similarity(
                current_last,
                past_last_hidden,
                dim=-1
            ).mean()

            threshold = 0.9

            if sim > threshold:
                skip_count += 1

                hidden_states = hidden_states.clone()
                hidden_states[:, last_idx, :] = past_last_hidden

        past_last_hidden = hidden_states[:, last_idx, :].clone()
```

---

# 6. Add Statistics Update After Decoder Loop

Find:

## BEFORE

```python id="e6u9st"
hidden_states = self.norm(hidden_states)
```

## AFTER

```python id="m2k8vp"
if self.enable_reuse:
    print(f"[Reuse skips] {skip_count}")

    self._reuse_skip_accum += skip_count
    self._reuse_token_count += 1

hidden_states = self.norm(hidden_states)
```

---

# 7. Enable Reuse in Drafter Only

Inside `appInference2.py`

Find:

## BEFORE

```python id="c6xj2f"
self.target.eval()

self.drafter.eval()
```

---

## AFTER

```python id="nd6d1u"
self.target.eval()
self.target.model.enable_reuse = False

self.drafter.eval()
self.drafter.model.enable_reuse = True
```

---

# 8. Run Project

Example:

```bash id="wwcvto"
python appInference.py \
    --target_model "llama-70b" \
    --drafter_model "llama-8b" \
    --device_target cuda:1 \
    --device_drafter cuda:0
```

---

# Modified Files

* `transformers/models/qwen3/modeling_qwen3.py`
* `transformers/models/llama/modeling_llama.py`

 
