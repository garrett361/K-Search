# LLM API Usage in K-Search

## Overview

K-Search uses the OpenAI Python SDK with two API patterns:

1. **Responses API (default)** - enables reasoning computation (`reasoning_tokens > 0` in usage stats)
2. **Chat Completions API** - standard API without reasoning computation

## Responses API (Default)

Used by default for all models. Produces `reasoning_tokens > 0` in `response.usage.output_tokens_details`.

```python
response = self.client.responses.create(
    model=self.model_name,
    input=effective_prompt,
    reasoning={"effort": self.reasoning_effort}  # "low", "medium", or "high"
)
generated_code = response.output_text.strip()
```

**Disable**: `--no-reasoning-api` CLI flag or `use_reasoning_api=False` parameter.

## Chat Completions API

Used when reasoning API is disabled. Produces `reasoning_tokens == 0`.

```python
response = self.client.chat.completions.create(
    model=self.model_name,
    messages=[{"role": "user", "content": effective_prompt}]
)
generated_code = response.choices[0].message.content.strip()
```

## Usage

**CLI**:
```bash
# Reasoning API (default)
python generate_kernels_and_eval.py --model openai/gpt-oss-120b ...

# Disable reasoning API
python generate_kernels_and_eval.py --model openai/gpt-oss-120b --no-reasoning-api ...
```

**Python API**:
```python
# Default: reasoning enabled
gen = KernelGenerator(model_name="openai/gpt-oss-120b", ...)

# Disable reasoning
gen = KernelGenerator(model_name="openai/gpt-oss-120b", use_reasoning_api=False, ...)
```

## Custom Endpoints

K-Search sends structured messages via the OpenAI SDK and does not apply tokenizer templates.

```python
generator = KernelGenerator(
    model_name="your-model-name",
    base_url="http://your-endpoint:8000/v1",
    api_key="your-api-key"
)
```

**Tokenizer template handling**:
- For OpenAI models: API provider handles this internally
- For OSS models via litellm: **Unverified** - litellm should auto-detect from `tokenizer_config.json` but this has not been tested with K-Search
- If using a custom proxy, ensure it properly formats messages for your model

## Implementation

- `k_search/kernel_generators/kernel_generator.py:162` - API selection logic
- `k_search/kernel_generators/kernel_generator_world_model.py:103` - World model LLM calls
