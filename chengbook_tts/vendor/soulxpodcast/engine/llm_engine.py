import os
import sys
import types
import atexit
from time import perf_counter
from functools import partial
from dataclasses import fields, asdict

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
from transformers import RepetitionPenaltyLogitsProcessor


def _detect_best_attn_implementation():
    """Detect the best available attention implementation for faster inference.

    Priority: flash_attention_2 > sdpa > None (default eager).
    Flash Attention 2: O(n) memory, 2-3x faster attention via kernel fusion (requires pip install flash-attn).
    SDPA: PyTorch 2.0+ built-in fused attention, ~1.5-2x faster than eager, no extra deps.
    """
    if not torch.cuda.is_available():
        return None
    # 1. Try flash_attention_2 first (best performance)
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    # 2. Fall back to SDPA (PyTorch >= 2.0 built-in)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        return "sdpa"
    return None
try:
    from vllm import LLM
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt as TokensPrompt
    SUPPORT_VLLM = True
except ImportError:
    SUPPORT_VLLM = False

from soulxpodcast.config import Config, SamplingParams
from soulxpodcast.models.modules.sampler import _ras_sample_hf_engine


class _EosTokenCriteria(StoppingCriteria):
    """Compatibility shim for EosTokenCriteria removed in some transformers versions."""
    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.eos_token_id


class HFLLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        config.eos = config.hf_config.eos_token_id # speech eos token;
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # --- Phase 1a: Auto-detect best attention backend ---
        attn_impl = _detect_best_attn_implementation()
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        if attn_impl:
            model_kwargs['attn_implementation'] = attn_impl
            print(f'[HFLLMEngine] Using attn_implementation={attn_impl}', flush=True)

        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        # --- Phase 1c: torch.compile for JIT kernel fusion ---
        # fullgraph=False + dynamic=True: safe for auto-regressive (KV cache grows each step)
        if self.device.startswith("cuda"):
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
                print('[HFLLMEngine] torch.compile enabled (fullgraph=False, dynamic=True)', flush=True)
            except Exception as e:
                print(f'[HFLLMEngine] torch.compile failed: {e}, continuing without', flush=True)

        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        prompt: list[str],
        sampling_param: SamplingParams,
        past_key_values=None,
    ) -> dict:

        stopping_criteria = StoppingCriteriaList([_EosTokenCriteria(eos_token_id=self.config.hf_config.eos_token_id)])
        if sampling_param.use_ras:
            sample_hf_engine_handler = partial(_ras_sample_hf_engine,
                    use_ras=sampling_param.use_ras,
                    win_size=sampling_param.win_size, tau_r=sampling_param.tau_r)
        else:
            sample_hf_engine_handler = None
        try:
            rep_pen_processor = RepetitionPenaltyLogitsProcessor(
                penalty=sampling_param.repetition_penalty,
                prompt_ignore_length=len(prompt)
            ) # exclude the input prompt, consistent with vLLM implementation;
        except TypeError:
            rep_pen_processor = RepetitionPenaltyLogitsProcessor(
                penalty=sampling_param.repetition_penalty
            )
        with torch.no_grad():
            input_len = len(prompt)
            gen_kwargs = dict(
                input_ids = torch.tensor([prompt], dtype=torch.int64).to(self.device),
                do_sample=True,
                top_k=sampling_param.top_k,
                top_p=sampling_param.top_p,
                min_new_tokens=sampling_param.min_tokens,
                max_new_tokens=sampling_param.max_tokens,
                temperature=sampling_param.temperature,
                stopping_criteria=stopping_criteria,
                past_key_values=past_key_values,
                use_cache=True,
                logits_processor=[rep_pen_processor]
            )
            # custom_generate (RAS sampler) bypasses standard HF sample/greedy.
            # transformers 5.x: _validate_model_kwargs may reject it as unrecognized
            # model_kwarg. Monkey-patch to filter it out before validation.
            if sample_hf_engine_handler is not None:
                gen_kwargs['custom_generate'] = sample_hf_engine_handler
                _orig_validate = getattr(self.model, '_validate_model_kwargs', lambda _: None)
                def _patched_validate(kwargs_dict):
                    kwargs_dict.pop('custom_generate', None)
                    _orig_validate(kwargs_dict)
                self.model._validate_model_kwargs = _patched_validate

            try:
                generated_ids = self.model.generate(**gen_kwargs)
            finally:
                if sample_hf_engine_handler is not None:
                    self.model._validate_model_kwargs = _orig_validate
            generated_ids = generated_ids[:, input_len:].cpu().numpy().tolist()[0]
        output = {
            "text": self.tokenizer.decode(generated_ids),
            "token_ids": generated_ids,
        }
        return output

class VLLMEngine:

    def __init__(self, model, **kwargs):

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = config.hf_config.eos_token_id # speech eos token;
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        os.environ["VLLM_USE_V1"] = "0"
        if SUPPORT_VLLM:
            self.model = LLM(model=model, enforce_eager=True, dtype="bfloat16", max_model_len=8192, enable_prefix_caching=True,)
        else:
            raise ImportError("Not Support VLLM now!!!")
        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        prompt: list[str],
        sampling_param: SamplingParams,
        past_key_values=None,
    ) -> dict:
        sampling_param.stop_token_ids = [self.config.hf_config.eos_token_id]
        with torch.no_grad():
            generated_ids = self.model.generate(
                TokensPrompt(prompt_token_ids=prompt),
                VllmSamplingParams(**asdict(sampling_param)),
                use_tqdm=False,
            )[0].outputs[0].token_ids
        output = {
            "text": self.tokenizer.decode(generated_ids),
            "token_ids": list(generated_ids),
        }
        return output
