import os
import types
import atexit
from time import perf_counter
from functools import partial
from dataclasses import fields, asdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
from transformers import LogitsProcessor
try:
    from vllm import LLM
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt as TokensPrompt
    SUPPORT_VLLM = True
except ImportError:
    SUPPORT_VLLM = False

from soulxpodcast.config import Config, SamplingParams


# ===================== RAS LogitsProcessor =====================

class RASLogitsProcessor(LogitsProcessor):
    """Repetition Aware Sampling (VALL-E 2) 的 LogitsProcessor 实现。

    替代已废弃的 ``custom_generate`` 回调方式。
    结合 repetition_penalty + RAS 检查:

    1. 对 scores 施加 repetition_penalty
    2. 找到最高概率候选 token
    3. 若该 token 在最近 ``win_size`` 个 token 中出现次数 >= ``win_size * tau_r``，
       则回退到原始 logits（撤销 repetition_penalty 效果）
    """

    def __init__(self, penalty: float, win_size: int = 25, tau_r: float = 0.2):
        self.penalty = penalty
        self.win_size = win_size
        self.tau_r = tau_r

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 保存原始 scores（pre-penalty）
        original_scores = scores.clone()

        # ---- 1. 施加 repetition penalty ----
        for i in range(input_ids.shape[0]):
            for token_id in set(input_ids[i].tolist()):
                if scores[i, token_id] < 0:
                    scores[i, token_id] *= self.penalty
                else:
                    scores[i, token_id] /= self.penalty

        # ---- 2. RAS 检查 ----
        probs = nn.functional.softmax(scores, dim=-1)
        top_candidates = torch.argmax(probs, dim=-1)

        for i in range(input_ids.shape[0]):
            recent = input_ids[i, -self.win_size:]
            rep_count = (recent == top_candidates[i]).sum().item() + 1  # +1 = 至少出现1次
            if rep_count >= self.win_size * self.tau_r:
                # 回退到 raw logits — 绕过 repetition penalty 的压制
                scores[i] = original_scores[i]

        return scores


# ===================== Stopping Criteria =====================

class _EosTokenCriteria(StoppingCriteria):
    """Compatibility shim for EosTokenCriteria removed in some transformers versions."""
    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.eos_token_id


# ===================== HF Engine =====================

class HFLLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        config.eos = config.hf_config.eos_token_id # speech eos token;
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map=self.device)
        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        prompt: list[str],
        sampling_param: SamplingParams,
        past_key_values=None,
    ) -> dict:

        stopping_criteria = StoppingCriteriaList([_EosTokenCriteria(eos_token_id=self.config.hf_config.eos_token_id)])

        # 构建 logits processors
        logits_processors = []

        if sampling_param.use_ras:
            # 使用 RASLogitsProcessor 同时处理 repetition_penalty + RAS
            ras_processor = RASLogitsProcessor(
                penalty=sampling_param.repetition_penalty,
                win_size=sampling_param.win_size,
                tau_r=sampling_param.tau_r,
            )
            logits_processors.append(ras_processor)
        else:
            # 无 RAS: 仅用标准的 RepetitionPenaltyLogitsProcessor
            from transformers import RepetitionPenaltyLogitsProcessor
            try:
                rep_pen_processor = RepetitionPenaltyLogitsProcessor(
                    penalty=sampling_param.repetition_penalty,
                    prompt_ignore_length=len(prompt)
                )
            except TypeError:
                rep_pen_processor = RepetitionPenaltyLogitsProcessor(
                    penalty=sampling_param.repetition_penalty
                )
            logits_processors.append(rep_pen_processor)

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
                logits_processor=logits_processors,
            )
            # 不再使用已废弃的 custom_generate — RAS 已通过 LogitsProcessor 实现
            generated_ids = self.model.generate(**gen_kwargs)
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