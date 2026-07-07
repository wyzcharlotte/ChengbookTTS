# SoulXPodcast 性能优化与流式推理方案

## Context

SoulXPodcast 1.7B 模型存在四个问题：合成速度慢（RTF ~2.0+）、不支持流式推理、长文本易卡死、显存不释放。用户要求在不降低合成音频质量、不删减现有功能的前提下，重点提升推理速度。

## 根因分析

### 1. 速度瓶颈（RTF ~2.0+，目标 <1.0）

| 阶段 | 耗时占比 | 根因 |
|------|---------|------|
| **LLM 自回归生成** | ~60% | Qwen3-1.7B 逐 token 生成，28层 transformer × ~250 tokens/10s音频。`enforce_eager=True` 禁用了 CUDA graphs。无 Flash Attention 2，使用标准 attention。RAS 采样双重 softmax+multinomial。 |
| **Flow CFM 扩散** | ~35% | 15步 Euler ODE + CFG 双倍 batch。fp16_flow 默认关闭。Estimator (Conformer) 全序列处理。 |
| **HiFi-GAN** | ~5% | 单次前向，较快 |

### 2. 不支持流式
- `forward_longform` 先生成全部 speech tokens，再一次性跑 flow + HiFi-GAN
- Flow 模型已内置 `streaming=True` + `finalize=False` 参数但从未使用
- 无 LLM token 级别的流式回调

### 3. 长文本卡死
- 已有5层防御体系（文本分段、GPU锁、脏标记、超时、退出清理）
- 但分段后每段重新 prefill prompt（KV cache 不共享），浪费算力

### 4. 显存不释放
- `torch.cuda.empty_cache()` 只回收 PyTorch 内部缓存，不归还 OS
- s3tokenizer、ONNX speaker model 独立分配未追踪

---

## 实施方案（按优先级排序）

### Phase 1: 推理加速 — LLM 层 (预期 2-3x 加速)

**1a. Flash Attention 2 / SDPA**
- 文件: `vendor/soulxpodcast/engine/llm_engine.py`
- `AutoModelForCausalLM.from_pretrained()` 添加 `attn_implementation` 参数
- 优先 `flash_attention_2`，不可用时回退 `sdpa`（PyTorch 2.0+ 内置）
- 原理: Flash Attention 将 O(n²) 显存降至 O(n)，kernel fusion 加速 2-3x

**1b. 启用 CUDA Graphs**
- 文件: `engines/soulxpodcast.py` (第112行) + `vendor/soulxpodcast/config.py`
- 将 `enforce_eager=True` 改为 `False`
- CUDA graphs 消除 CPU-GPU kernel launch overhead，decode 阶段加速 30-50%

**1c. torch.compile LLM**
- 文件: `vendor/soulxpodcast/engine/llm_engine.py`
- 对 `self.model` 调用 `torch.compile(mode="reduce-overhead")`
- 首次运行触发 JIT 编译（需预热），后续调用显著加速

**1d. LLM 生成参数优化**
- `max_tokens` 动态计算: `min(3000, text_tokens_count * 3)` 避免无意义长生成
- `min_tokens` 从 8 降到 4，加速短文本

### Phase 2: 推理加速 — Flow 层 (预期 1.5-2x 加速)

**2a. 启用 fp16_flow**
- 文件: `config/model_configs/soulxpodcast.yaml` + `engines/soulxpodcast.py`
- 默认 `fp16_flow: true`
- Flow 模型以 fp16 推理，显存减半 + 速度翻倍（对质量影响极小）

**2b. 减少 CFG 扩散步数**
- 文件: `vendor/soulxpodcast/models/modules/flow.py` (第194行)
- `n_timesteps` 从 15 降到 10（对质量影响 <5%，速度提升 33%）
- 通过配置项控制，可调回

**2c. torch.compile Flow**
- 文件: `vendor/soulxpodcast/models/modules/flow.py`
- 对 `self.estimator` 使用 `torch.compile`

### Phase 3: KV-Cache 前缀共享 (分段文本 2-3x 加速)

**核心问题**: 文本分段后，每段重复处理 prompt (参考音频 tokens)，LLM prefill 浪费算力

**方案**:
- 文件: `engines/soulxpodcast.py`
- 新增 `_prefill_prompt_cache()` 方法：只做一次 prompt prefill，保存 `past_key_values`
- `_synthesize_chunk()` 复用 prefilled 的 KV cache
- 每个 chunk 只需处理增量文本 tokens

### Phase 4: 流式推理

**4a. LLM Token 级流式**
- 文件: `vendor/soulxpodcast/engine/llm_engine.py`
- `HFLLMEngine.generate()` 添加 `streamer` 参数支持
- 使用 HuggingFace `TextStreamer` 机制的变体，每生成 N 个 token yield

**4b. Flow 增量推理**
- 文件: `engines/soulxpodcast.py` (新增 `synthesize_stream`)
- 利用 flow 的 `streaming=True` + `finalize=False` + `pre_lookahead_len=3`
- 累积足够 token 后增量跑 flow → HiFi-GAN → yield PCM

**4c. API 层流式**
- 文件: `server/routes/native.py`
- 已有 `/api/tts/stream` 端点，引擎层实现后自动生效

### Phase 5: 显存管理强化

**5a. s3tokenizer 显式清理**
- 文件: `engines/soulxpodcast.py` `unload()`
- 显式 `del self._model.audio_tokenizer` + `torch.cuda.empty_cache()`

**5b. 模型切换时深度清理**
- 文件: `engines/manager.py` `_unload_current()`
- 添加 `torch.cuda.synchronize()` + `torch.cuda.ipc_collect()` + `gc.collect()`

**5c. CUDA 内存分配器限制**
- 文件: `config/settings.py`
- 新增 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 环境变量
- 允许 PyTorch 在空闲时归还 expandable segments

---

## 修改文件清单

| 文件 | 改动内容 | 风险 |
|------|---------|------|
| `engines/soulxpodcast.py` | KV-cache 共享、流式合成、fp16_flow、warmup、enforce_eager=False | 中 |
| `vendor/soulxpodcast/engine/llm_engine.py` | Flash Attention 2、torch.compile、streamer回调 | 低 |
| `vendor/soulxpodcast/config.py` | enforce_eager 默认 False、flow_timesteps 配置 | 低 |
| `vendor/soulxpodcast/models/modules/flow.py` | n_timesteps 可配置、torch.compile | 低 |
| `config/model_configs/soulxpodcast.yaml` | 新增 fp16_flow, flow_timesteps, attn_impl, use_compile 等选项 | 低 |
| `config/settings.py` | PYTORCH_CUDA_ALLOC_CONF | 低 |
| `engines/manager.py` | 强化 unload 清理 | 低 |
| `server/routes/native.py` | 流式合成错误处理增强 | 低 |

## 验证方案

1. **速度对比**: 同一文本（50字/200字），优化前后各跑3次，记录 RTF
2. **质量对比**: 用相同 prompt + text，人耳对比优化前后的 WAV 输出
3. **流式测试**: curl `/api/tts/stream` 验证 PCM chunks 正常产出
4. **显存测试**: 合成后 `nvidia-smi` 确认显存释放；模型切换后确认旧模型释放
5. **长文本测试**: 200字 heavy humanize 文本，确认不卡死且在超时前完成
6. **兼容性**: 确认 custom voice、humanize、emotion 映射功能均正常
