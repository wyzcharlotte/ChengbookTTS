"""
诚书记 TTS 全功能交互式测试客户端
-----------------------------------
覆盖所有服务端功能：
  - 原生 API：非流式 / 流式（音色 + 情绪 + 语速）
  - OpenAI 兼容 API：/v1/audio/speech（格式切换）
  - 模型切换、自定义音色管理、Profile 管理
  - 服务端历史记录查询、Benchmark

用法:
    python -m chengbook_tts.client.interactive
    python -m chengbook_tts.client.interactive --url http://192.168.80.74:8080
"""
import io
import os
import sys
import time
import json
import argparse
from datetime import datetime

import requests
import numpy as np

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


# ============================================================
# 客户端
# ============================================================
class ChengbookTester:
    """诚书记 TTS 全功能测试客户端"""

    def __init__(self, base_url: str = 'http://127.0.0.1:8080'):
        self.base_url = base_url.rstrip('/')
        self.history = []          # 本地合成记录
        self.stream_mode = True    # 默认流式
        self.api_mode = 'native'   # 'native' | 'openai'
        self.voice = 'woman'
        self.emotion = 'calm'
        self.speed = 1.0
        self.oai_fmt = 'wav'       # OpenAI 输出格式
        self.oai_voice = 'alloy'   # OpenAI 语音名
        self.segment = True        # 分词+[]预处理

        # 缓存服务端信息
        self.voices: list[dict] = []
        self.emotions: list[dict] = []
        self.models: list[dict] = []
        self.active_model: str = ''
        self.sample_rate: int = 24000
        self.engine_name: str = '?'

        # 可用 OpenAI 语音名
        self._openai_voices = ['alloy', 'echo', 'fable', 'nova', 'onyx', 'shimmer']

    # ========== 查询接口 ==========

    def _get(self, path: str, timeout: int = 10) -> dict:
        resp = requests.get(f'{self.base_url}{path}', timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json_data: dict, timeout: int = 120) -> requests.Response:
        resp = requests.post(f'{self.base_url}{path}', json=json_data, timeout=timeout)
        resp.raise_for_status()
        return resp

    def _delete(self, path: str, timeout: int = 10) -> dict:
        resp = requests.delete(f'{self.base_url}{path}', timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        return self._get('/api/health')

    def fetch_voices(self) -> list[dict]:
        return self._get('/api/voices')['voices']

    def fetch_emotions(self) -> list[dict]:
        return self._get('/api/emotions')['emotions']

    def fetch_models(self) -> dict:
        """获取可用模型列表"""
        return self._get('/api/models')

    def switch_model(self, model_type: str) -> dict:
        """切换模型"""
        resp = self._post('/api/model/switch', {'model_type': model_type}, timeout=300)
        return resp.json()

    def get_profile(self) -> dict:
        return self._get('/api/profile')

    def update_profile(self, **kwargs) -> dict:
        resp = self._post('/api/profile', kwargs, timeout=10)
        return resp.json()

    def get_server_history(self, limit: int = 30) -> dict:
        """获取服务端合成历史"""
        return self._get(f'/api/history?limit={limit}')

    def clear_server_history(self) -> dict:
        return self._delete('/api/history')

    # ========== 自定义音色 ==========

    def upload_custom_voice(self, name: str, wav_path: str, description: str = '') -> dict:
        with open(wav_path, 'rb') as f:
            resp = requests.post(
                f'{self.base_url}/api/voices/custom',
                files={'file': (os.path.basename(wav_path), f, 'audio/wav')},
                data={'name': name, 'description': description},
                timeout=120,
            )
        resp.raise_for_status()
        result = resp.json()
        self.voices = self.fetch_voices()
        return result

    def delete_custom_voice(self, voice_id: str) -> dict:
        result = self._delete(f'/api/voices/custom/{voice_id}')
        if voice_id == self.voice:
            self.voice = 'woman'
        self.voices = self.fetch_voices()
        return result

    # ========== 原生非流式 ==========

    def synthesize(self, text: str) -> tuple[np.ndarray, int, dict]:
        t0 = time.time()
        resp = self._post('/api/tts', {
            'text': text, 'voice': self.voice, 'emotion': self.emotion,
            'speed': self.speed, 'segment': self.segment,
        })
        elapsed = time.time() - t0

        audio, sr = sf.read(io.BytesIO(resp.content), dtype='float32')
        duration = len(audio) / sr if sr > 0 else 0
        rtf = elapsed / duration if duration > 0 else 0

        record = {
            'time': time.strftime('%H:%M:%S'),
            'api': '原生非流式',
            'text': text,
            'voice': self.voice,
            'emotion': self.emotion,
            'speed': self.speed,
            'text_len': len(text),
            'duration': duration,
            'elapsed': elapsed,
            'rtf': rtf,
            'first_packet': None,
            'chunks': None,
            'format': 'wav',
        }
        self.history.append(record)
        return audio, sr, record

    # ========== 原生流式 ==========

    def synthesize_stream(self, text: str) -> tuple[np.ndarray, int, dict]:
        t0 = time.time()
        resp = requests.post(
            f'{self.base_url}/api/tts/stream',
            json={'text': text, 'voice': self.voice, 'emotion': self.emotion,
                  'segment': self.segment},
            timeout=120, stream=True,
        )
        resp.raise_for_status()

        sr = int(resp.headers.get('X-Audio-SampleRate', self.sample_rate))
        t_first = None
        chunks = []

        for raw in resp.iter_content(chunk_size=None):
            if not raw:
                continue
            if t_first is None:
                t_first = time.time()
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            chunks.append(pcm)

        elapsed = time.time() - t0
        first_packet = t_first - t0 if t_first else 0
        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        duration = len(audio) / sr if sr > 0 else 0
        rtf = elapsed / duration if duration > 0 else 0

        record = {
            'time': time.strftime('%H:%M:%S'),
            'api': '原生流式',
            'text': text,
            'voice': self.voice,
            'emotion': self.emotion,
            'speed': 1.0,
            'text_len': len(text),
            'duration': duration,
            'elapsed': elapsed,
            'rtf': rtf,
            'first_packet': first_packet,
            'chunks': len(chunks),
            'format': 'pcm',
        }
        self.history.append(record)
        return audio, sr, record

    # ========== OpenAI 兼容 ==========

    def synthesize_openai(self, text: str) -> tuple[np.ndarray, int, dict]:
        t0 = time.time()
        resp = self._post('/v1/audio/speech', {
            'model': 'tts-1',
            'input': text,
            'voice': self.oai_voice,
            'response_format': 'wav',
        })
        elapsed = time.time() - t0

        audio, sr = sf.read(io.BytesIO(resp.content), dtype='float32')
        duration = len(audio) / sr if sr > 0 else 0
        rtf = elapsed / duration if duration > 0 else 0

        record = {
            'time': time.strftime('%H:%M:%S'),
            'api': 'OpenAI',
            'text': text,
            'voice': self.oai_voice,
            'emotion': resp.headers.get('X-Emotion', '-'),
            'speed': 1.0,
            'text_len': len(text),
            'duration': duration,
            'elapsed': elapsed,
            'rtf': rtf,
            'first_packet': None,
            'chunks': None,
            'format': self.oai_fmt,
        }
        self.history.append(record)
        return audio, sr, record

    # ========== 播放 ==========

    def play(self, audio: np.ndarray, sr: int):
        if len(audio) == 0:
            return
        if HAS_SD:
            sd.play(audio, sr)
            sd.wait()
            return
        # 降级：写临时文件
        if HAS_SF:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()
            sf.write(tmp.name, audio, sr)
            os.startfile(tmp.name)
            time.sleep(len(audio) / sr + 0.5)
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# ============================================================
# 显示
# ============================================================
def show_history(records: list[dict]):
    if not records:
        print('  暂无记录\n')
        return

    has_stream = any(r.get('first_packet') is not None for r in records)

    header = (f'\n  {"#":<3} {"时间":<9} {"API":<10} {"音色":<8} {"情绪":<6} '
              f'{"字数":>4} {"音频(s)":>7} {"耗时(s)":>7} {"RTF":>7}')
    if has_stream:
        header += f' {"首包(s)":>8} {"块数":>5}'
    print(header)
    print('  ' + '-' * (88 if has_stream else 70))

    for i, r in enumerate(records, 1):
        fp = f'{r["first_packet"]:.3f}' if r.get('first_packet') else '  -   '
        chunks = str(r['chunks']) if r.get('chunks') else '  -  '
        line = (f'  {i:<3} {r["time"]:<9} {r["api"]:<10} {r["voice"]:<8} '
                f'{r["emotion"]:<6} {r["text_len"]:>4} {r["duration"]:>6.2f}  '
                f'{r["elapsed"]:>6.2f}  {r["rtf"]:>6.3f}')
        if has_stream:
            line += f' {fp:>8} {chunks:>5}'
        print(line)

    if len(records) > 1:
        avg_rtf = sum(r['rtf'] for r in records) / len(records)
        avg_elapsed = sum(r['elapsed'] for r in records) / len(records)
        print(f'  {"平均":>13} {"":>10} {"":>8} {"":>6} {"":>4} {"":>7} {avg_elapsed:>6.2f}  {avg_rtf:>6.3f}')

    stream_records = [r for r in records if r.get('first_packet')]
    if stream_records:
        avg_fp = sum(r['first_packet'] for r in stream_records) / len(stream_records)
        print(f'  流式平均首包: {avg_fp:.3f}s ({int(avg_fp * 1000)}ms)')
    print()


def _render_server_history(data: dict):
    """渲染服务端合成历史记录的表格"""
    records = data.get('records', [])
    total = data.get('total', 0)
    if not records:
        print(f'  服务端暂无合成记录\n')
        return

    print(f'\n  📊 服务端合成记录（共 {total} 条，显示最近 {len(records)} 条）\n')
    print(f'  {"#":<5} {"时间":<9} {"模型":<24} {"音色":<8} {"情绪":<6} '
          f'{"语速":>5} {"音频(s)":>7} {"耗时(s)":>7} {"RTF":>7} {"状态":<6}')
    print('  ' + '-' * 95)

    for r in records:
        status = '✅' if r.get('success') else '❌'
        stream_tag = '⚡' if r.get('streaming') else ' '
        model_short = r.get('model', '')[:24]
        print(f'  {r.get("id","-"):<5} {r.get("timestamp",""):<9} {model_short:<24} '
              f'{r.get("voice_name",""):<8} {r.get("emotion_name",""):<6} '
              f'{r.get("speed",1):>4.1f}x {r.get("duration",0):>6.2f}  '
              f'{r.get("elapsed",0):>6.2f}{stream_tag} {r.get("rtf",0):>7.3f} '
              f'{status:<6}')
    print()


# ============================================================
# 命令处理
# ============================================================
def print_help():
    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  命令列表                                                      │
  ├──────────────────────────────────────────────────────────────┤
  │  直接输入文本    → 合成并播放                                    │
  │  :q              → 退出                                        │
  │  :h              → 查看本地合成历史                              │
  │  :sh             → 查看服务端合成记录展板                          │
  │  :c              → 清空本地历史记录                               │
  │  :s              → 切换 流式/非流式                                │
  │  :seg            → 切换 分词[]预处理                               │
  │  :m              → 切换 API 模式 (native / openai)                │
  │  :v              → 切换音色（轮询）                                │
  │  :e              → 切换情绪（轮询）                                │
  │  :speed <值>     → 设置语速 (0.5~2.0)                             │
  │  :oav            → 切换 OpenAI 语音名（轮询）                       │
  │  :fmt            → 切换 OpenAI 输出格式                            │
  │  :info           → 显示当前配置                                    │
  │  :list           → 列出所有可用音色/情绪/模型                        │
  │  :models         → 列出可用模型及状态                               │
  │  :model <type>   → 切换 TTS 模型                                  │
  │  :upload <wav> <名称> → 上传自定义音色                             │
  │  :delvoice <id>  → 删除自定义音色                                 │
  │  :profile        → 查看当前 Profile                               │
  │  :b              → 跑 Benchmark                                  │
  │  :help           → 显示此帮助                                    │
  │                                                                  │
  │  快捷前缀: w:文本=女客户  m:文本=男客户                              │
  │  快捷后缀:  -angry -happy -sad -impatient -confused               │
  │           -soft -loud                                             │
  └──────────────────────────────────────────────────────────────┘
""")


def print_info(tts: ChengbookTester):
    model_info = ''
    if tts.active_model:
        model_info = f'\n  │  当前模型:    {tts.active_model:<28} │'
    print(f"""
  ┌──────────────────────────────────────────┐
  │  当前配置                                  │{model_info}
  │  API 模式:   {tts.api_mode:<30} │
  │  流式:       {'✅ 开启' if tts.stream_mode else '⬛ 关闭':<30} │
  │  分词[]:     {'✅ 开启' if tts.segment else '⬛ 关闭':<30} │
  │  音色:       {tts.voice:<30} │
  │  情绪:       {tts.emotion:<30} │
  │  语速:       {tts.speed:<30} │""")
    if tts.api_mode == 'openai':
        print(f"""  │  OpenAI 语音: {tts.oai_voice:<28} │
  │  输出格式:    {tts.oai_fmt:<30} │""")
    print("""  └──────────────────────────────────────────┘""")


# ============================================================
# Benchmark
# ============================================================
def run_benchmark(tts: ChengbookTester):
    tests = [
        ('你好，我想查一下订单。', 'woman', 'calm', '短句-平和'),
        ('太好了，终于有货了，谢谢你们！', 'woman', 'happy', '短句-开心'),
        ('你们怎么回事！我等了一个月了，今天必须给我说法！', 'man', 'angry', '中句-愤怒'),
        ('快点行不行，我还有别的事呢，别磨叽了！', 'woman', 'impatient', '中句-急躁'),
        ('非常抱歉给您带来了不便，您反馈的问题我们已经记录并提交给了相关部门，会在24小时内给您一个明确的处理结果，请您保持电话畅通。', 'woman', 'calm', '长句-平和'),
    ]

    print('\n  ' + '=' * 60)
    print(f'  Running Benchmark — 模型: {tts.active_model or tts.engine_name}')
    print('  ' + '=' * 60)

    orig_voice, orig_emotion = tts.voice, tts.emotion
    orig_api, orig_stream = tts.api_mode, tts.stream_mode

    rows_ns = []
    rows_stream = []

    for text, voice, emotion, label in tests:
        tts.voice, tts.emotion = voice, emotion

        # 非流式
        tts.api_mode = 'native'
        tts.stream_mode = False
        print(f'  [{label}] {voice}/{emotion} {len(text)}字 非流式...', end=' ', flush=True)
        _, _, r1 = tts.synthesize(text)
        rows_ns.append(r1)
        print(f'RTF={r1["rtf"]:.3f}')

        # 流式
        tts.stream_mode = True
        print(f'  [{label}] {voice}/{emotion} {len(text)}字 流式...', end=' ', flush=True)
        _, _, r2 = tts.synthesize_stream(text)
        rows_stream.append(r2)
        fp_ms = int(r2.get('first_packet', 0) * 1000)
        print(f'首包={fp_ms}ms RTF={r2["rtf"]:.3f}')

    # 恢复
    tts.voice, tts.emotion = orig_voice, orig_emotion
    tts.api_mode, tts.stream_mode = orig_api, orig_stream

    # 生成报告
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'bench_chengbook_{ts}.md'

    lines = [
        f'# 诚书记 TTS Benchmark',
        f'',
        f'**时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'**引擎**: {tts.engine_name}  |  **采样率**: {tts.sample_rate} Hz',
        f'**活跃模型**: {tts.active_model}',
        f'**音色**: {len(tts.voices)}  |  **情绪**: {len(tts.emotions)}',
        f'',
        f'## 非流式结果',
        f'',
        f'| 测试 | 音色 | 情绪 | 字数 | 音频(s) | 耗时(s) | RTF |',
        f'|------|------|------|-----:|--------:|--------:|----:|',
    ]
    for r in rows_ns:
        lines.append(f'| {r["text"][:15]} | {r["voice"]} | {r["emotion"]} | {r["text_len"]} | {r["duration"]:.2f} | {r["elapsed"]:.2f} | {r["rtf"]:.4f} |')
    ns_avg_rtf = sum(r['rtf'] for r in rows_ns) / len(rows_ns)
    lines.append(f'| **平均** | | | | | | **{ns_avg_rtf:.4f}** |')
    lines.append(f'')

    lines += [
        f'## 流式结果',
        f'',
        f'| 测试 | 音色 | 情绪 | 字数 | 首包(s) | 首包(ms) | 耗时(s) | RTF | 块数 |',
        f'|------|------|------|-----:|--------:|---------:|--------:|----:|-----:|',
    ]
    for r in rows_stream:
        lines.append(f'| {r["text"][:15]} | {r["voice"]} | {r["emotion"]} | {r["text_len"]} | {r["first_packet"]:.3f} | {int(r["first_packet"]*1000)} | {r["elapsed"]:.2f} | {r["rtf"]:.4f} | {r["chunks"]} |')
    s_avg_fp = sum(r['first_packet'] for r in rows_stream) / len(rows_stream)
    s_avg_rtf = sum(r['rtf'] for r in rows_stream) / len(rows_stream)
    lines.append(f'| **平均** | | | | **{s_avg_fp:.3f}** | **{int(s_avg_fp*1000)}** | | **{s_avg_rtf:.4f}** | |')
    lines.append(f'')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # 终端汇总
    print(f'\n  {"=" * 75}')
    print(f'  Benchmark 汇总')
    print(f'  {"=" * 75}')
    print(f'  {"测试":<14} {"非流RTF":>8}  {"流式首包":>9} {"流式RTF":>8} {"流式块数":>7}')
    print(f'  {"-"*14} {"-"*8}  {"-"*9} {"-"*8} {"-"*7}')
    for i in range(len(tests)):
        rn = rows_ns[i]
        rs = rows_stream[i]
        fp_s = f'{rs["first_packet"]:.3f}s'
        print(f'  {tests[i][3]:<14} {rn["rtf"]:>7.4f}  {fp_s:>8}  {rs["rtf"]:>7.4f} {rs["chunks"]:>6}')
    print(f'  {"-"*14} {"-"*8}  {"-"*9} {"-"*8} {"-"*7}')
    print(f'  {"平均":<14} {ns_avg_rtf:>7.4f}  {s_avg_fp:.3f}s {"":>4} {s_avg_rtf:>7.4f}')
    print(f'\n  📄 报告: {report_path}\n')


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='诚书记 TTS 全功能交互式测试客户端')
    parser.add_argument('--url', default='http://127.0.0.1:8080', help='服务地址')
    args = parser.parse_args()

    tts = ChengbookTester(args.url)

    # ---- 连接 ----
    print('=' * 60)
    print('  诚书记 TTS 全功能交互式测试客户端')
    print('  诚太乙 AI 教练 — 模拟客户语音训练真人坐席')
    print('=' * 60)

    try:
        health = tts.health()
        tts.engine_name = health.get('engine', '?')
        tts.sample_rate = health.get('sample_rate', 24000)
        tts.voices = tts.fetch_voices()
        tts.emotions = tts.fetch_emotions()

        print(f'\n✅ 已连接: {health["service"]}')
        print(f'   引擎: {tts.engine_name}  |  采样率: {tts.sample_rate}Hz')
        print(f'   音色: {[v["id"] for v in tts.voices]}')
        print(f'   情绪: {[e["id"] for e in tts.emotions]}')
        print(f'   能力: {json.dumps(health.get("capabilities", {}), ensure_ascii=False)}')

        # 模型列表
        try:
            models_data = tts.fetch_models()
            tts.models = models_data.get('models', [])
            tts.active_model = models_data.get('active', '')
            loaded = [m['type'] for m in tts.models if m.get('loaded')]
            all_models = [m['type'] for m in tts.models]
            print(f'   模型: {all_models}  |  当前: {tts.active_model}')
        except Exception:
            print(f'   ⚠️ 模型列表不可用')

        # OpenAI
        try:
            resp = requests.get(f'{tts.base_url}/v1/models', timeout=5)
            model_ids = [m['id'] for m in resp.json().get('data', [])]
            print(f'   OpenAI 模型: {model_ids}')
        except Exception:
            print(f'   ⚠️ OpenAI API 不可用')

        if not HAS_SD:
            print(f'   ⚠️ sounddevice 未安装，播放降级为文件模式')
    except Exception as e:
        print(f'\n❌ 连接失败: {e}')
        print(f'   请先启动: python -m chengbook_tts.cli serve --port 8080')
        return

    # ---- 交互循环 ----
    print_info(tts)
    print('  输入 :help 查看命令列表')
    print('-' * 60)

    while True:
        try:
            user_input = input('\n✏️  ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n👋 再见!')
            break

        if not user_input:
            continue

        # ---- 命令 ----
        if user_input == ':q':
            break

        elif user_input == ':help':
            print_help()
            continue

        elif user_input == ':h':
            show_history(tts.history)
            continue

        elif user_input == ':sh':
            try:
                data = tts.get_server_history(30)
                _render_server_history(data)
            except Exception as e:
                print(f'  ❌ 获取服务端历史失败: {e}')
            continue

        elif user_input == ':c':
            tts.history.clear()
            print('  本地记录已清空')
            continue

        elif user_input == ':s':
            tts.stream_mode = not tts.stream_mode
            print(f'  → 流式: {"✅ 开" if tts.stream_mode else "⬛ 关"}')
            continue

        elif user_input == ':seg':
            tts.segment = not tts.segment
            print(f'  → 分词[]: {"✅ 开" if tts.segment else "⬛ 关"}')
            continue

        elif user_input == ':m':
            tts.api_mode = 'openai' if tts.api_mode == 'native' else 'native'
            print(f'  → API 模式: {tts.api_mode}')
            continue

        elif user_input == ':v':
            if not tts.voices:
                tts.voices = tts.fetch_voices()
            ids = [v['id'] for v in tts.voices]
            if ids:
                idx = ids.index(tts.voice) if tts.voice in ids else 0
                tts.voice = ids[(idx + 1) % len(ids)]
                print(f'  → 音色: {tts.voice}')
            continue

        elif user_input == ':e':
            if not tts.emotions:
                tts.emotions = tts.fetch_emotions()
            ids = [e['id'] for e in tts.emotions]
            if ids:
                idx = ids.index(tts.emotion) if tts.emotion in ids else 0
                tts.emotion = ids[(idx + 1) % len(ids)]
                print(f'  → 情绪: {tts.emotion}')
            continue

        elif user_input.startswith(':speed'):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    tts.speed = float(parts[1])
                    tts.speed = max(0.5, min(2.0, tts.speed))
                except ValueError:
                    pass
            else:
                tts.speed = 1.5 if tts.speed == 1.0 else (0.75 if tts.speed == 1.5 else 1.0)
            print(f'  → 语速: {tts.speed}')
            continue

        elif user_input == ':oav':
            if tts.oai_voice in tts._openai_voices:
                idx = tts._openai_voices.index(tts.oai_voice)
                tts.oai_voice = tts._openai_voices[(idx + 1) % len(tts._openai_voices)]
            else:
                tts.oai_voice = tts._openai_voices[0]
            print(f'  → OpenAI 语音: {tts.oai_voice}')
            continue

        elif user_input == ':fmt':
            formats = ['wav', 'mp3', 'pcm', 'flac']
            idx = formats.index(tts.oai_fmt) if tts.oai_fmt in formats else 0
            tts.oai_fmt = formats[(idx + 1) % len(formats)]
            print(f'  → 输出格式: {tts.oai_fmt}')
            continue

        elif user_input == ':info':
            print_info(tts)
            continue

        elif user_input == ':models':
            try:
                data = tts.fetch_models()
                tts.models = data.get('models', [])
                tts.active_model = data.get('active', '')
                print(f'\n  可用模型:')
                for m in tts.models:
                    loaded = '★ 当前' if m.get('loaded') else '  '
                    info_parts = []
                    if m.get('supports_streaming'):
                        info_parts.append('流式')
                    if m.get('supports_emotion'):
                        info_parts.append('情绪')
                    if m.get('supports_multi_speaker'):
                        info_parts.append('多人')
                    extras = ', '.join(info_parts)
                    path_ok = '✅' if m.get('model_dir') else '⚠️ 路径未配置'
                    print(f'    {loaded} {m["type"]:<16} {m.get("name",""):<20} '
                          f'sr={m.get("sample_rate",24000)} {path_ok} [{extras}]')
                print()
            except Exception as e:
                print(f'  ❌ 获取模型列表失败: {e}')
            continue

        elif user_input.startswith(':model '):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print('  用法: :model <cosyvoice3|soulxpodcast|indextts_v1|indextts_v2>')
                continue
            model_type = parts[1].strip()
            print(f'  ⏳ 正在切换到 {model_type}（可能需要数十秒）...')
            try:
                result = tts.switch_model(model_type)
                print(f'  ✅ {result.get("message", "切换成功")}')
                # 刷新缓存
                tts.voices = tts.fetch_voices()
                tts.emotions = tts.fetch_emotions()
                health = tts.health()
                tts.engine_name = health.get('engine', '?')
                tts.sample_rate = health.get('sample_rate', 24000)
                models_data = tts.fetch_models()
                tts.models = models_data.get('models', [])
                tts.active_model = models_data.get('active', '')
                # 重置音色/情绪到有效值
                if tts.voice not in [v['id'] for v in tts.voices]:
                    tts.voice = tts.voices[0]['id'] if tts.voices else 'woman'
                if tts.emotion not in [e['id'] for e in tts.emotions]:
                    tts.emotion = tts.emotions[0]['id'] if tts.emotions else 'calm'
                print_info(tts)
            except Exception as e:
                print(f'  ❌ 切换失败: {e}')
            continue

        elif user_input == ':profile':
            try:
                profile = tts.get_profile()
                print(f'\n  当前 Profile:')
                print(f'    voice={profile.get("voice")}  emotion={profile.get("emotion")}  '
                      f'speed={profile.get("speed")}  segment={profile.get("segment")}')
                print()
            except Exception as e:
                print(f'  ❌ {e}')
            continue

        elif user_input.startswith(':upload '):
            parts = user_input.split(maxsplit=3)
            if len(parts) < 3:
                print('  用法: :upload <wav路径> <名称>')
                continue
            wav_path = parts[1]
            voice_name = parts[2]
            if not os.path.exists(wav_path):
                print(f'  ❌ 文件不存在: {wav_path}')
                continue
            try:
                result = tts.upload_custom_voice(voice_name, wav_path)
                print(f'  ✅ {result["message"]}')
            except Exception as e:
                print(f'  ❌ 上传失败: {e}')
            continue

        elif user_input.startswith(':delvoice '):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print('  用法: :delvoice <voice_id>')
                continue
            voice_id = parts[1].strip()
            try:
                result = tts.delete_custom_voice(voice_id)
                print(f'  ✅ {result["message"]}')
            except Exception as e:
                print(f'  ❌ 删除失败: {e}')
            continue

        elif user_input == ':list':
            print(f'\n  音色:')
            for v in tts.voices:
                tag = ' [自定义]' if v.get('is_custom') else ' [预设]'
                print(f'    {v["id"]:<18} {v.get("name",""):<12} {v.get("description","")}{tag}')
            print(f'\n  情绪:')
            for e in tts.emotions:
                print(f'    {e["id"]:<12} {e.get("name",""):<8} {e.get("description","")}')
            print(f'\n  OpenAI 语音: alloy / echo / fable / nova / onyx / shimmer')
            print(f'  情绪后缀: -angry -happy -sad -impatient -confused -soft -loud\n')
            continue

        elif user_input == ':b':
            run_benchmark(tts)
            continue

        # ---- 快捷前缀 ----
        if user_input.startswith(('w:', 'W:')):
            tts.voice = 'woman'
            user_input = user_input[2:].strip()
        elif user_input.startswith(('m:', 'M:')):
            tts.voice = 'man'
            user_input = user_input[2:].strip()

        if not user_input:
            continue

        # ---- 快捷后缀 ----
        suffix_emotions = {
            '-angry': 'angry', '-happy': 'happy', '-sad': 'sad',
            '-impatient': 'impatient', '-confused': 'confused',
            '-soft': 'soft', '-loud': 'loud',
        }
        for suffix, emotion in suffix_emotions.items():
            if user_input.endswith(suffix):
                tts.emotion = emotion
                user_input = user_input[:-len(suffix)].strip()
                break

        if not user_input:
            continue

        # ---- 合成 ----
        try:
            if tts.api_mode == 'openai':
                _do_openai(tts, user_input)
            elif tts.stream_mode:
                _do_stream(tts, user_input)
            else:
                _do_normal(tts, user_input)
        except requests.exceptions.ConnectionError:
            print('\n  ❌ 连接断开，服务是否还在运行？')
        except Exception as e:
            print(f'\n  ❌ 错误: {e}')

    # 退出汇总
    if tts.history:
        print('\n' + '=' * 60)
        print('  本次测试汇总')
        print('=' * 60)
        show_history(tts.history)
    print('👋 再见!\n')


def _do_normal(tts: ChengbookTester, text: str):
    print(f'  ⏳ 非流式 [{tts.voice}/{tts.emotion}] speed={tts.speed} ...', end=' ', flush=True)
    audio, sr, r = tts.synthesize(text)
    print(f'✅ 音频:{r["duration"]:.2f}s 耗时:{r["elapsed"]:.2f}s RTF:{r["rtf"]:.3f}')
    print(f'  🔊 播放中...', end=' ', flush=True)
    tts.play(audio, sr)
    print('✓')


def _do_stream(tts: ChengbookTester, text: str):
    print(f'  ⚡ 流式 [{tts.voice}/{tts.emotion}] ...', flush=True)
    audio, sr, r = tts.synthesize_stream(text)
    fp = int(r.get('first_packet', 0) * 1000)
    print(f'  ✅ 首包:{fp}ms 共{r["chunks"]}块 音频:{r["duration"]:.2f}s '
          f'耗时:{r["elapsed"]:.2f}s RTF:{r["rtf"]:.3f}')
    if len(audio) > 0:
        print(f'  🔊 播放中...', end=' ', flush=True)
        tts.play(audio, sr)
        print('✓')


def _do_openai(tts: ChengbookTester, text: str):
    print(f'  🤖 OpenAI [{tts.oai_voice}] fmt={tts.oai_fmt} ...', end=' ', flush=True)
    audio, sr, r = tts.synthesize_openai(text)
    print(f'✅ 音频:{r["duration"]:.2f}s 耗时:{r["elapsed"]:.2f}s RTF:{r["rtf"]:.3f}')
    print(f'  🔊 播放中...', end=' ', flush=True)
    tts.play(audio, sr)
    print('✓')


if __name__ == '__main__':
    main()
