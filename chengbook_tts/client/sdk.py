"""
诚书记 TTS 客户端
诚太乙 AI 教练 → 将模拟客户对话文本转为语音 → 训练真人坐席

用法:
    from chengbook_tts.client.sdk import ChengbookTTSClient

    tts = ChengbookTTSClient('http://127.0.0.1:8080')

    # 模拟客户正常咨询（非流式）
    audio, sr, meta = tts.synthesize('你好，我想查一下我的订单。', voice='woman', emotion='calm', speed=1.0)

    # 模拟客户愤怒投诉（流式）
    audio, sr, meta = tts.synthesize_stream('你们这是什么服务！我要投诉！', voice='man', emotion='angry')

    # 保存
    import soundfile as sf
    sf.write('customer_voice.wav', audio, sr)
"""
import io
import os
import json
import time
from typing import Optional, List, Dict, Any, Tuple

import requests
import numpy as np
import soundfile as sf


class ChengbookTTSClient:
    """诚书记 TTS API 客户端 — 诚太乙 AI 教练调用，生成模拟客户语音"""

    def __init__(self, base_url: str = 'http://127.0.0.1:8080'):
        self.base_url = base_url.rstrip('/')
        self._session = requests.Session()

    # ========== 查询接口 ==========

    def health(self) -> dict:
        """健康检查，返回服务状态与可用配置"""
        resp = self._session.get(f'{self.base_url}/api/health', timeout=5)
        resp.raise_for_status()
        return resp.json()

    def list_voices(self) -> List[dict]:
        """获取可用客户音色列表"""
        resp = self._session.get(f'{self.base_url}/api/voices', timeout=5)
        resp.raise_for_status()
        return resp.json()['voices']

    def list_emotions(self) -> List[dict]:
        """获取可用情绪列表"""
        resp = self._session.get(f'{self.base_url}/api/emotions', timeout=5)
        resp.raise_for_status()
        return resp.json()['emotions']

    def list_models(self) -> dict:
        """获取可用模型列表及当前激活模型"""
        resp = self._session.get(f'{self.base_url}/api/models', timeout=5)
        resp.raise_for_status()
        return resp.json()

    # ========== 模型切换 ==========

    def switch_model(self, model_type: str) -> dict:
        """
        切换到指定 TTS 模型。

        参数:
            model_type: 'cosyvoice3' | 'soulxpodcast'

        返回:
            {'success': True, 'model_type': '...', 'engine_name': '...', 'message': '...'}
        """
        resp = self._session.post(
            f'{self.base_url}/api/model/switch',
            json={'model_type': model_type},
            timeout=300,  # 模型加载可能需要较长时间
        )
        resp.raise_for_status()
        result = resp.json()
        print(f'  [Switch] {result["message"]}')
        return result

    # ========== 自定义音色管理 ==========

    def upload_custom_voice(self, name: str, wav_path: str, description: str = '') -> dict:
        """
        上传自定义音色（两个模型共享音色库）。

        参数:
            name: 音色名称
            wav_path: 本地 WAV 文件路径
            description: 音色描述（可选）

        返回:
            {'success': True, 'voice_id': 'custom_xxx', 'name': '...', 'message': '...'}
        """
        with open(wav_path, 'rb') as f:
            resp = self._session.post(
                f'{self.base_url}/api/voices/custom',
                files={'file': (os.path.basename(wav_path), f, 'audio/wav')},
                data={'name': name, 'description': description},
                timeout=120,
            )
        resp.raise_for_status()
        result = resp.json()
        print(f'  [Upload] {result["message"]}')
        return result

    def delete_custom_voice(self, voice_id: str) -> dict:
        """
        删除自定义音色。

        参数:
            voice_id: 音色 ID（如 custom_abc12345）

        返回:
            {'success': True, 'voice_id': '...', 'name': '...', 'message': '...'}
        """
        resp = self._session.delete(
            f'{self.base_url}/api/voices/custom/{voice_id}',
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f'  [Delete] {result["message"]}')
        return result

    # ========== Profile ==========

    def get_profile(self) -> dict:
        """获取当前配置（音色/情绪/语速偏好）"""
        resp = self._session.get(f'{self.base_url}/api/profile', timeout=5)
        resp.raise_for_status()
        return resp.json()

    def update_profile(self, **kwargs) -> dict:
        """更新配置"""
        resp = self._session.post(f'{self.base_url}/api/profile', json=kwargs, timeout=5)
        resp.raise_for_status()
        return resp.json()

    # ========== 合成接口 ==========

    def synthesize(self, text: str, voice: str = 'woman', emotion: str = 'calm',
                   speed: float = 1.0, segment: bool = True) -> Tuple[np.ndarray, int, dict]:
        """
        非流式合成 — 返回完整音频（模拟客户说一句话）

        参数:
            text: 客户对话文本
            voice: 客户音色 (woman/man 或自定义 custom_xxx)
            emotion: 客户情绪 (calm/happy/sad/angry/soft/loud/impatient/confused)
            speed: 语速 (0.5~2.0)
            segment: 是否启用分词+[]预处理

        返回:
            (audio_float32, sample_rate, meta_dict)
            meta_dict 包含: duration, elapsed, rtf, voice, emotion, speed
        """
        t0 = time.time()

        resp = self._session.post(
            f'{self.base_url}/api/tts',
            json={
                'text': text,
                'voice': voice,
                'emotion': emotion,
                'speed': speed,
                'segment': segment,
            },
            timeout=120,
        )
        resp.raise_for_status()

        audio, sr = sf.read(io.BytesIO(resp.content), dtype='float32')

        elapsed = time.time() - t0
        duration = len(audio) / sr
        rtf = elapsed / duration if duration > 0 else 0

        meta = {
            'duration': duration,
            'elapsed': elapsed,
            'rtf': rtf,
            'voice': resp.headers.get('X-Voice', voice),
            'emotion': resp.headers.get('X-Emotion', emotion),
            'speed': float(resp.headers.get('X-Speed', speed)),
        }

        print(f'  [TTS] voice={voice} | emotion={emotion} | speed={speed} | '
              f'text_len={len(text)} | duration={duration:.2f}s | elapsed={elapsed:.2f}s | RTF={rtf:.3f}')
        return audio, sr, meta

    def synthesize_stream(self, text: str, voice: str = 'woman',
                          emotion: str = 'calm', segment: bool = True) -> Tuple[np.ndarray, int, dict]:
        """
        流式合成 — 实时返回音频块（适合模拟客户实时对话）

        参数:
            text: 客户对话文本
            voice: 客户音色 (woman/man 或自定义 custom_xxx)
            emotion: 客户情绪
            segment: 是否启用分词+[]预处理
            (流式不支持 speed 参数，仅 CosyVoice3 支持)

        返回:
            (audio_float32, sample_rate, meta_dict)
        """
        t0 = time.time()

        resp = self._session.post(
            f'{self.base_url}/api/tts/stream',
            json={
                'text': text,
                'voice': voice,
                'emotion': emotion,
                'segment': segment,
            },
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        sr = int(resp.headers.get('X-Audio-SampleRate', 24000))
        t_first = None
        all_chunks = []

        for raw in resp.iter_content(chunk_size=None):
            if not raw:
                continue
            if t_first is None:
                t_first = time.time()

            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            all_chunks.append(pcm)

        elapsed = time.time() - t0
        first_packet = t_first - t0 if t_first else 0
        audio = np.concatenate(all_chunks) if all_chunks else np.array([])
        duration = len(audio) / sr
        rtf = elapsed / duration if duration > 0 else 0

        meta = {
            'duration': duration,
            'elapsed': elapsed,
            'rtf': rtf,
            'first_packet': first_packet,
            'chunks': len(all_chunks),
            'voice': resp.headers.get('X-Voice', voice),
            'emotion': resp.headers.get('X-Emotion', emotion),
        }

        print(f'  [Stream] voice={voice} | emotion={emotion} | text_len={len(text)} | '
              f'chunks={len(all_chunks)} | duration={duration:.2f}s | '
              f'first={first_packet:.3f}s | RTF={rtf:.3f}')
        return audio, sr, meta

    def synthesize_to_file(self, text: str, output_path: str, voice: str = 'woman',
                           emotion: str = 'calm', speed: float = 1.0) -> dict:
        """非流式合成并直接保存到文件"""
        audio, sr, meta = self.synthesize(text, voice=voice, emotion=emotion, speed=speed)
        sf.write(output_path, audio, sr)
        print(f'  → Saved: {output_path}')
        return meta

    # ========== OpenAI 兼容 ==========

    def tts_bytes(self, text: str, voice: str = 'alloy', emotion: str = 'calm',
                  speed: float = 1.0, fmt: str = 'wav') -> bytes:
        """
        OpenAI 兼容格式合成 → 返回音频 bytes

        参数:
            text: 输入文本
            voice: 音色（alloy/echo/fable/onyx/nova/shimmer → 自动映射）
            speed: 语速
            fmt: wav | mp3 | opus | aac | flac
        """
        resp = self._session.post(
            f'{self.base_url}/v1/audio/speech',
            json={
                'model': 'tts-1',
                'input': text,
                'voice': voice,
                'response_format': fmt,
                'speed': speed,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content


# ---------- 自测 ----------
if __name__ == '__main__':
    client = ChengbookTTSClient()

    # 健康检查
    print('=' * 60)
    print('诚书记 TTS 客户端 — 自测')
    print('诚太乙 AI 教练：模拟客户语音')
    print('=' * 60)

    try:
        health = client.health()
        print(f'\n✅ 服务正常: {health["service"]}')
        print(f'   引擎: {health.get("engine", health.get("model", "?"))}')
        print(f'   版本: {health.get("version", "?")}')
        print(f'   采样率: {health["sample_rate"]}Hz')
        print(f'   能力: {health.get("capabilities", {})}')
        print(f'   客户音色: {health["voices"]}')
        print(f'   客户情绪: {health["emotions"]}')

        voices = client.list_voices()
        emotions = client.list_emotions()
        models = client.list_models()
        print(f'\n可用音色: {[v["id"] for v in voices]}')
        print(f'可用情绪: {[e["id"] for e in emotions]}')
        print(f'当前模型: {models["active"]}')
        print(f'可用模型: {[m["type"] for m in models["models"]]}')
    except Exception as e:
        print(f'❌ 连接失败: {e}')
        print('请先启动服务: python -m chengbook_tts.cli serve --port 8080')
        exit(1)

    # 模拟客户不同场景
    print('\n' + '-' * 60)
    print('模拟客户不同情绪（非流式）')
    print('-' * 60)

    test_cases = [
        # (客户台词, 音色, 情绪, 语速, 文件名, 场景说明)
        ('你好，我想查一下我的订单到哪了，订单号是8823。', 'woman', 'calm', 1.0, 'test_calm_query.wav', '普通咨询'),
        ('太好了！终于有货了，我等你家这个等了好久了！', 'woman', 'happy', 1.0, 'test_happy_customer.wav', '客户高兴'),
        ('唉，我昨天才买的，今天就降价了，这让我心里很不舒服。', 'man', 'sad', 1.0, 'test_sad_customer.wav', '客户不满'),
        ('你们到底怎么回事！我已经等了半个月了，一拖再拖，今天必须给我一个说法！', 'man', 'angry', 1.0, 'test_angry_complaint.wav', '愤怒投诉'),
        ('那个……我想问一下，这个东西怎么退啊？我不太懂这些……', 'woman', 'soft', 1.0, 'test_hesitant_customer.wav', '犹豫客户'),
        ('快点快点，我还有事呢，能不能别磨叽，赶紧给我办了！', 'woman', 'impatient', 1.0, 'test_impatient_customer.wav', '急躁催促'),
        ('你刚才说的那个什么优惠券，我没太听明白，能再说一遍吗？', 'man', 'confused', 1.0, 'test_confused_customer.wav', '困惑客户'),
        ('我告诉你，今天这事不解决我就坐这儿不走了！！', 'woman', 'loud', 1.0, 'test_loud_customer.wav', '大声喧哗'),
    ]

    for text, voice, emotion, speed, filename, scene in test_cases:
        print(f'\n🎯 [{scene}] voice={voice}, emotion={emotion}')
        print(f'   客户: {text[:50]}...')
        audio, sr, meta = client.synthesize(text, voice=voice, emotion=emotion, speed=speed)
        sf.write(filename, audio, sr)
        print(f'   → {filename} | 音频:{meta["duration"]:.2f}s | RTF:{meta["rtf"]:.3f}')

    # 流式测试
    print('\n' + '-' * 60)
    print('模拟客户实时对话（流式）')
    print('-' * 60)

    try:
        audio, sr, meta = client.synthesize_stream(
            '喂，你好，我问一下我这个快递什么时候能到？', voice='woman', emotion='calm'
        )
        sf.write('test_stream_customer.wav', audio, sr)
        print(f'  → test_stream_customer.wav')
    except Exception as e:
        print(f'  ⚠ 流式合成失败（当前引擎可能不支持）: {e}')

    # 语速测试（模拟不同说话速度的客户）
    print('\n' + '-' * 60)
    print('语速对比测试（模拟不同语速客户）')
    print('-' * 60)

    for spd, fname, desc in [
        (0.75, 'test_speed_slow.wav', '语速慢的客户'),
        (1.0, 'test_speed_normal.wav', '正常语速客户'),
        (1.5, 'test_speed_fast.wav', '语速快的客户'),
    ]:
        try:
            audio, sr, meta = client.synthesize(
                '你好，我想问一下我这个订单什么时候能发货？', voice='woman', emotion='calm', speed=spd
            )
            sf.write(fname, audio, sr)
            print(f'  [{desc}] speed={spd}: duration={meta["duration"]:.2f}s → {fname}')
        except Exception as e:
            print(f'  [{desc}] speed={spd}: ❌ {e}')

    print('\n' + '=' * 60)
    print('全部测试完成! 🎉')
    print('=' * 60)
