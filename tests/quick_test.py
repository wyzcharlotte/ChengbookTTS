"""快速冒烟测试 — 需要服务正在运行"""

import time
import requests
import argparse


def quick_test(base_url: str = 'http://localhost:8080'):
    """快速冒烟测试"""
    passed = 0
    failed = 0

    def check(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f'  [PASS] {name}')
            passed += 1
        except Exception as e:
            print(f'  [FAIL] {name}: {e}')
            failed += 1

    print('=== ChengbookTTS Smoke Test ===')
    print(f'Target: {base_url}')
    print()

    # 1. Health
    def test_health():
        r = requests.get(f'{base_url}/api/health', timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data['status'] == 'ok'
        assert 'engine' in data

    check('GET /api/health', test_health)

    # 2. Voices
    def test_voices():
        r = requests.get(f'{base_url}/api/voices', timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert len(data['voices']) > 0

    check('GET /api/voices', test_voices)

    # 3. Emotions
    def test_emotions():
        r = requests.get(f'{base_url}/api/emotions', timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert len(data['emotions']) > 0

    check('GET /api/emotions', test_emotions)

    # 4. TTS
    def test_tts():
        r = requests.post(
            f'{base_url}/api/tts',
            json={'text': '测试语音合成', 'voice': 'woman', 'emotion': 'calm'},
            timeout=60,
        )
        assert r.status_code == 200
        assert len(r.content) > 0
        assert r.headers.get('content-type') == 'audio/wav'

    check('POST /api/tts', test_tts)

    # 5. OpenAI API
    def test_openai_tts():
        r = requests.post(
            f'{base_url}/v1/audio/speech',
            json={'model': 'tts-1', 'input': '测试OpenAI接口', 'voice': 'alloy'},
            timeout=60,
        )
        assert r.status_code == 200
        assert len(r.content) > 0

    check('POST /v1/audio/speech', test_openai_tts)

    # 6. Profile
    def test_profile():
        r = requests.get(f'{base_url}/api/profile', timeout=10)
        assert r.status_code == 200

    check('GET /api/profile', test_profile)

    print()
    print(f'=== Results: {passed} passed, {failed} failed ===')
    return failed == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8080')
    args = parser.parse_args()
    quick_test(args.url)
