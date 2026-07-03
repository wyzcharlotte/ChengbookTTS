"""
情绪预设配置
-----------
定义 8 种通用情绪的 instruct 文本（用于 CosyVoice3 的 LLM 引导）。
其他模型通过各自的适配器映射到对应的控制机制。
"""

EMOTIONS: dict[str, dict] = {
    'calm': {
        'name': '平和',
        'instruct': 'You are a helpful assistant. 请用自然、平和的语气说一句话。<|endofprompt|>',
        'description': '普通客户正常咨询，默认推荐',
    },
    'happy': {
        'name': '开心',
        'instruct': 'You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>',
        'description': '客户心情愉悦，配合度高',
    },
    'sad': {
        'name': '悲伤',
        'instruct': 'You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>',
        'description': '客户情绪低落，需要安抚',
    },
    'angry': {
        'name': '生气',
        'instruct': 'You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>',
        'description': '客户愤怒投诉，考验坐席应对能力',
    },
    'soft': {
        'name': '温柔',
        'instruct': 'You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>',
        'description': '客户语气轻柔，可能犹豫不决',
    },
    'loud': {
        'name': '大声',
        'instruct': 'You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>',
        'description': '客户大声喧哗，情绪激动',
    },
    'impatient': {
        'name': '急躁',
        'instruct': 'You are a helpful assistant. 请用非常不耐烦、急躁的语气说一句话。<|endofprompt|>',
        'description': '客户不耐烦催促，考验坐席效率',
    },
    'confused': {
        'name': '困惑',
        'instruct': 'You are a helpful assistant. 请用困惑、迷茫的语气说一句话。<|endofprompt|>',
        'description': '客户对业务不熟悉，需要耐心引导',
    },
}


def get_emotions() -> dict:
    """获取所有情绪预设"""
    return dict(EMOTIONS)
