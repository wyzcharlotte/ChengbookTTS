"""
模型管理 API 路由
----------------
GET  /api/models       — 列出所有可用模型
POST /api/model/switch — 切换模型
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from chengbook_tts.engines.manager import ModelManager
from chengbook_tts.server.dependencies import get_model_manager, get_engine
from chengbook_tts.engines.base import TTSEngine

router = APIRouter(tags=['Model Management'])


class ModelSwitchRequest(BaseModel):
    model_type: str = Field(..., description='模型类型: cosyvoice3 | soulxpodcast | indextts_v1 | indextts_v2')


@router.get('/models')
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """列出所有可用模型及当前激活状态"""
    return {
        'models': manager.list_models(),
        'active': manager.model_type,
    }


@router.post('/model/switch')
async def switch_model(
    req: ModelSwitchRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """
    切换到指定模型。

    - 卸载当前模型（释放 GPU 显存）
    - 加载目标模型
    - 新模型初始化后自动可用
    """
    if req.model_type not in ModelManager.ENGINE_CLASS_MAP:
        raise HTTPException(
            status_code=400,
            detail=f'不支持的模型类型: {req.model_type}。'
                   f'可用: {list(ModelManager.ENGINE_CLASS_MAP.keys())}',
        )

    try:
        engine = manager.load_model(req.model_type)

        # 恢复自定义音色（切换模型后需要重新注册）
        from chengbook_tts.server.routes.custom_voices import load_custom_voices_on_startup
        try:
            load_custom_voices_on_startup()
        except Exception as e:
            logging.warning(f'Custom voice restore after switch failed (non-critical): {e}')

        return {
            'success': True,
            'model_type': req.model_type,
            'engine_name': engine.engine_name,
            'engine_version': engine.engine_version,
            'voices': len(engine.voice_ids),
            'emotions': len(engine.emotion_ids),
            'message': f'已切换到 {engine.engine_name} v{engine.engine_version}',
        }
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f'模型 [{req.model_type}] 依赖缺失，请先安装对应依赖:\n{str(e)}',
        )
    except Exception as e:
        logging.error(f'Model switch failed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'模型切换失败: {str(e)}')
