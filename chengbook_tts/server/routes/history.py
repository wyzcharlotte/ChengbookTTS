"""
合成历史 API 路由
----------------
GET  /api/history         — 获取合成历史记录
DELETE /api/history       — 清空历史记录
"""

from fastapi import APIRouter, Query

from chengbook_tts.server.history import get_history_manager

router = APIRouter(tags=['Synthesis History'])


@router.get('/history')
async def get_history(
    limit: int = Query(default=50, ge=1, le=200, description='返回条数'),
    offset: int = Query(default=0, ge=0, description='偏移量'),
):
    """获取合成历史记录（倒序，最新的在前）"""
    return get_history_manager().get_history(limit=limit, offset=offset)


@router.delete('/history')
async def clear_history():
    """清空合成历史记录"""
    mgr = get_history_manager()
    count = mgr.count
    mgr.clear()
    return {'success': True, 'cleared': count, 'message': f'已清空 {count} 条合成记录'}
