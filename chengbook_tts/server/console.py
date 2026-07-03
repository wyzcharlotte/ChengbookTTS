"""
Web 控制台
---------
内联 SPA 页面 (HTML + JS)，保留原项目全部功能：
- 音色选择、情绪选择、语速调节、分词开关
- 自定义音色上传/删除
- 配置持久化到 /api/profile
- 实时状态指示

新增功能：
- 模型切换 — 在控制台中选择并切换到不同 TTS 模型
"""

CONSOLE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>诚书记 TTS 控制台 — ChengbookTTS</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Microsoft YaHei','PingFang SC',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;justify-content:center;padding:20px}
.wrap{max-width:700px;width:100%}
h1{font-size:22px;text-align:center;margin:16px 0 4px}
h1 span{color:#38bdf8}
.sub{text-align:center;color:#64748b;font-size:13px;margin-bottom:20px}
.card{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:14px}
.card h2{font-size:15px;color:#94a3b8;margin-bottom:14px;border-bottom:1px solid #334155;padding-bottom:8px}
.row{display:flex;flex-wrap:wrap;gap:8px}
.btn{border:none;border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;transition:.15s;background:#334155;color:#cbd5e1}
.btn:hover{background:#475569}
.btn.on{background:#0ea5e9;color:#fff}
.btn.model-on{background:#8b5cf6;color:#fff}
.btn-switch{background:#0ea5e9!important;color:#fff!important;font-weight:bold;margin-top:8px;width:100%;padding:10px;font-size:14px}
.btn-switch:hover{background:#0284c7!important}
.btn-del{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:50%;border:none;background:rgba(239,68,68,0.5);color:#fecaca;font-size:12px;cursor:pointer;margin-left:6px;transition:.15s;line-height:1;flex-shrink:0}
.btn-del:hover{background:#dc2626;color:#fff}
.voice-btn-wrap{display:inline-flex;align-items:center}
.section-disabled{opacity:.35;pointer-events:none}
.section-disabled h2::after{content:' (当前模型不支持)';font-size:11px;color:#ef4444;font-weight:normal;margin-left:4px}
.slider-wrap{display:flex;align-items:center;gap:12px;margin:4px 0}
.slider-wrap input[type=range]{flex:1;accent-color:#0ea5e9}
.speed-val{min-width:36px;text-align:center;font-weight:bold;color:#38bdf8;font-size:14px}
textarea{width:100%;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-size:14px;resize:vertical;min-height:60px}
textarea:focus{outline:none;border-color:#0ea5e9}
.btn-play{width:100%;margin-top:10px;background:#0ea5e9;color:#fff;font-size:15px;padding:12px;border-radius:8px;border:none;cursor:pointer}
.btn-play:hover{background:#0284c7}
.btn-play:disabled{opacity:.5;cursor:not-allowed}
.status{color:#94a3b8;font-size:12px;margin-top:10px;text-align:center}
.status.ok{color:#22c55e}
.status.err{color:#ef4444}
.status.warn{color:#f59e0b}
.note{font-size:12px;color:#64748b;margin-top:8px}
.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:6px;animation:pulse 2s infinite}
.model-badge{display:inline-block;background:#8b5cf6;color:#fff;font-size:11px;padding:2px 8px;border-radius:10px;margin-left:6px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes loading-bar-stripes{0%{background-position:40px 0}100%{background-position:0 0}}
.model-info{font-size:12px;color:#64748b;margin-top:6px}
.model-info span{color:#a78bfa}
.progress-wrap{display:none;margin-top:8px}
.progress-wrap.active{display:block}
.progress-bar{height:6px;border-radius:3px;background:#334155;overflow:hidden}
.progress-bar-inner{height:100%;border-radius:3px;background:linear-gradient(90deg,#8b5cf6,#0ea5e9,#8b5cf6);background-size:40px 100%;animation:loading-bar-stripes 0.8s linear infinite;transition:width 0.3s}
.progress-text{font-size:12px;color:#94a3b8;margin-top:4px}
.history-table{width:100%;border-collapse:collapse;font-size:11px;margin-top:6px}
.history-table th{background:#0f172a;color:#94a3b8;padding:6px 8px;text-align:left;font-weight:normal;position:sticky;top:0;z-index:1}
.history-table td{padding:5px 8px;border-bottom:1px solid #1e293b;color:#cbd5e1;white-space:nowrap}
.history-table tr:hover td{background:#1e293b}
.history-table .ht-text{max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.history-table .ht-ok{color:#22c55e}
.history-table .ht-err{color:#ef4444}
.history-table .ht-rtf{font-family:'Cascadia Code','Fira Code',monospace;color:#38bdf8}
.history-scroll{max-height:360px;overflow-y:auto;border-radius:8px}
.history-bar{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.history-bar .ht-count{font-size:12px;color:#64748b}
.btn-sm{border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;transition:.15s;background:#334155;color:#cbd5e1}
.btn-sm:hover{background:#475569}
.btn-sm.danger{background:rgba(239,68,68,0.3);color:#fecaca}
.btn-sm.danger:hover{background:#dc2626;color:#fff}
</style>
</head>
<body>
<div class="wrap">
<h1>&#x1f399; <span>诚书记 TTS</span> 控制台</h1>
<div class="sub">
  当前生效: <span style="color:#e2e8f0" id="liveLabel">-</span>
  <span class="live" id="liveDot" style="display:none"></span>
  <span class="model-badge" id="modelBadge" style="display:none"></span>
</div>

<!-- 模型切换 -->
<div class="card">
<h2>&#x1f504; 模型切换 <span style="font-size:11px;color:#64748b;font-weight:normal">— 切换后原模型自动卸载</span></h2>
<div class="row" id="models"></div>
<div class="model-info" id="modelInfo"></div>
<div class="progress-wrap" id="progressWrap">
  <div class="progress-bar"><div class="progress-bar-inner" id="progressBarInner" style="width:0%"></div></div>
  <div class="progress-text" id="progressText">⏳ 正在加载模型...</div>
</div>
</div>

<!-- 音色 -->
<div class="card">
<h2>&#x1f464; 音色</h2>
<div class="row" id="voices"></div>
<div style="margin-top:12px;padding-top:12px;border-top:1px dashed #334155">
  <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">&#x1f3a4; 自定义音色克隆</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <input type="file" id="custWav" accept=".wav,audio/wav" style="flex:1;min-width:130px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:6px;font-size:12px">
    <input type="text" id="custName" placeholder="音色名称" style="flex:1;min-width:110px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:6px 10px;font-size:12px">
    <button class="btn on" onclick="uploadVoice()" style="white-space:nowrap;font-size:12px;padding:6px 12px">&#x1f4e4; 上传克隆</button>
  </div>
  <div class="status" id="uploadStatus"></div>
  <div id="customVoices" style="margin-top:6px"></div>
</div>
</div>

<!-- 情绪 -->
<div class="card">
<h2>&#x1f60a; 情绪</h2>
<div class="row" id="emotions"></div>
</div>

<!-- 语速 -->
<div class="card">
<h2>&#x26a1; 语速</h2>
<div class="slider-wrap">
  <span style="font-size:12px;color:#94a3b8">0.5x</span>
  <input type="range" id="speed" min="0.5" max="2.0" step="0.05" value="1.0">
  <span style="font-size:12px;color:#94a3b8">2.0x</span>
  <span class="speed-val" id="speedVal">1.0</span>
</div>
</div>

<!-- 分词 -->
<div class="card">
<h2>&#x1f524; 分词 []</h2>
<div class="row">
  <button class="btn on" id="segOn" onclick="setSegment(true)">&#x2705; 开启</button>
  <button class="btn" id="segOff" onclick="setSegment(false)">&#x2b1b; 关闭</button>
</div>
<div class="note">二字词自动合并加 []，改善断句</div>
</div>

<!-- 试听 -->
<div class="card">
<h2>&#x1f50a; 试听</h2>
<textarea id="text" placeholder="输入客户对话文本...">你好，我想查一下我的订单到哪了</textarea>
<button class="btn-play" id="btnPlay" onclick="synthesize()">&#x25b6; 合成试听</button>
<div style="margin-top:10px" id="audioWrap"></div>
<div class="status" id="synthStatus"></div>
</div>

<!-- 合成记录展板 -->
<div class="card">
<div class="history-bar">
<h2>&#x1f4ca; 合成记录展板</h2>
<div>
<button class="btn-sm" onclick="refreshHistory()" title="刷新">&#x1f504; 刷新</button>
<button class="btn-sm danger" onclick="clearHistory()" title="清空记录">&#x1f5d1; 清空</button>
</div>
</div>
<span class="ht-count" id="htCount">加载中...</span>
<div class="history-scroll" id="historyScroll">
<table class="history-table">
<thead><tr>
<th>时间</th><th>文本</th><th>音色</th><th>情绪</th><th>语速</th><th>模型</th><th>时长</th><th>耗时</th><th>RTF</th><th>状态</th>
</tr></thead>
<tbody id="historyBody"></tbody>
</table>
</div>
</div>
</div>

<script>
let voices=[], emotions=[], models=[];
let voice='woman', emotion='calm', speed=1.0, segment=true;
let activeModel='', switching=false;
let caps={streaming:true,emotion:true,multi_speaker:true,speed_control:true,segmentation:true};

// ---- 工具函数: 安全解析 JSON（处理服务器返回 HTML 错误页的情况） ----
async function safeJson(r){
  let text=await r.text();
  try{ return JSON.parse(text); }
  catch(e){
    // 如果是 HTML 错误页，提取有用信息
    let m=text.match(/<title>(.*?)<\/title>/);
    if(m) throw new Error(m[1]);
    // 截取前 100 个字符
    throw new Error(text.substring(0,100));
  }
}

// ==================== 初始化 ====================
async function init(){
  try{
    // 1. 加载模型列表
    await loadModels();
    // 2. 加载配置（音色/情绪/语速）
    await loadConfig();
    // 3. 加载音色列表
    await loadVoices();
    // 4. 加载情绪列表
    await loadEmotions();
    // 5. 恢复上次设置
    await restoreProfile();
    // 6. 加载历史记录
    loadHistory();
    // 刷新UI
    renderAll();
  }catch(err){
    document.getElementById('liveLabel').textContent='❌ '+err.message;
  }
}

// ==================== 模型 ====================
async function loadModels(){
  try{
    let r=await fetch('/api/models');
    if(!r.ok) throw new Error('无法获取模型列表');
    let d=await r.json();
    models=d.models||[];
    activeModel=d.active||'';
    renderModels();
  }catch(err){ console.error(err); }
}

function renderModels(){
  let el=document.getElementById('models');
  el.innerHTML=models.map(m=>{
    let cls='btn';
    if(m.loaded) cls+=' model-on';
    if(!m.model_dir) cls+='';
    let label=m.name||m.type;
    if(m.loaded) label='● '+label;
    let tip=m.model_dir?'模型已就绪':'⚠ 模型路径未配置';
    return `<button class="${cls}" data-m="${m.type}" onclick="switchModel('${m.type}')" title="${tip}">${label}</button>`;
  }).join('');
}

async function switchModel(mtype){
  if(switching) return;
  if(mtype===activeModel){
    showInfo('已经是当前模型: '+mtype);
    return;
  }
  switching=true;
  let info=document.getElementById('modelInfo');
  info.innerHTML='';
  let pw=document.getElementById('progressWrap');
  let pb=document.getElementById('progressBarInner');
  let pt=document.getElementById('progressText');
  pw.classList.add('active');
  pb.style.width='0%';
  pt.textContent='⏳ 正在卸载当前模型...';

  // 模拟进度：先用 2s 到 30%
  let progressTimer=setInterval(()=>{
    let w=parseFloat(pb.style.width);
    if(w<90) pb.style.width=(w+Math.random()*8+1)+'%';
  },400);

  try{
    pt.textContent='⏳ 正在加载新模型（可能需要数十秒）...';
    let r=await fetch('/api/model/switch',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model_type:mtype})
    });
    clearInterval(progressTimer);
    if(!r.ok){ let e=await safeJson(r); throw new Error(e.detail||r.statusText); }
    pb.style.width='100%';
    pt.textContent='✅ 加载完成，正在刷新界面...';
    let d=await safeJson(r);
    activeModel=d.model_type;

    // 重新加载音色/情绪/配置 + 刷新 UI
    await Promise.all([loadVoices(), loadEmotions(), loadConfig()]);
    await restoreProfile();
    await loadModels();  // 刷新模型按钮状态
    renderAll();
    updateCapUI();

    setTimeout(()=>{
      pw.classList.remove('active');
      info.innerHTML='✅ '+d.message;
      info.style.color='#22c55e';
      setTimeout(()=>{ info.innerHTML=''; }, 4000);
    },400);
  }catch(err){
    clearInterval(progressTimer);
    pw.classList.remove('active');
    info.innerHTML='❌ 切换失败: '+err.message;
    info.style.color='#ef4444';
  }
  switching=false;
}

// ==================== 配置加载 ====================
async function loadConfig(){
  try{
    let r=await fetch('/api/health');
    if(!r.ok) throw new Error('服务未就绪');
    let d=await safeJson(r);
    // 更新模型badge
    let badge=document.getElementById('modelBadge');
    badge.style.display='inline';
    badge.textContent=(d.engine||'N/A')+' v'+(d.version||'');
    // 存储能力标志
    if(d.capabilities) caps=d.capabilities;
    updateCapUI();
  }catch(err){ console.error(err); }
}

function updateCapUI(){
  // 情绪
  let emoCard=document.getElementById('emotions').parentElement;
  emoCard.classList.toggle('section-disabled',!caps.emotion);
  // 语速
  let speedCard=document.getElementById('speed').closest('.card');
  if(speedCard) speedCard.classList.toggle('section-disabled',!caps.speed_control);
  // 分词
  let segCard=document.getElementById('segOn').closest('.card');
  if(segCard) segCard.classList.toggle('section-disabled',!caps.segmentation);
  // 多说话人 — 自定义音色区域
  let custArea=document.getElementById('customVoices').parentElement;
  if(custArea) custArea.style.display=caps.multi_speaker?'':'none';
}

async function loadVoices(){
  try{
    let r=await fetch('/api/voices');
    if(!r.ok) throw new Error('无法获取音色');
    let d=await safeJson(r);
    voices=d.voices||[];
  }catch(err){ console.error(err); }
}

async function loadEmotions(){
  try{
    let r=await fetch('/api/emotions');
    if(!r.ok) throw new Error('无法获取情绪');
    let d=await safeJson(r);
    emotions=d.emotions||[];
  }catch(err){ console.error(err); }
}

async function restoreProfile(){
  try{
    let r=await fetch('/api/profile');
    if(!r.ok) return;
    let p=await safeJson(r);
    if(p.voice && voices.find(v=>v.id===p.voice)) voice=p.voice;
    else if(voices.length) voice=voices[0].id;
    if(p.emotion && emotions.find(e=>e.id===p.emotion)) emotion=p.emotion;
    else if(emotions.length) emotion=emotions[0].id;
    if(p.speed!==undefined) speed=p.speed;
    if(p.segment!==undefined) segment=p.segment;
  }catch(err){ console.error(err); }
}

// ==================== UI 渲染 ====================
function renderAll(){
  renderVoices();
  renderEmotions();
  renderModels();
  document.getElementById('speed').value=speed;
  document.getElementById('speedVal').textContent=speed.toFixed(2);
  updateSegUI();
  updateLive();
}

function renderVoices(){
  document.getElementById('voices').innerHTML=voices.map((v,i)=>{
    let delBtn=i>=2?`<button class="btn-del" onclick="deleteVoiceInline('${v.id}')" title="删除此音色">&times;</button>`:'';
    return `<span class="voice-btn-wrap"><button class="btn" data-v="${v.id}" onclick="setVoice('${v.id}')">${v.name}${v.is_custom?' (自定义)':''}</button>${delBtn}</span>`;
  }).join('');
  highlight('voices','data-v',voice);
  renderCustomVoices();
}

function renderCustomVoices(){
  let el=document.getElementById('customVoices');
  let cust=voices.filter(v=>v.is_custom);
  if(!cust.length){ el.innerHTML=''; return; }
  el.innerHTML='<span style="font-size:12px;color:#64748b">已克隆的音色：</span> '+
    cust.map(v=>`<span style="color:#94a3b8;font-size:12px;margin-right:8px">${v.name}</span>`).join('');
}

function renderEmotions(){
  document.getElementById('emotions').innerHTML=emotions.map(e=>
    `<button class="btn" data-e="${e.id}" onclick="setEmotion('${e.id}')">${e.name}</button>`
  ).join('');
  highlight('emotions','data-e',emotion);
}

function highlight(id,attr,val){
  document.getElementById(id).querySelectorAll('.btn').forEach(b=>b.classList.toggle('on',b.getAttribute(attr)===val));
}

// ==================== 设置 ====================
async function setVoice(v){ voice=v; highlight('voices','data-v',v); await saveProfile(); }
async function setEmotion(e){ emotion=e; highlight('emotions','data-e',e); await saveProfile(); }

document.getElementById('speed').addEventListener('change',async function(){
  speed=parseFloat(this.value);
  document.getElementById('speedVal').textContent=speed.toFixed(2);
  await saveProfile();
});

async function setSegment(v){ segment=v; updateSegUI(); await saveProfile(); }

function updateSegUI(){
  document.getElementById('segOn').classList.toggle('on',segment);
  document.getElementById('segOff').classList.toggle('on',!segment);
}

async function saveProfile(){
  try{
    await fetch('/api/profile',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({voice,emotion,speed,segment})
    });
    updateLive();
  }catch(err){ console.error(err); }
}

function updateLive(){
  document.getElementById('liveDot').style.display='inline';
  let vn=voices.find(v=>v.id===voice), en=emotions.find(e=>e.id===emotion);
  document.getElementById('liveLabel').textContent=
    (vn?.name||voice)+' / '+(en?.name||emotion)+' / '+speed.toFixed(2)+'x';
}

// ==================== 自定义音色 ====================
async function uploadVoice(){
  let file=document.getElementById('custWav').files[0];
  let name=document.getElementById('custName').value.trim();
  if(!file){ setUplStatus('❌ 请选择 WAV 文件'); return; }
  if(!name){ setUplStatus('❌ 请输入音色名称'); return; }
  setUplStatus('⏳ 上传中...');
  let fd=new FormData(); fd.append('file',file); fd.append('name',name);
  try{
    let r=await fetch('/api/voices/custom',{method:'POST',body:fd});
    if(!r.ok){ let e=await safeJson(r); throw new Error(e.detail||r.statusText); }
    let d=await safeJson(r);
    setUplStatus('✅ '+d.message);
    document.getElementById('custWav').value=''; document.getElementById('custName').value='';
    await loadVoices(); renderVoices();
  }catch(err){ setUplStatus('❌ '+err.message); }
}

async function deleteVoiceInline(vid){
  if(!confirm('确定删除自定义音色 '+vid+' ？')) return;
  try{
    let r=await fetch('/api/voices/custom/'+vid,{method:'DELETE'});
    if(!r.ok){ let e=await safeJson(r); throw new Error(e.detail||r.statusText); }
    if(voice===vid) voice='woman';
    await loadVoices(); renderVoices(); await saveProfile();
  }catch(err){ setUplStatus('❌ '+err.message); }
}

async function deleteVoice(vid){
  if(!confirm('确定删除自定义音色 '+vid+' ？')) return;
  try{
    let r=await fetch('/api/voices/custom/'+vid,{method:'DELETE'});
    if(!r.ok){ let e=await safeJson(r); throw new Error(e.detail||r.statusText); }
    let d=await safeJson(r);
    setUplStatus('✅ '+d.message);
    if(voice===vid) voice='woman';
    await loadVoices(); renderVoices(); await saveProfile();
  }catch(err){ setUplStatus('❌ '+err.message); }
}

function setUplStatus(msg){
  let el=document.getElementById('uploadStatus');
  el.textContent=msg;
  el.className='status'+(msg.startsWith('✅')?' ok':msg.startsWith('❌')?' err':msg.startsWith('⏳')?' warn':'');
}

function showInfo(msg){
  let el=document.getElementById('modelInfo');
  el.innerHTML=msg;
  el.style.color='#94a3b8';
  setTimeout(()=>{ el.innerHTML=''; }, 3000);
}

// ==================== 合成 ====================
async function synthesize(){
  let text=document.getElementById('text').value.trim();
  if(!text){ document.getElementById('synthStatus').textContent='❌ 请输入文本'; document.getElementById('synthStatus').className='status err'; return; }
  let btn=document.getElementById('btnPlay'); btn.disabled=true; btn.textContent='⏳ 生成中...';
  document.getElementById('audioWrap').innerHTML=''; document.getElementById('synthStatus').textContent='';
  try{
    let resp=await fetch('/api/tts',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text,voice,emotion,speed,segment})
    });
    if(!resp.ok){ let e=await safeJson(resp); throw new Error(e.detail||resp.statusText); }
    let blob=await resp.blob(), url=URL.createObjectURL(blob);
    document.getElementById('audioWrap').innerHTML=`<audio controls autoplay src="${url}"></audio>`;
    document.getElementById('synthStatus').textContent='✅ 完成';
    document.getElementById('synthStatus').className='status ok';
    // 刷新历史记录
    loadHistory();
  }catch(err){
    document.getElementById('synthStatus').textContent='❌ '+err.message;
    document.getElementById('synthStatus').className='status err';
    // 也刷新（会记录失败）
    loadHistory();
  }
  btn.disabled=false; btn.textContent='▶ 合成试听';
}

// ==================== 合成记录展板 ====================
async function loadHistory(){
  try{
    let r=await fetch('/api/history?limit=50');
    if(!r.ok) return;
    let d=await safeJson(r);
    renderHistory(d.records||[]);
    document.getElementById('htCount').textContent='共 '+(d.total||0)+' 条记录';
  }catch(err){ console.error(err); }
}

function renderHistory(records){
  let tbody=document.getElementById('historyBody');
  if(!records.length){
    tbody.innerHTML='<tr><td colspan="10" style="text-align:center;color:#64748b;padding:16px">暂无合成记录</td></tr>';
    return;
  }
  tbody.innerHTML=records.map(r=>{
    let statusCls=r.success?'ht-ok':'ht-err';
    let statusText=r.success?'✅':'❌ '+r.error;
    let streamTag=r.streaming?' ⚡':'';
    return `<tr>
      <td>${r.timestamp}</td>
      <td class="ht-text" title="${r.text.replace(/"/g,'&quot;')}">${r.text}</td>
      <td>${r.voice_name}</td>
      <td>${r.emotion_name}</td>
      <td>${r.speed.toFixed(2)}x</td>
      <td>${r.model}</td>
      <td>${r.duration}s</td>
      <td>${r.elapsed}s${streamTag}</td>
      <td class="ht-rtf">${r.rtf.toFixed(3)}</td>
      <td class="${statusCls}">${statusText}</td>
    </tr>`;
  }).join('');
}

function refreshHistory(){ loadHistory(); }

async function clearHistory(){
  if(!confirm('确定清空所有合成记录？')) return;
  try{
    let r=await fetch('/api/history',{method:'DELETE'});
    if(!r.ok){ let e=await safeJson(r); throw new Error(e.detail||r.statusText); }
    let d=await safeJson(r);
    loadHistory();
  }catch(err){ console.error(err); }
}

// ==================== 启动 ====================
document.addEventListener('DOMContentLoaded',init);
</script>
</body>
</html>"""


def serve_console():
    """返回 Web 控制台 HTML 响应"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=CONSOLE_HTML)
