let running = false;
let timer = null;
let initAttempts = 0;

const canvas = document.getElementById('priceChart');
const ctx = canvas.getContext('2d');
const candles = [];
let viewBars = 180;

function resizeCanvas(){
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth;
  const height = 420;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.height = `${height}px`;
  canvas.style.width = `${width}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawCandles();
}

function drawCandles(){
  const width = canvas.clientWidth || 800;
  const height = 420;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#1b2140';
  ctx.fillRect(0, 0, width, height);

  const data = candles.slice(-viewBars);
  if(!data.length){
    ctx.fillStyle = '#c9d1ff';
    ctx.fillText('Aucune bougie disponible', 16, 24);
    return;
  }

  let minP = Infinity;
  let maxP = -Infinity;
  data.forEach(b => {
    minP = Math.min(minP, b.low);
    maxP = Math.max(maxP, b.high);
  });
  const pad = Math.max((maxP - minP) * 0.08, 0.01);
  minP -= pad;
  maxP += pad;

  const px = p => 20 + (height - 40) * (1 - (p - minP) / Math.max(maxP - minP, 1e-9));
  const barW = Math.max(2, Math.floor((width - 30) / data.length) - 1);

  data.forEach((b, i) => {
    const x = 20 + i * ((width - 30) / data.length);
    const yOpen = px(b.open);
    const yClose = px(b.close);
    const yHigh = px(b.high);
    const yLow = px(b.low);
    const up = b.close >= b.open;
    ctx.strokeStyle = up ? '#2ecc71' : '#e74c3c';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + barW / 2, yHigh);
    ctx.lineTo(x + barW / 2, yLow);
    ctx.stroke();

    ctx.fillStyle = up ? '#2ecc71' : '#e74c3c';
    const top = Math.min(yOpen, yClose);
    const body = Math.max(1, Math.abs(yClose - yOpen));
    ctx.fillRect(x, top, barW, body);
  });
}

function parseCandle(c){
  if(!c) return null;
  const required = ['datetime','open','high','low','close'];
  for(const k of required){ if(c[k] === undefined || c[k] === null) return null; }
  const dt = new Date(c.datetime);
  const bar = {
    time: Math.floor(dt.getTime() / 1000),
    open: Number(c.open),
    high: Number(c.high),
    low: Number(c.low),
    close: Number(c.close),
  };
  if(!Number.isFinite(bar.time) || !Number.isFinite(bar.open) || !Number.isFinite(bar.high) || !Number.isFinite(bar.low) || !Number.isFinite(bar.close)) return null;
  return bar;
}

function logEvent(msg){
  const el = document.getElementById('events');
  el.innerHTML = `<div>${new Date().toLocaleTimeString()} - ${msg}</div>` + el.innerHTML;
}

function setStatus(msg){
  document.getElementById('status').textContent = `${msg} • ${new Date().toLocaleTimeString()}`;
}

function renderDebug(debug, errorMsg=''){
  const info = document.getElementById('debugInfo');
  info.textContent = JSON.stringify(debug || {}, null, 2);
  const err = document.getElementById('debugError');
  err.textContent = errorMsg || (debug && debug.error ? debug.error : '');
}

async function api(path, payload={}, {timeoutMs=12000, method='POST'} = {}){
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const opts = { method, signal: ctrl.signal };
    if(method !== 'GET'){
      opts.headers = {'Content-Type':'application/json'};
      opts.body = JSON.stringify(payload || {});
    }
    const r = await fetch(path, opts);
    const data = await r.json();
    return data;
  } finally {
    clearTimeout(t);
  }
}

function format(n){
  return Number(n || 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function updatePortfolio(data){
  const m = data?.metrics || {};
  document.getElementById('balance').innerText = format(m.balance);
  document.getElementById('equity').innerText = format(m.equity);
  document.getElementById('pnl').innerText = format(m.unrealized_pnl);
}

function renderPositions(ps){
  const el = document.getElementById('positions');
  const rows = (ps || []).map(p => `
    <div class="pos">
      ${String(p.side).toUpperCase()} | ${format(p.size)} | ${Number(p.entry).toFixed(2)}
    </div>
  `).join('');
  el.innerHTML = rows || '<div class="pos">Aucune position ouverte</div>';
}

function refreshFromSnapshot(snap){
  if(!snap) return;
  updatePortfolio(snap);
  renderPositions(snap.positions || []);
  if(Array.isArray(snap.recent_events)){
    document.getElementById('events').innerHTML = snap.recent_events.map(e => `<div>${e}</div>`).join('');
  }
  if(snap.state && snap.last_price !== undefined){
    const localTs = snap.timestamp ? new Date(snap.timestamp).toLocaleString() : new Date().toLocaleString();
    const mode = (snap.data_mode || '').toUpperCase();
    setStatus(`Prêt • ${mode || 'REAL DATA'} • state=${snap.state} • last=${Number(snap.last_price).toFixed(2)} • ${localTs} • ${snap.learning_mode || ''}`);
  }
}

function setCandles(raw){
  candles.length = 0;
  (raw || []).forEach(c => {
    const b = parseCandle(c);
    if(b) candles.push(b);
  });
  drawCandles();
}

function appendCandle(candle){
  const b = parseCandle(candle);
  if(!b) return;
  const last = candles[candles.length - 1];
  if(last && last.time === b.time) candles[candles.length - 1] = b;
  else candles.push(b);
  while(candles.length > 600) candles.shift();
  drawCandles();
}

function updateChart(candle){
  appendCandle(candle);
}

async function waitUntilReady(maxWaitMs=45000){
  const start = performance.now();
  while(performance.now() - start < maxWaitMs){
    const st = await api('/api/init_status', {}, {method:'GET', timeoutMs:5000});
    renderDebug(st);
    if(st.status === 'ready') return st;
    setStatus(`Initialisation... ${st.phase || 'loading'}`);
    await new Promise(r => setTimeout(r, 600));
  }
  throw new Error('Timeout initialisation moteur');
}

async function init(){
  initAttempts += 1;
  setStatus('Chargement des données...');
  try {
    const st = await waitUntilReady();
    setStatus(`Moteur prêt (${st.fallback_used ? 'fallback' : 'datasets'}) • tf=${st.timeframe} • files=${st.files_detected} • ds=${st.datasets_retained}`);
    if(st.fallback_used){
      const b = document.getElementById('fallbackBanner');
      b.style.display = 'block';
      b.textContent = `FALLBACK MODE — ${st.error || st.fallback_reason || 'données réelles indisponibles'}`;
    } else {
      document.getElementById('fallbackBanner').style.display = 'none';
    }
    const t0 = performance.now();
    const d = await api('/api/init', {}, {timeoutMs:12000});
    const elapsed = Math.round(performance.now() - t0);

    if(!d.ok){
      renderDebug(d.debug || st, d.reason || 'Init non prête');
      setStatus('Init en attente...');
      return;
    }

    const incoming = d.initial_candles || [];
    setCandles(incoming);
    if(!incoming.length){
      logEvent('Init OK mais initial_candles vide');
    }
    refreshFromSnapshot(d.snapshot || d);
    renderDebug({...d.debug, init_request_ms: elapsed});
    logEvent(`Init terminé en ${elapsed}ms, candles=${incoming.length}`);
    if(!d.disable_auto_run) startMarket();
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    setStatus('Erreur d\'initialisation');
    renderDebug({}, message);
    logEvent(`Init ERROR: ${message}`);
  }
}

async function step(){
  try {
    const d = await api('/api/step');
    if(!d.ok){ logEvent(`STEP refusé: ${d.reason || 'unknown'}`); return; }
    updateChart(d.candle);
    refreshFromSnapshot(d.snapshot);
    const tr = d.market_structure?.trend || 'range';
    logEvent(`amd=${d.amd_phase || 'n/a'} phase=${d.phase || 'n/a'} trend=${tr} close=${Number(d.candle.close).toFixed(2)}`);
  } catch (err) {
    logEvent(`STEP ERROR: ${err.message || err}`);
  }
}

function startMarket(){
  if(running) return;
  running = true;
  timer = setInterval(step, 3000);
}

function stopMarket(){
  running = false;
  if(timer) clearInterval(timer);
}

document.getElementById('btnStep').onclick = step;
document.getElementById('btnRun').onclick = startMarket;
document.getElementById('btnPause').onclick = stopMarket;
document.getElementById('recenter').onclick = ()=>{ viewBars = 180; drawCandles(); };

document.getElementById('buy').onclick = async()=>{
  try {
    const size=Number(document.getElementById('size').value); const lev=Number(document.getElementById('leverage').value);
    const r=await api('/api/order',{side:'buy',size,leverage:lev});
    if(!r.ok){ logEvent(`BUY refusé: ${r.reason}`); return; }
    refreshFromSnapshot(r.snapshot);
    logEvent(`BUY fill=${r.execution.fill.toFixed(2)}`);
  } catch (err){ logEvent(`BUY ERROR: ${err.message || err}`); }
};
document.getElementById('sell').onclick = async()=>{
  try {
    const size=Number(document.getElementById('size').value); const lev=Number(document.getElementById('leverage').value);
    const r=await api('/api/order',{side:'sell',size,leverage:lev});
    if(!r.ok){ logEvent(`SELL refusé: ${r.reason}`); return; }
    refreshFromSnapshot(r.snapshot);
    logEvent(`SELL fill=${r.execution.fill.toFixed(2)}`);
  } catch (err){ logEvent(`SELL ERROR: ${err.message || err}`); }
};
document.getElementById('closeAll').onclick = async()=>{ try { const r=await api('/api/close',{fraction:1}); refreshFromSnapshot(r.snapshot); logEvent(`Close all realized=${r.realized.toFixed(2)}`);} catch(err){logEvent(`CLOSE ERROR: ${err.message || err}`);} };
document.getElementById('close10').onclick = async()=>{ try { const r=await api('/api/close',{fraction:0.1}); refreshFromSnapshot(r.snapshot); logEvent(`Close 10% realized=${r.realized.toFixed(2)}`);} catch(err){logEvent(`CLOSE10 ERROR: ${err.message || err}`);} };
document.getElementById('close20').onclick = async()=>{ try { const r=await api('/api/close',{fraction:0.2}); refreshFromSnapshot(r.snapshot); logEvent(`Close 20% realized=${r.realized.toFixed(2)}`);} catch(err){logEvent(`CLOSE20 ERROR: ${err.message || err}`);} };
document.getElementById('close50').onclick = async()=>{ try { const r=await api('/api/close',{fraction:0.5}); refreshFromSnapshot(r.snapshot); logEvent(`Close 50% realized=${r.realized.toFixed(2)}`);} catch(err){logEvent(`CLOSE50 ERROR: ${err.message || err}`);} };
document.getElementById('deposit').onclick = async()=>{ try { const amount=Number(document.getElementById('cashAmount').value); const r=await api('/api/deposit',{amount}); refreshFromSnapshot(r.snapshot); logEvent(`Deposit ${amount}`);} catch(err){logEvent(`DEPOSIT ERROR: ${err.message || err}`);} };
document.getElementById('withdraw').onclick = async()=>{ try { const amount=Number(document.getElementById('cashAmount').value); const r=await api('/api/withdraw',{amount}); refreshFromSnapshot(r.snapshot); logEvent(r.ok?`Withdraw ${amount}`:`Withdraw refusé (${r.reason})`);} catch(err){logEvent(`WITHDRAW ERROR: ${err.message || err}`);} };
document.getElementById('reset').onclick = async()=>{ try { const r=await api('/api/reset'); if(!r.ok){logEvent(`RESET refusé: ${r.reason}`); return;} candles.length=0; drawCandles(); refreshFromSnapshot(r.snapshot); logEvent('RESET TOTAL'); await init(); } catch(err){logEvent(`RESET ERROR: ${err.message || err}`);} };

window.addEventListener('resize', resizeCanvas);
resizeCanvas();
init();
