let running = false;
let timer = null;

const chartContainer = document.getElementById('priceChart');
const chart = LightweightCharts.createChart(chartContainer, {
  layout: { background: { color: '#1b2140' }, textColor: '#e8ecff' },
  rightPriceScale: { borderColor: '#30365a' },
  timeScale: { borderColor: '#30365a', timeVisible: true },
  grid: { vertLines: { color: '#252b4a' }, horzLines: { color: '#252b4a' } },
  crosshair: { mode: 0 },
});
const candleSeries = chart.addCandlestickSeries({
  upColor: '#2ecc71',
  downColor: '#e74c3c',
  wickUpColor: '#2ecc71',
  wickDownColor: '#e74c3c',
  borderVisible: false,
});

window.addEventListener('resize', () => {
  chart.applyOptions({ width: chartContainer.clientWidth, height: 420 });
});
chart.applyOptions({ width: chartContainer.clientWidth, height: 420 });

const candles = [];

function toBar(c) {
  return {
    time: Math.floor(new Date(c.datetime).getTime() / 1000),
    open: Number(c.open),
    high: Number(c.high),
    low: Number(c.low),
    close: Number(c.close),
  };
}

function logEvent(msg){
  const el = document.getElementById('events');
  el.innerHTML = `<div>${new Date().toLocaleTimeString()} - ${msg}</div>` + el.innerHTML;
}

async function api(path, payload={}){
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  return await r.json();
}

function renderMetrics(m){
  document.getElementById('metrics').textContent = JSON.stringify(m, null, 2);
}

function renderPositions(ps){
  document.getElementById('positions').innerHTML = ps.map(p => `#${p.id} ${p.side} size=${p.size.toFixed(2)} entry=${p.entry.toFixed(2)} lev=${p.leverage}`).join('<br/>') || 'Aucune';
}

function refreshFromSnapshot(snap){
  if(!snap) return;
  renderMetrics(snap.metrics);
  renderPositions(snap.positions || []);
  if(Array.isArray(snap.recent_events)){
    const el = document.getElementById('events');
    el.innerHTML = snap.recent_events.map(e => `<div>${e}</div>`).join('');
  }
  if(snap.state && snap.last_price !== undefined){
    document.getElementById('status').textContent = `state=${snap.state} last=${Number(snap.last_price).toFixed(2)}`;
  }
}

function appendCandle(candle){
  const bar = toBar(candle);
  const last = candles[candles.length - 1];
  if(last && last.time === bar.time){
    candles[candles.length - 1] = bar;
  } else {
    candles.push(bar);
  }
  const trimmed = candles.slice(-400);
  candles.length = 0;
  candles.push(...trimmed);
  candleSeries.setData(candles);
}

async function step(){
  const d = await api('/api/step');
  if(!d.ok) return;
  appendCandle(d.candle);
  refreshFromSnapshot(d.snapshot);
  logEvent(`state=${d.state} close=${Number(d.candle.close).toFixed(2)} sweep=${d.sweep.recent_sweep_flag}`);
}

async function init(){
  const d = await api('/api/init');
  document.getElementById('status').textContent = `Prêt - timeframe ${d.timeframe}`;
  candles.length = 0;
  (d.initial_candles || []).forEach(c => candles.push(toBar(c)));
  candleSeries.setData(candles);
  chart.timeScale().fitContent();
  refreshFromSnapshot(d.snapshot || d);
}

function runLoop(){
  if(!running) return;
  step();
  timer = setTimeout(runLoop, Number(document.getElementById('speed').value || 500));
}

document.getElementById('btnStep').onclick = step;
document.getElementById('btnRun').onclick = ()=>{ running=true; runLoop(); };
document.getElementById('btnPause').onclick = ()=>{ running=false; if(timer) clearTimeout(timer); };
document.getElementById('recenter').onclick = ()=> chart.timeScale().fitContent();

document.getElementById('buy').onclick = async()=>{
  const size=Number(document.getElementById('size').value); const lev=Number(document.getElementById('leverage').value);
  const r=await api('/api/order',{side:'buy',size,leverage:lev});
  refreshFromSnapshot(r.snapshot);
  logEvent(r.ok?`BUY fill=${r.execution.fill.toFixed(2)}`:`BUY refusé: ${r.reason}`);
};
document.getElementById('sell').onclick = async()=>{
  const size=Number(document.getElementById('size').value); const lev=Number(document.getElementById('leverage').value);
  const r=await api('/api/order',{side:'sell',size,leverage:lev});
  refreshFromSnapshot(r.snapshot);
  logEvent(r.ok?`SELL fill=${r.execution.fill.toFixed(2)}`:`SELL refusé: ${r.reason}`);
};
document.getElementById('closeAll').onclick = async()=>{ const r=await api('/api/close',{fraction:1}); refreshFromSnapshot(r.snapshot); logEvent(`Close all realized=${r.realized.toFixed(2)}`); };
document.getElementById('close10').onclick = async()=>{ const r=await api('/api/close',{fraction:0.1}); refreshFromSnapshot(r.snapshot); logEvent(`Close 10% realized=${r.realized.toFixed(2)}`); };
document.getElementById('close20').onclick = async()=>{ const r=await api('/api/close',{fraction:0.2}); refreshFromSnapshot(r.snapshot); logEvent(`Close 20% realized=${r.realized.toFixed(2)}`); };
document.getElementById('close50').onclick = async()=>{ const r=await api('/api/close',{fraction:0.5}); refreshFromSnapshot(r.snapshot); logEvent(`Close 50% realized=${r.realized.toFixed(2)}`); };
document.getElementById('deposit').onclick = async()=>{ const amount=Number(document.getElementById('cashAmount').value); const r=await api('/api/deposit',{amount}); refreshFromSnapshot(r.snapshot); logEvent(`Deposit ${amount}`); };
document.getElementById('withdraw').onclick = async()=>{ const amount=Number(document.getElementById('cashAmount').value); const r=await api('/api/withdraw',{amount}); refreshFromSnapshot(r.snapshot); logEvent(r.ok?`Withdraw ${amount}`:`Withdraw refusé (${r.reason})`); };
document.getElementById('reset').onclick = async()=>{ const r=await api('/api/reset'); candles.length=0; candleSeries.setData([]); refreshFromSnapshot(r.snapshot); logEvent('RESET TOTAL'); await init(); };

init();
