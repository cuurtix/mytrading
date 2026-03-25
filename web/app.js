let running = false;
let timer = null;
const labels = [];
const prices = [];

const chart = new Chart(document.getElementById('priceChart'), {
  type: 'line',
  data: { labels, datasets: [{ label: 'XAUUSD', data: prices, borderColor: '#5eead4' }] },
  options: { animation: false, scales: { x: { display:false } } }
});

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

async function step(){
  const d = await api('/api/step');
  if(!d.ok) return;
  const c = d.candle;
  labels.push(c.datetime);
  prices.push(c.close);
  if(labels.length>300){labels.shift(); prices.shift();}
  chart.update();
  refreshFromSnapshot(d.snapshot);
  logEvent(`state=${d.state} close=${c.close.toFixed(2)} sweep=${d.sweep.recent_sweep_flag}`);
}

async function init(){
  const d = await api('/api/init');
  document.getElementById('status').textContent = `Prêt - timeframe ${d.timeframe}`;
  labels.length = 0; prices.length = 0;
  (d.initial_candles || []).forEach(c => { labels.push(c.datetime); prices.push(c.close); });
  chart.update();
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
document.getElementById('reset').onclick = async()=>{ const r=await api('/api/reset'); labels.length=0; prices.length=0; chart.update(); refreshFromSnapshot(r.snapshot); logEvent('RESET TOTAL'); await init(); };

init();
