const stateClass = (v='') => `state-${String(v).toLowerCase().replace('connected_', '').replace(/[^a-z_]/g,'')}`;
const fmt = (v) => v ? new Date(v).toLocaleString() : '-';
const pct = (v) => (v==null||isNaN(v)) ? '-' : `${Math.round(v*100)}%`;
const signalCols = ['timestamp','ticker','action_permission','best_action','confidence','fragility','size_guidance','one_line_reason','cooldown_status','emitted'];
let currentReplayRunId = null;

function showBanner(targetId, msg, cls='warn') { document.getElementById(targetId).innerHTML = `<div class="banner ${cls}">${msg}</div>`; }
function drawTable(elId, rows) {
  const el = document.getElementById(elId);
  if (!rows || !rows.length) { el.innerHTML = '<tr><td class="muted">No rows available.</td></tr>'; return; }
  const cols = [...new Set(signalCols.concat(Object.keys(rows[0])))].slice(0, 14);
  el.innerHTML = `<thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>` + rows.map(r => `<tr data-row='${JSON.stringify(r).replace(/'/g, '&apos;')}'>${cols.map(c=>`<td>${typeof r[c]==='number'? (c.includes('confidence')||c.includes('fragility')||c.includes('score') ? pct(r[c]) : (r[c].toFixed?.(3) ?? r[c])) : (r[c] ?? '')}</td>`).join('')}</tr>`).join('') + '</tbody>';
  el.querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => openDetail(JSON.parse(tr.dataset.row.replaceAll('&apos;', "'"))));
}
function openDetail(row) {
  const detail = document.getElementById('detail');
  const invalidations = (row.invalidation_conditions || []).map(x=>`<li>${x}</li>`).join('') || '<li>None listed</li>';
  detail.innerHTML = `<div class="controls" style="justify-content:space-between"><h2>Signal Detail</h2><button onclick="document.getElementById('detail').classList.remove('open')">Close</button></div><div class="grid grid-2"><div><div class="label">Timestamp</div><div class="val">${row.timestamp || '-'}</div><div class="label">Ticker</div><div class="val">${row.ticker || '-'}</div><div class="label">Permission / Action</div><div class="val">${row.action_permission || '-'} / ${row.best_action || '-'}</div><div class="label">Reason</div><div class="val">${row.one_line_reason || '-'}</div></div><div><div class="label">Confidence</div><div class="val">${pct(row.confidence)}</div><div class="label">Fragility</div><div class="val">${pct(row.fragility)}</div><div class="label">Validity score</div><div class="val">${pct(row.validity_score)}</div><div class="label">Size guidance</div><div class="val">${row.size_guidance || '-'}</div></div></div><div class="panel"><h2>Rationale & diagnostics</h2><div class="val">Market state: ${row.market_state || '-'} | Transition: ${row.transition_state || '-'}</div><div class="val">Structural drift: ${pct(row.structural_drift_score)} | Instability: ${pct(row.latest_instability)}</div><div class="val">Cooldown/suppression: ${row.cooldown_status || 'not suppressed'} | Emitted: ${String(row.emitted ?? true)}</div></div><div class="panel"><h2>Invalidation conditions</h2><ul>${invalidations}</ul></div><details class="panel"><summary>Advanced / debug</summary><pre>${JSON.stringify(row, null, 2)}</pre></details>`;
  detail.classList.add('open');
}
async function jget(url, opts={}) { const res = await fetch(url, opts); const data = await res.json().catch(()=>({detail:'Invalid response'})); if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`); return data; }

async function refreshAll() {
  const [summary, status, provider] = await Promise.all([
    jget('/operator/summary'),
    jget('/live/status'),
    jget('/integrations/massive/status').catch(e=>({status:'error', error:e.message}))
  ]);
  document.getElementById('providerPill').textContent = `Provider: ${provider.status || 'unknown'}`;
  document.getElementById('sessionPill').textContent = `State: ${status.session_state}`;
  document.getElementById('sessionPill').className = `pill ${stateClass(status.session_state)}`;
  document.getElementById('eventPill').textContent = `Last event: ${fmt(status.last_event_at)}`;
  document.getElementById('signalPill').textContent = `Last signal: ${fmt(status.last_signal_at)}`;
  document.getElementById('statusStrip').innerHTML = `<div class="controls"><span class="pill ${stateClass(status.session_state)}">${status.session_state}</span><span class="pill">Readiness ${status.readiness_state}</span><span class="pill">Warmup ${pct(status.warmup_progress)} (${status.bars_collected}/${status.bars_required})</span><span class="pill">Timeframe ${status.timeframe}</span></div>`;
  document.getElementById('sessionSummary').innerHTML = `<div>Buffered symbols: ${status.buffered_symbol_count}</div><div>Suppressed: ${status.suppressed_count} | Abstain: ${status.abstain_count}</div><div>Latest error: ${status.latest_error || 'none'}</div>`;
  document.getElementById('warnings').innerHTML = (summary.recent_warnings||[]).map(w=>`<div class="banner warn">${w.error || JSON.stringify(w)}</div>`).join('') || '<div class="muted">No recent warnings.</div>';
  drawTable('latestTable', (summary.latest_signals||[]));
  const liveLatest = await jget('/live/signals/latest').catch(()=>({signals:[]}));
  drawTable('liveSignals', liveLatest.signals || []);
  document.getElementById('providerStatus').innerHTML = `<div>Status: ${provider.status}</div><div>Config present: ${provider.config_present}</div><div>API key: ${provider.api_key_present ? (provider.api_key_valid ? 'valid' : 'present/invalid') : 'missing'}</div><div>REST reachable: ${provider.rest_reachable}</div><div>WebSocket configured: ${provider.websocket_configured}</div><div>Recent live event: ${fmt(provider.recent_live_event_at)}</div><div>Error: ${provider.error || 'none'}</div>`;
  const datasets = await jget('/integrations/massive/datasets').catch(()=>({datasets:[]}));
  const ds = datasets.datasets || [];
  const dsSelect = document.getElementById('datasetSelect');
  dsSelect.innerHTML = `<option value="">Latest configured data_dir</option>` + ds.map(x=>`<option value="${x.dataset_path}">${x.dataset_id} | ${x.timeframe} | ${x.symbols.join(',')}</option>`).join('');
  document.getElementById('datasets').innerHTML = ds.length ? `<table><thead><tr><th>Dataset</th><th>Symbols</th><th>Timeframe</th><th>Range</th><th>Path</th></tr></thead><tbody>${ds.map(d=>`<tr><td>${d.dataset_id}</td><td>${d.symbols.join(', ')}</td><td>${d.timeframe}</td><td>${d.start_date} → ${d.end_date}</td><td>${d.dataset_path}</td></tr>`).join('')}</tbody></table>` : '<div class="banner warn">No cached datasets found.</div>';
}

async function loadHistory() {
  const q = new URLSearchParams({limit:'300'});
  for (const [id,key] of [['histTicker','ticker'],['histSession','session_type'],['histPermission','action_permission'],['histBestAction','best_action'],['histStart','start_at'],['histEnd','end_at']]) { const val = document.getElementById(id).value.trim(); if (val) q.set(key,val); }
  if (document.getElementById('histSuppressed').checked) q.set('include_suppressed','true');
  const data = await jget(`/signals/history?${q}`);
  drawTable('historyTable', data.signals || []);
  document.getElementById('historySummary').textContent = `Suppressed: ${data.summary?.suppression_count || 0} | Abstentions: ${data.summary?.abstention_count || 0}`;
}

document.querySelectorAll('.nav-btn').forEach(btn => btn.onclick = () => { document.querySelectorAll('.nav-btn').forEach(x=>x.classList.remove('active')); btn.classList.add('active'); document.querySelectorAll('.page').forEach(p=>p.classList.remove('active')); document.getElementById(`page-${btn.dataset.page}`).classList.add('active'); });
document.getElementById('startBtn').onclick = async () => { const symbols = document.getElementById('symbolsInput').value.split(',').map(s=>s.trim()).filter(Boolean); const timeframe = document.getElementById('timeframeInput').value; try { await jget('/live/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({symbols, timeframe})}); showBanner('commandMessages', 'Live session started.', ''); } catch (e) { showBanner('commandMessages', e.message, 'err'); } refreshAll(); };
document.getElementById('stopBtn').onclick = async () => { await jget('/live/stop', {method:'POST'}); refreshAll(); };
document.getElementById('fetchBtn').onclick = async () => { try { const body = {symbols: document.getElementById('fetchSymbols').value.split(',').map(s=>s.trim()).filter(Boolean), timeframe: document.getElementById('fetchTimeframe').value, start_date: document.getElementById('fetchStart').value, end_date: document.getElementById('fetchEnd').value}; await jget('/integrations/massive/historical/fetch', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}); showBanner('replayMessages', 'Historical data fetched and cached.', ''); } catch (e) { showBanner('replayMessages', e.message, 'err'); } refreshAll(); };
document.getElementById('runReplayBtn').onclick = async () => { const params = new URLSearchParams({timeframe: document.getElementById('fetchTimeframe').value}); const selected = document.getElementById('datasetSelect').value; const manualDir = document.getElementById('replayDataDir').value.trim(); if (manualDir) params.set('data_dir', manualDir); else if (selected) params.set('data_dir', selected); try { const out = await jget(`/run-replay?${params}`, {method:'POST'}); currentReplayRunId = out.run_id; document.getElementById('replayMeta').innerHTML = `Run ${out.run_id} | ${out.meta.timeframe} | ${out.meta.signal_count} signals`; drawTable('replayTable', out.replay); } catch (e) { showBanner('replayMessages', e.message, 'err'); } };
document.getElementById('exportReplayBtn').onclick = () => { if (!currentReplayRunId) { showBanner('replayMessages', 'Run replay first to export CSV.', 'warn'); return; } window.location.href = `/replay/runs/${currentReplayRunId}/export`; };
document.getElementById('loadHistoryBtn').onclick = loadHistory;
setInterval(refreshAll, 15000);refreshAll();
