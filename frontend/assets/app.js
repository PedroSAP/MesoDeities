async function postJSON(url, data) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data) });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
const qEl=document.getElementById('q'), langEl=document.getElementById('lang'), askBtn=document.getElementById('askBtn');
const answerEl=document.getElementById('answer'), suggEl=document.getElementById('suggestions'), statusEl=document.getElementById('status');
function setStatus(m){ statusEl.textContent=m||'' } function showAnswer(t){ answerEl.textContent=t||'' }
function showSuggestions(list){ if(!list||!list.length){suggEl.innerHTML='';return} const pills=list.map(i=>`<span class="pill">${i.name} · ${i.domain||''} (${i.score.toFixed(3)})</span>`).join(''); suggEl.innerHTML=`<div><strong>Suggestions:</strong> ${pills}</div>` }
async function ask(){ const question=qEl.value.trim(); const lang=langEl.value; if(!question) return; setStatus('Thinking… (first query may take longer while models load)'); showAnswer(''); showSuggestions([]); askBtn.disabled=true;
  try{ const data=await postJSON('/api/query',{question,lang}); showAnswer(data.answer); showSuggestions(data.suggestions); setStatus('') }catch(err){ setStatus('Error: '+err.message) }finally{ askBtn.disabled=false } }
askBtn.addEventListener('click',ask); qEl.addEventListener('keydown',e=>{ if(e.key==='Enter') ask() }); fetch('/health').catch(()=>{});
