const $ = (sel) => document.querySelector(sel);
const api = (path, params={}) => {
  const qs = new URLSearchParams(params).toString();
  return fetch(`${path}${qs?`?${qs}`:''}`).then(r=>r.json());
};

// ====================== Warp Bubble ======================
async function renderWarp(){
  const E = +$('#E').value, rho = +$('#rho').value;
  const classical = $('#mode').value === 'class';
  const data = await api('/api/warp', {E, rho, classical});

  // Ellipsoid lines
  const ellTraces = data.ellipsoid_lines.map(l=>({
    type:'scatter3d', mode:'lines', x:l.x, y:l.y, z:l.z,
    line:{color:'#2a3758', width:2}, hoverinfo:'skip', showlegend:false
  }));
  // Power lines
  const pTraces = data.power_lines.map(l=>({
    type:'scatter3d', mode:'lines', x:l.x, y:l.y, z:l.z,
    line:{color:'orangered', width:3}, opacity:0.9, name:'Power line'
  }));
  // Trajectories
  const trajLineTraces = [];
  const qubitTraces = [];
  data.trajectories.forEach((t,i)=>{
    // Color by vfrac using per-point colors; Plotly supports a colorscale with c array
    trajLineTraces.push({
      type:'scatter3d', mode:'lines', x:t.x, y:t.y, z:t.z,
      line:{width:4, color:t.vfrac, colorscale:'Plasma'},
      opacity:0.95, name:`Trajectory ${i+1}`
    });
    qubitTraces.push({
      type:'scatter3d', mode:'markers', x:t.qubits.x, y:t.qubits.y, z:t.qubits.z,
      marker:{color:'goldenrod', size:t.qubits.size}, name:'Qubits'
    });
  });

  const layout = {
    scene:{xaxis:{title:'x'}, yaxis:{title:'y'}, zaxis:{title:'z'}},
    margin:{l:0,r:0,t:10,b:0},
    showlegend:true
  };
  Plotly.newPlot('warp3d', [...ellTraces, ...pTraces, ...trajLineTraces, ...qubitTraces], layout);
}

// ====================== Thruster + Wormhole ======================
async function renderThruster(jumpHighlight=false){
  const Q = +$('#Q').value, R = +$('#R').value;
  const data = await api('/api/thruster', {Q, R});
  const x = data.x;
  const traces = [
    {x, y:data.F, type:'scatter', mode:'lines', name:'QVT thrust', line:{color:'#33cc99'}},
    {x, y:data.xn, type:'scatter', mode:'lines', name:'x_n', line:{color:'#888'}},
    {x, y:data.xnp1, type:'scatter', mode:'lines', name:'x_{n+1}', line:{color:'#bbb', dash:'dot'}},
    {x, y:data.M, type:'scatter', mode:'lines', name:'Gear', line:{color:'#4ea1ff'}}
  ];
  // Wormhole schematic overlays as separate traces sharing same axes
  const deco = [
    {x:[data.wormhole_a.x], y:[data.wormhole_a.y], type:'scatter', mode:'markers', name:'Wormhole A', marker:{color:'lime', size:10, symbol:'circle'}},
    {x:[data.wormhole_b.x], y:[data.wormhole_b.y], type:'scatter', mode:'markers', name:'Wormhole B', marker:{color:'deepskyblue', size:10, symbol:'circle'}},
    {x:[data.sun.x], y:[data.sun.y], type:'scatter', mode:'markers', name:'Sun', marker:{color:'yellow', size:14, symbol:'star'}},
    {x:[data.ship.x], y:[data.ship.y], type:'scatter', mode:'markers', name:'Ship', marker:{color: jumpHighlight?'magenta':'red', size:12, symbol:'triangle-up'}}
  ];
  // Connection line
  deco.push({x:[data.wormhole_a.x, data.wormhole_b.x], y:[data.wormhole_a.y, data.wormhole_b.y], type:'scatter', mode:'lines', name:'Parallel Universe', line:{color:'purple', dash:'dash'}});

  const layout = {
    xaxis:{title:'Input / x'}, yaxis:{title:'Value / Position'},
    margin:{l:40,r:10,t:10,b:40}, legend:{orientation:'h'}
  };
  Plotly.newPlot('thruster', [...traces, ...deco], layout);
}

// ====================== Time Graphs ======================
async function renderTime(){
  const data = await api('/api/timegraphs');
  // 3D paths
  const e = data.earth, m = data.moon;
  const t3d = [
    {type:'scatter3d', mode:'lines', x:e.x, y:e.y, z:e.z, name:'Earth', line:{color:'#66aaff', width:4}},
    {type:'scatter3d', mode:'lines', x:m.x, y:m.y, z:m.z, name:'Moon', line:{color:'#ff6666', width:4}},
    {type:'scatter3d', mode:'markers', x:e.x.filter((_,i)=>e.dc.includes(i)), y:e.y.filter((_,i)=>e.dc.includes(i)), z:e.z.filter((_,i)=>e.dc.includes(i)), name:'Earth DC', marker:{color:'#ffffff', size:3}},
    {type:'scatter3d', mode:'markers', x:m.x.filter((_,i)=>m.dc.includes(i)), y:m.y.filter((_,i)=>m.dc.includes(i)), z:m.z.filter((_,i)=>m.dc.includes(i)), name:'Moon DC', marker:{color:'#ffff00', size:3}},
  ];
  Plotly.newPlot('time3d', t3d, {scene:{xaxis:{title:'X'},yaxis:{title:'Y'},zaxis:{title:'Z'}}, margin:{l:0,r:0,t:10,b:0}});

  // Invariance plots
  Plotly.newPlot('invSpace', [{y:data.inv_space, type:'scatter', mode:'lines+markers', name:'ΔTime vs Space', line:{color:'goldenrod'}}], {yaxis:{title:'Time Invariance'}, xaxis:{title:'Segment index'}, margin:{l:40,r:10,t:10,b:30}});
  Plotly.newPlot('invTime',  [{y:data.inv_time,  type:'scatter', mode:'lines', name:'ΔTime vs Step', line:{color:'#ff9f43'}}], {yaxis:{title:'Stepwise Δ'}, xaxis:{title:'Step'}, margin:{l:40,r:10,t:10,b:30}});
}

// ====================== CTC ======================
async function renderCTC(){
  const data = await api('/api/ctc');
  const cylR = data.r_ctc;
  const fam = data.family;
  const traces3d = fam.map((ray,i)=>({
    type:'scatter3d', mode:'lines', x:ray.x, y:ray.y, z:ray.t, name:`L=${ray.L.toFixed(1)}`, line:{width:3}, opacity:0.95
  }));
  // Cylinder wire (approx)
  const theta = Array.from({length:64}, (_,k)=>k*2*Math.PI/63);
  const z = Array.from({length:32},(_,k)=>-5 + k*(10/31));
  const cyl = [];
  z.forEach(zz=>{
    cyl.push({type:'scatter3d', mode:'lines', x:theta.map(th=>cylR*Math.cos(th)), y:theta.map(th=>cylR*Math.sin(th)), z:theta.map(_=>zz), line:{color:'#aaaaaa'}, opacity:.25, hoverinfo:'skip', showlegend:false});
  });
  Plotly.newPlot('ctc3d', [...traces3d, ...cyl], {scene:{xaxis:{title:'x'}, yaxis:{title:'y'}, zaxis:{title:'t'}}, margin:{l:0,r:0,t:10,b:0}});

  // Top‑down
  const top = fam.map((ray,i)=>({type:'scatter', mode:'lines', x:ray.x, y:ray.y, name:`L=${ray.L.toFixed(1)}`, line:{width:2}}));
  top.push({type:'scatter', mode:'lines', x:theta.map(th=>cylR*Math.cos(th)), y:theta.map(th=>cylR*Math.sin(th)), name:'CTC radius', line:{color:'#999', dash:'dot'}});
  Plotly.newPlot('ctc2d', top, {xaxis:{title:'x'}, yaxis:{title:'y', scaleanchor:'x', scaleratio:1}, margin:{l:40,r:10,t:10,b:30}});
}

// ====================== Wiring ======================
function wireControls(){
  ['E','rho','mode'].forEach(id=> $('#'+id).addEventListener('input', renderWarp));
  ['Q','R'].forEach(id=> $('#'+id).addEventListener('input', ()=>renderThruster(false)));
  $('#jump').addEventListener('click', ()=>renderThruster(true));
}

async function bootstrap(){
  wireControls();
  await renderWarp();
  await renderThruster(false);
  await renderTime();
  await renderCTC();
}

bootstrap();
