import { useState } from 'react'
import './App.css'

async function api<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, { headers: { 'Content-Type': 'application/json' }, ...opts })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export default function App() {
  const [disease, setDisease] = useState('Type 2 diabetes')
  const [research, setResearch] = useState<any>(null)
  const [compoundName, setCompoundName] = useState('aspirin')
  const [compoundSmiles, setCompoundSmiles] = useState('')
  const [compoundAnalysis, setCompoundAnalysis] = useState<any>(null)
  const [integratedDisease, setIntegratedDisease] = useState('Type 2 diabetes')
  const [integratedResults, setIntegratedResults] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [researchLoading, setResearchLoading] = useState(false)
  const [compoundLoading, setCompoundLoading] = useState(false)

  return (
    <div style={{ padding: 16, maxWidth: 960, margin: '0 auto', fontFamily: 'sans-serif' }}>
      <h1>Drug Dev Agents</h1>

      <section>
        <h2>Disease Research</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <input value={disease} onChange={(e)=>setDisease(e.target.value)} placeholder="Disease to research" style={{ flex: 1 }} />
          <button 
            disabled={researchLoading}
            onClick={async()=>{
              setResearchLoading(true);
              try {
                setResearch(await api('/researcher/research',{method:'POST', body: JSON.stringify({disease})}));
              } finally {
                setResearchLoading(false);
              }
            }}>
            {researchLoading ? 'Researching...' : 'Research'}
          </button>
        </div>
        {research && (
          <div style={{ marginTop: 8 }}>
            <pre style={{ whiteSpace: 'pre-wrap' }}>{research.summary}</pre>
            <div>Sources:</div>
            <ul>
              {research.sources.map((s:string,i:number)=>(<li key={i}><a href={s} target="_blank" rel="noopener noreferrer">{s}</a></li>))}
            </ul>
          </div>
        )}
      </section>

      <section>
        <h2>Compound Synthesis & Manufacturability Analysis</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <input value={compoundName} onChange={(e)=>setCompoundName(e.target.value)} placeholder="Compound name" style={{ flex: 1 }} />
          <input value={compoundSmiles} onChange={(e)=>setCompoundSmiles(e.target.value)} placeholder="SMILES (optional)" style={{ flex: 1 }} />
          <button 
            disabled={compoundLoading}
            onClick={async()=>{
              setCompoundLoading(true);
              try {
                setCompoundAnalysis(await api('/synthesis/analyze',{method:'POST', body: JSON.stringify({compound_name: compoundName, smiles: compoundSmiles || undefined})}));
              } finally {
                setCompoundLoading(false);
              }
            }}>
            {compoundLoading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
        {compoundAnalysis && (
          <div style={{ marginTop: 8 }}>
            <h3>{compoundAnalysis.compound_name} ({compoundAnalysis.molecule_type})</h3>
            {compoundAnalysis.smiles && <div><strong>SMILES:</strong> {compoundAnalysis.smiles}</div>}
            <h4>Synthesis Analysis</h4>
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.9em' }}>{JSON.stringify(compoundAnalysis.synthesis_analysis, null, 2)}</pre>
            <h4>Manufacturability</h4>
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.9em' }}>{JSON.stringify(compoundAnalysis.manufacturability, null, 2)}</pre>
            <h4>Summary</h4>
            <p style={{ whiteSpace: 'pre-wrap' }}>{compoundAnalysis.integrated_summary}</p>
          </div>
        )}
      </section>

      <section>
        <h2>Integrated Research & Manufacturing</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <input value={integratedDisease} onChange={(e)=>setIntegratedDisease(e.target.value)} placeholder="Disease to research" style={{ flex: 1 }} />
          <button 
            disabled={loading}
            onClick={async()=>{
              setLoading(true);
              try {
                setIntegratedResults(await api('/integrated/research-and-manufacture',{method:'POST', body: JSON.stringify({disease: integratedDisease})}));
              } finally {
                setLoading(false);
              }
            }}>
            {loading ? 'Processing...' : 'Research & Analyze'}
          </button>
        </div>
        {integratedResults && (
          <div style={{ marginTop: 8 }}>
            <h3>Research Summary for {integratedResults.disease}</h3>
            <pre style={{ whiteSpace: 'pre-wrap' }}>{integratedResults.research_summary}</pre>
            <div>Sources:</div>
            <ul>
              {integratedResults.research_sources.map((s:string,i:number)=>(<li key={i}><a href={s} target="_blank" rel="noopener noreferrer">{s}</a></li>))}
            </ul>
            <h3>Compound Analyses</h3>
            {integratedResults.compound_analyses.map((ca:any,i:number)=>(
              <div key={i} style={{ marginBottom: 16, padding: 12, border: '1px solid #ddd', borderRadius: 4 }}>
                <h4>{ca.compound_name} ({ca.molecule_type})</h4>
                {ca.smiles && <div><strong>SMILES:</strong> {ca.smiles}</div>}
                <div><strong>Synthesis feasibility:</strong> {ca.synthesis_analysis.feasibility_score}/10</div>
                <div><strong>Manufacturability score:</strong> {ca.manufacturability.score}/10</div>
                <p style={{ whiteSpace: 'pre-wrap', marginTop: 8 }}>{ca.integrated_summary}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      <p style={{marginTop:24, color:'#666'}}>All outputs are conceptual placeholders and not lab instructions.</p>
    </div>
  )
}
