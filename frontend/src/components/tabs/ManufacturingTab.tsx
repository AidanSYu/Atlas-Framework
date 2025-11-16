import { useState } from 'react'
import '../../styles/ManufacturingTab.css'

interface Project {
  id: string
  name: string
  disease: string
}

interface ManufacturingTabProps {
  project: Project
}

interface ManufacturingResult {
  compound_name: string
  smiles: string | null
  analysis: any
}

export default function ManufacturingTab({ project }: ManufacturingTabProps) {
  const [compoundName, setCompoundName] = useState('')
  const [smiles, setSmiles] = useState('')
  const [compoundType, setCompoundType] = useState('small_molecule')
  const [complexity, setComplexity] = useState('moderate')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ManufacturingResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    if (!compoundName.trim()) {
      setError('Please enter a compound name')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/manufacturing/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          compound_name: compoundName,
          smiles: smiles || null,
          compound_type: compoundType,
          synthesis_complexity: complexity,
          disease_context: project.disease
        })
      })

      if (!response.ok) throw new Error(`Analysis failed: ${response.statusText}`)
      const data = await response.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze manufacturability')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="manufacturing-tab">
      <div className="tab-header">
        <h2>Manufacturing Analysis</h2>
        <p>Evaluate scalability, production costs, and regulatory considerations</p>
      </div>

      <div className="analysis-form">
        <div className="form-section">
          <h3>Compound Information</h3>
          
          <div className="form-group">
            <label>Compound Name *</label>
            <input
              type="text"
              value={compoundName}
              onChange={(e) => setCompoundName(e.target.value)}
              placeholder="e.g., Aspirin, Insulin"
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label>SMILES (Optional)</label>
            <input
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O"
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label>Compound Type</label>
            <select value={compoundType} onChange={(e) => setCompoundType(e.target.value)} className="form-select">
              <option value="small_molecule">Small Molecule</option>
              <option value="protein">Protein</option>
              <option value="antibody">Antibody</option>
              <option value="nucleic_acid">Nucleic Acid</option>
            </select>
          </div>

          <div className="form-group">
            <label>Synthesis Complexity</label>
            <select value={complexity} onChange={(e) => setComplexity(e.target.value)} className="form-select">
              <option value="simple">Simple</option>
              <option value="moderate">Moderate</option>
              <option value="complex">Complex</option>
            </select>
          </div>

          <div className="form-group">
            <label>Disease Context</label>
            <div className="context-display">{project.disease}</div>
          </div>

          <button onClick={handleAnalyze} disabled={loading} className="btn-analyze">
            {loading ? 'Analyzing...' : 'Analyze Manufacturability'}
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error"><strong>Error:</strong> {error}</div>}

      {result && (
        <div className="results-section">
          <div className="result-header">
            <h3>{result.compound_name}</h3>
            {result.smiles && <div className="smiles-display">{result.smiles}</div>}
          </div>

          <div className="results-grid">
            {result.analysis.scalability_score !== undefined && (
              <div className="result-card">
                <h4>Scalability Score</h4>
                <div className="score-large">
                  {result.analysis.scalability_score}<span>/100</span>
                </div>
              </div>
            )}

            {result.analysis.cost_estimate && (
              <div className="result-card">
                <h4>Cost Estimate</h4>
                <p>{result.analysis.cost_estimate}</p>
              </div>
            )}

            {result.analysis.production_timeline && (
              <div className="result-card">
                <h4>Production Timeline</h4>
                <p>{result.analysis.production_timeline}</p>
              </div>
            )}

            {result.analysis.manufacturability_risks && result.analysis.manufacturability_risks.length > 0 && (
              <div className="result-card full-width">
                <h4>Manufacturing Risks</h4>
                <ul className="risk-list">
                  {result.analysis.manufacturability_risks.map((risk: string, idx: number) => (
                    <li key={idx}>{risk}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.analysis.recommendations && result.analysis.recommendations.length > 0 && (
              <div className="result-card full-width">
                <h4>Recommendations</h4>
                <ul className="recommendation-list">
                  {result.analysis.recommendations.map((rec: string, idx: number) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.analysis.detailed_assessment && (
              <div className="result-card full-width">
                <h4>Detailed Assessment</h4>
                <p>{result.analysis.detailed_assessment}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
