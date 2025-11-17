import { useState } from 'react'
import '../../styles/RetrosynthesisTab.css'

interface Project {
  id: string
  name: string
  disease: string
}

interface RetrosynthesisTabProps {
  project: Project
}

interface RetrosynthesisResult {
  compound_name: string
  smiles: string | null
  analysis: any
}

export default function RetrosynthesisTab({ project }: RetrosynthesisTabProps) {
  const [compoundName, setCompoundName] = useState('')
  const [smiles, setSmiles] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<RetrosynthesisResult | null>(null)
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
      const response = await fetch('/api/retrosynthesis/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          compound_name: compoundName,
          smiles: smiles || null,
          disease_context: project.disease
        })
      })

      if (!response.ok) throw new Error(`Analysis failed: ${response.statusText}`)
      const data = await response.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze compound')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="retrosynthesis-tab">
      <div className="tab-header">
        <h2>Retrosynthetic Analysis</h2>
        <p>Analyze synthesis routes and molecular complexity for therapeutic compounds</p>
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
              placeholder="e.g., Aspirin, Metformin"
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
            <label>Disease Context</label>
            <div className="context-display">{project.disease}</div>
          </div>

          <button onClick={handleAnalyze} disabled={loading} className="btn-analyze">
            {loading ? 'Analyzing...' : 'Analyze Retrosynthesis'}
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error"><strong>Error:</strong> {error}</div>}

      {result && (
        <div className="results-section">
          <div className="result-header">
            <h3>{result.compound_name}</h3>
            {result.smiles && <div className="smiles-display">SMILES: {result.smiles}</div>}
          </div>

          <div className="results-grid">
            {result.analysis.molecular_properties && !result.analysis.molecular_properties.error && (
              <div className="result-card">
                <h4>Molecular Properties</h4>
                <div className="property-list">
                  {result.analysis.molecular_properties.molecular_weight !== undefined && (
                    <div className="property-item">
                      <span>Molecular Weight:</span>
                      <strong>{result.analysis.molecular_properties.molecular_weight.toFixed(2)} g/mol</strong>
                    </div>
                  )}
                  {result.analysis.molecular_properties.logp !== undefined && (
                    <div className="property-item">
                      <span>LogP:</span>
                      <strong>{result.analysis.molecular_properties.logp.toFixed(2)}</strong>
                    </div>
                  )}
                  {result.analysis.molecular_properties.h_bond_donors !== undefined && (
                    <div className="property-item">
                      <span>H-bond Donors:</span>
                      <strong>{result.analysis.molecular_properties.h_bond_donors}</strong>
                    </div>
                  )}
                  {result.analysis.molecular_properties.h_bond_acceptors !== undefined && (
                    <div className="property-item">
                      <span>H-bond Acceptors:</span>
                      <strong>{result.analysis.molecular_properties.h_bond_acceptors}</strong>
                    </div>
                  )}
                  {result.analysis.molecular_properties.aromatic_rings !== undefined && (
                    <div className="property-item">
                      <span>Aromatic Rings:</span>
                      <strong>{result.analysis.molecular_properties.aromatic_rings}</strong>
                    </div>
                  )}
                  {result.analysis.molecular_properties.complexity_score !== undefined && (
                    <div className="property-item">
                      <span>Complexity Score:</span>
                      <strong>{result.analysis.molecular_properties.complexity_score.toFixed(1)}/100</strong>
                    </div>
                  )}
                </div>
              </div>
            )}

            {result.analysis.functional_groups && result.analysis.functional_groups.length > 0 && (
              <div className="result-card">
                <h4>Functional Groups</h4>
                <div className="functional-groups">
                  {result.analysis.functional_groups.map((group: string, idx: number) => (
                    <span key={idx} className="functional-group-badge">{group}</span>
                  ))}
                </div>
              </div>
            )}

            {result.analysis.retrosynthetic_routes && (
              <div className="result-card full-width">
                <h4>Retrosynthetic Analysis</h4>
                <div className="retro-analysis">
                  <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', lineHeight: '1.6' }}>
                    {result.analysis.retrosynthetic_routes}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
