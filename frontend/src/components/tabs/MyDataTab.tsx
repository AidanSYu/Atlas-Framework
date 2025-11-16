import { useState } from 'react'
import '../../styles/MyDataTab.css'

interface Project {
  id: string
  name: string
  disease: string
}

interface MyDataTabProps {
  project: Project
}

export default function MyDataTab({ project }: MyDataTabProps) {
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    // TODO: Handle file upload
    const files = Array.from(e.dataTransfer.files)
    console.log('Files dropped:', files)
  }

  return (
    <div className="mydata-tab">
      <div className="tab-header">
        <h2>My Data</h2>
        <p>Upload and manage your research data, spectra, and experimental results</p>
      </div>

      <div className="data-content">
        <div 
          className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <h3>Drop files here or click to upload</h3>
          <p>Supported formats: PDF, Excel, CSV, Images (NMR, Mass Spec), Word documents</p>
          <button className="btn btn-primary">Browse Files</button>
        </div>

        <div className="data-categories">
          <div className="category-card">
            <div className="category-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h4>Spectral Data</h4>
            <p className="category-count">0 files</p>
            <span className="category-badge">NMR, Mass Spec, IR</span>
          </div>

          <div className="category-card">
            <div className="category-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h4>Trial Data</h4>
            <p className="category-count">0 files</p>
            <span className="category-badge">Excel, CSV</span>
          </div>

          <div className="category-card">
            <div className="category-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </div>
            <h4>Lab Notes</h4>
            <p className="category-count">0 files</p>
            <span className="category-badge">PDF, Word</span>
          </div>

          <div className="category-card">
            <div className="category-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h4>Images</h4>
            <p className="category-count">0 files</p>
            <span className="category-badge">JPG, PNG, TIFF</span>
          </div>
        </div>

        <div className="ai-insights-preview">
          <h3>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            AI Insights
          </h3>
          <p>Upload data to receive AI-powered analysis and suggestions</p>
        </div>
      </div>
    </div>
  )
}
