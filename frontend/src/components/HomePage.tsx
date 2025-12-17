import { useState } from 'react'
import '../styles/HomePage.css'

interface Project {
  id: string
  name: string
  disease: string
  description: string
  createdAt: string
  lastModified: string
}

interface HomePageProps {
  onOpenProject: (project: Project) => void
}

export default function HomePage({ onOpenProject }: HomePageProps) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [projectName, setProjectName] = useState('')
  const [disease, setDisease] = useState('')
  const [description, setDescription] = useState('')

  // Placeholder projects
  const [projects] = useState<Project[]>([])

  const handleCreateProject = () => {
    if (!projectName.trim() || !disease.trim()) {
      alert('Please fill in project name and disease target')
      return
    }

    const newProject: Project = {
      id: Date.now().toString(),
      name: projectName,
      disease: disease,
      description: description,
      createdAt: new Date().toISOString(),
      lastModified: new Date().toISOString()
    }

    // Close modal and reset form
    setShowCreateModal(false)
    setProjectName('')
    setDisease('')
    setDescription('')

    // Open the project workspace
    onOpenProject(newProject)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleCreateProject()
    }
  }

  return (
    <div className="home-page">
      <div className="home-container">

        {/* Modern Hero Section */}
        <header className="hero-section">
          <div className="hero-content">
            <div className="logo-badge glass-panel">
              <span className="logo-icon-small">✨</span>
              <span>Next-Gen Discovery</span>
            </div>
            <h1 className="hero-title">
              <span className="text-gradient">ContAInnum</span>
            </h1>
            <p className="hero-slogan">
              Simplifying Complexity, Continuously
            </p>
            <p className="hero-subtitle">
              The unified platform for accelerating pharmaceutical breakthroughs through integrated AI research, retrosynthesis, and manufacturing assessment.
            </p>

            <div className="hero-actions">
              <button className="cta-button primary" onClick={() => setShowCreateModal(true)}>
                <span className="icon">+</span>
                New Project
              </button>
              <button className="cta-button secondary glass-panel">
                Documentation
              </button>
            </div>
          </div>
        </header>

        {/* Project Grid Section */}
        <div className="content-section">
          {projects.length === 0 ? (
            <div className="empty-state glass-panel">
              <div className="empty-icon-wrapper">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <h3>No active projects</h3>
              <p>Start your first research campaign to see AI agents in action.</p>
            </div>
          ) : (
            <div className="projects-grid">
              {projects.map(project => (
                <div
                  key={project.id}
                  className="project-card glass-panel"
                  onClick={() => onOpenProject(project)}
                >
                  <div className="card-image-placeholder"></div>
                  <div className="project-card-content">
                    <div className="project-header">
                      <h3>{project.name}</h3>
                      <span className="status-badge">Active</span>
                    </div>
                    <p className="project-disease">{project.disease}</p>
                    <div className="project-footer">
                      <span className="project-date">
                        {new Date(project.lastModified).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Modern Modal */}
      {showCreateModal && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="create-modal glass-panel-strong" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>New Project</h2>
              <button className="close-btn" onClick={() => setShowCreateModal(false)}>×</button>
            </div>

            <div className="modal-body">
              <div className="form-group">
                <label>Project Name</label>
                <input
                  type="text"
                  placeholder="e.g. Novel T2D Inhibitor"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="modern-input"
                  autoFocus
                />
              </div>

              <div className="form-group">
                <label>Disease Target</label>
                <input
                  type="text"
                  placeholder="e.g. Type 2 Diabetes"
                  value={disease}
                  onChange={(e) => setDisease(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="modern-input"
                />
              </div>

              <div className="form-group">
                <label>Description</label>
                <textarea
                  placeholder="Research goals..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="modern-textarea"
                  rows={3}
                />
              </div>
            </div>

            <div className="modal-footer">
              <button className="btn-ghost" onClick={() => setShowCreateModal(false)}>
                Cancel
              </button>
              <button className="cta-button primary full-width" onClick={handleCreateProject}>
                Create Project
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
