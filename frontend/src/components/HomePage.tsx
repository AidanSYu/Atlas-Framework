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
  
  // Placeholder projects - will be replaced with actual data later
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
        <header className="home-header">
          <div className="logo-section">
            <div className="logo-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <h1>ContAInnum</h1>
          </div>
          <p className="tagline">AI-Powered Drug Discovery & Development Platform</p>
        </header>

        <div className="home-content">
          <section className="welcome-section">
            <h2>Welcome to your Research Hub</h2>
            <p>Create a new project to begin your drug discovery journey, or continue working on existing research.</p>
          </section>

          <div className="action-cards">
            <div className="action-card primary" onClick={() => setShowCreateModal(true)}>
              <div className="card-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <h3>New Project</h3>
              <p>Start a new drug discovery project</p>
            </div>

            <div className="action-card secondary disabled">
              <div className="card-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
              </div>
              <h3>My Projects</h3>
              <p>Access and manage existing projects</p>
              <span className="coming-soon">Coming Soon</span>
            </div>
          </div>

          {projects.length > 0 && (
            <section className="recent-projects">
              <h2>Recent Projects</h2>
              <div className="projects-grid">
                {projects.map(project => (
                  <div 
                    key={project.id} 
                    className="project-card"
                    onClick={() => onOpenProject(project)}
                  >
                    <div className="project-header">
                      <h3>{project.name}</h3>
                      <span className="project-status">Active</span>
                    </div>
                    <p className="project-disease">{project.disease}</p>
                    <p className="project-description">{project.description}</p>
                    <div className="project-footer">
                      <span className="project-date">
                        Last modified: {new Date(project.lastModified).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>

      {/* Create Project Modal */}
      {showCreateModal && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="create-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Create New Project</h2>
              <button className="close-btn" onClick={() => setShowCreateModal(false)}>×</button>
            </div>
            
            <div className="modal-body">
              <div className="form-group">
                <label>Project Name *</label>
                <input
                  type="text"
                  placeholder="e.g., Novel T2D Treatment"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label>Disease Target *</label>
                <input
                  type="text"
                  placeholder="e.g., Type 2 Diabetes"
                  value={disease}
                  onChange={(e) => setDisease(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="form-input"
                />
              </div>

              <div className="form-group">
                <label>Description (Optional)</label>
                <textarea
                  placeholder="Describe your research goals and objectives..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="form-textarea"
                  rows={4}
                />
              </div>
            </div>

            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowCreateModal(false)}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleCreateProject}>
                Create Project
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
