import { useState } from 'react'
import HomePage from './components/HomePage'
import ProjectWorkspace from './components/ProjectWorkspace'

interface Project {
  id: string
  name: string
  disease: string
  description: string
  createdAt: string
  lastModified: string
}

export default function App() {
  const [currentView, setCurrentView] = useState<'home' | 'workspace'>('home')
  const [currentProject, setCurrentProject] = useState<Project | null>(null)

  const handleOpenProject = (project: Project) => {
    setCurrentProject(project)
    setCurrentView('workspace')
  }

  const handleBackToHome = () => {
    setCurrentView('home')
    setCurrentProject(null)
  }

  return (
    <div style={{ height: '100vh', overflow: 'hidden' }}>
      {currentView === 'home' && (
        <HomePage onOpenProject={handleOpenProject} />
      )}
      {currentView === 'workspace' && currentProject && (
        <ProjectWorkspace 
          project={currentProject} 
          onBack={handleBackToHome}
        />
      )}
    </div>
  )
}

