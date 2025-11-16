import { useState } from 'react'
import ResearchTab from './tabs/ResearchTab.tsx'
import RetrosynthesisTab from './tabs/RetrosynthesisTab.tsx'
import ManufacturingTab from './tabs/ManufacturingTab.tsx'
import MyDataTab from './tabs/MyDataTab.tsx'
import '../styles/ProjectWorkspace.css'

interface Project {
  id: string
  name: string
  disease: string
  description: string
  createdAt: string
  lastModified: string
}

interface ProjectWorkspaceProps {
  project: Project
  onBack: () => void
}

type TabType = 'research' | 'retrosynthesis' | 'manufacturing' | 'mydata'

export default function ProjectWorkspace({ project, onBack }: ProjectWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<TabType>('research')

  return (
    <div className="project-workspace">
      {/* Top Navigation Bar */}
      <nav className="workspace-nav">
        <div className="nav-left">
          <button className="back-btn" onClick={onBack}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Home
          </button>
          <div className="project-info">
            <h1>{project.name}</h1>
            <span className="disease-tag">{project.disease}</span>
          </div>
        </div>
        <div className="nav-right">
          <button className="icon-btn" title="Project Settings">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </nav>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button 
          className={`tab-btn ${activeTab === 'research' ? 'active' : ''}`}
          onClick={() => setActiveTab('research')}
        >
          <svg className="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          Research
        </button>
        
        <button 
          className={`tab-btn ${activeTab === 'retrosynthesis' ? 'active' : ''}`}
          onClick={() => setActiveTab('retrosynthesis')}
        >
          <svg className="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
          Retrosynthesis
        </button>
        
        <button 
          className={`tab-btn ${activeTab === 'manufacturing' ? 'active' : ''}`}
          onClick={() => setActiveTab('manufacturing')}
        >
          <svg className="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
          Manufacturing
        </button>
        
        <button 
          className={`tab-btn ${activeTab === 'mydata' ? 'active' : ''}`}
          onClick={() => setActiveTab('mydata')}
        >
          <svg className="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
          </svg>
          My Data
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'research' && <ResearchTab project={project} />}
        {activeTab === 'retrosynthesis' && <RetrosynthesisTab project={project} />}
        {activeTab === 'manufacturing' && <ManufacturingTab project={project} />}
        {activeTab === 'mydata' && <MyDataTab project={project} />}
      </div>
    </div>
  )
}
