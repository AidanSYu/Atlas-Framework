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
      {/* Glass Navigation Bar */}
      <nav className="workspace-nav glass-panel-strong">
        <div className="nav-left">
          <button className="back-btn" onClick={onBack}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="back-text">Back</span>
          </button>

          <div className="project-info">
            <h1 className="project-title">{project.name}</h1>
            <div className="project-meta">
              <span className="disease-badge">{project.disease}</span>
              <span className="status-dot"></span>
              <span className="status-text">Active Session</span>
            </div>
          </div>
        </div>

        {/* Centered Tab Navigation (Segmented Control) */}
        <div className="tab-navigation glass-panel">
          <button
            className={`tab-segment ${activeTab === 'research' ? 'active' : ''}`}
            onClick={() => setActiveTab('research')}
          >
            Research
          </button>
          <button
            className={`tab-segment ${activeTab === 'retrosynthesis' ? 'active' : ''}`}
            onClick={() => setActiveTab('retrosynthesis')}
          >
            Retrosynthesis
          </button>
          <button
            className={`tab-segment ${activeTab === 'manufacturing' ? 'active' : ''}`}
            onClick={() => setActiveTab('manufacturing')}
          >
            Manufacturing
          </button>
          <button
            className={`tab-segment ${activeTab === 'mydata' ? 'active' : ''}`}
            onClick={() => setActiveTab('mydata')}
          >
            My Data
          </button>
        </div>

        <div className="nav-right">
          <button className="icon-btn glass-panel" title="Project Settings">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </nav>

      {/* Tab Content Area with Persistence */}
      <div className="tab-viewport">
        <div style={{ display: activeTab === 'research' ? 'block' : 'none', height: '100%' }}>
          <ResearchTab project={project} />
        </div>
        <div style={{ display: activeTab === 'retrosynthesis' ? 'block' : 'none', height: '100%', overflowY: 'auto' }}>
          <RetrosynthesisTab project={project} />
        </div>
        <div style={{ display: activeTab === 'manufacturing' ? 'block' : 'none', height: '100%', overflowY: 'auto' }}>
          <ManufacturingTab project={project} />
        </div>
        <div style={{ display: activeTab === 'mydata' ? 'block' : 'none', height: '100%', overflowY: 'auto' }}>
          <MyDataTab project={project} />
        </div>
      </div>
    </div>
  )
}
