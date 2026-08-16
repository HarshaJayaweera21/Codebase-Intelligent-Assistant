import {
  AlertTriangle, ArrowRight, BarChart3, BookOpen, Bot, Braces, Check,
  CheckCircle2, ChevronRight, CircleDot, Clock3, Code2, ExternalLink,
  FileCode2, Folder, GitBranch, Github, Layers3, LoaderCircle, MessageSquareText,
  PanelLeftClose, PanelLeftOpen, Plus, RefreshCw, Search, Send, Settings, Sparkles,
  Trash2, X,
} from 'lucide-react'
import { FormEvent, ReactNode, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { ChatMessage, RagSource, Repository, RepositoryStatus, WorkspaceView } from './types'

const statusLabel: Record<RepositoryStatus, string> = {
  queued: 'Queued', cloning: 'Cloning', scanning: 'Scanning', chunking: 'Chunking',
  embedding: 'Embedding', indexing: 'Indexing', ready: 'Indexed', failed: 'Failed',
}

export function StatusBadge({ status }: { status: RepositoryStatus }) {
  return <span className={`status-badge status-${status}`}><span className="status-dot" />{statusLabel[status]}</span>
}

interface SidebarProps {
  repositories: Repository[]
  activeId: string | null
  collapsed: boolean
  onToggle: () => void
  onNew: () => void
  onSelect: (repository: Repository) => void
}

export function Sidebar({ repositories, activeId, collapsed, onToggle, onNew, onSelect }: SidebarProps) {
  return <>
    {collapsed && <button className="sidebar-open icon-button" onClick={onToggle} aria-label="Open sidebar" aria-expanded="false" title="Open sidebar"><PanelLeftOpen size={17} /></button>}
    <aside className={`sidebar ${collapsed ? 'is-collapsed' : ''}`}>
      <div className="brand-row">
        <div className="brand-mark"><Braces size={16} /></div>
        <div><div className="brand-name">Codebase Intelligence</div><div className="brand-subtitle">AI assistant</div></div>
        <button className="icon-button sidebar-toggle" onClick={onToggle} aria-label="Collapse sidebar"><PanelLeftClose size={16} /></button>
      </div>
      <button className="primary-button new-repository" onClick={onNew}><Plus size={15} />Analyze repository</button>
      <div className="sidebar-section-label">Repositories <span>{repositories.length}</span></div>
      <nav className="repository-list" aria-label="Repositories">
        {repositories.length === 0 ? <div className="sidebar-empty"><Folder size={18} /><p>No repositories yet</p><span>Analyze a public GitHub repository to begin.</span></div> :
          repositories.map((repo) => <button key={repo.repository_id} className={`repository-item ${activeId === repo.repository_id ? 'active' : ''}`} onClick={() => onSelect(repo)}>
            <span className="repo-icon"><Github size={15} /></span>
            <span className="repo-copy"><strong>{repo.repository_name}</strong><small>{repo.repository_owner}</small></span>
            <StatusBadge status={repo.status} />
          </button>)}
      </nav>
      <div className="sidebar-footer">
        <button><BookOpen size={15} />Documentation</button>
        <button><Settings size={15} />Settings</button>
      </div>
    </aside>
  </>
}

export function TopBar({ repository, view, onView, onDelete, onReindex }: {
  repository: Repository
  view: WorkspaceView
  onView: (view: WorkspaceView) => void
  onDelete: () => void
  onReindex: () => void
}) {
  const ready = repository.status === 'ready'
  return <header className="topbar">
    <div className="repository-identity">
      <Github size={17} />
      <div><strong>{repository.repository_name}</strong><span>{repository.repository_owner} / {repository.repository_name}</span></div>
      <StatusBadge status={repository.status} />
    </div>
    {ready && <nav className="workspace-tabs" aria-label="Repository workspace">
      <button className={view === 'chat' ? 'active' : ''} onClick={() => onView('chat')}><MessageSquareText size={15} />Chat</button>
      <button className={view === 'overview' ? 'active' : ''} onClick={() => onView('overview')}><BarChart3 size={15} />Overview</button>
    </nav>}
    <div className="topbar-actions">
      {ready && <button className="secondary-button compact" onClick={onReindex}><RefreshCw size={14} />Re-index</button>}
      <a className="icon-button" href={repository.repository_url} target="_blank" rel="noreferrer" aria-label="Open repository on GitHub"><ExternalLink size={16} /></a>
      <button className="icon-button danger-hover" onClick={onDelete} aria-label="Delete repository"><Trash2 size={16} /></button>
    </div>
  </header>
}

export function NewRepository({ onCreate, busy, error }: { onCreate: (url: string) => void; busy: boolean; error: string | null }) {
  const [url, setUrl] = useState('')
  const [localError, setLocalError] = useState<string | null>(null)
  const submit = (event: FormEvent) => {
    event.preventDefault()
    const value = url.trim()
    if (!/^https:\/\/(www\.)?github\.com\/[^/\s]+\/[^/\s]+\/?$/.test(value)) {
      setLocalError('Enter a repository URL like https://github.com/owner/repository')
      return
    }
    setLocalError(null)
    onCreate(value.replace(/\/$/, ''))
  }
  const examples = [
    ['Trace a flow', 'How is authentication implemented?'],
    ['Find configuration', 'Where is the database configured?'],
    ['Understand structure', 'Explain the main architecture.'],
    ['Locate a symbol', 'Which file defines the user model?'],
  ]
  return <main className="empty-workspace">
    <section className="empty-content">
      <div className="hero-glyph"><Code2 size={21} /></div>
      <p className="eyebrow">Repository-grounded intelligence</p>
      <h1>Understand any codebase with AI</h1>
      <p className="hero-copy">Analyze a public GitHub repository, explore its architecture, and ask questions grounded in the actual source code.</p>
      <form className={`repository-form ${(localError || error) ? 'has-error' : ''}`} onSubmit={submit}>
        <Github size={18} />
        <label className="sr-only" htmlFor="repository-url">GitHub repository URL</label>
        <input id="repository-url" value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://github.com/owner/repository" disabled={busy} autoFocus />
        <button className="primary-button" disabled={busy || !url.trim()}>{busy ? <LoaderCircle className="spin" size={16} /> : null}{busy ? 'Starting analysis' : 'Analyze repository'}<ArrowRight size={15} /></button>
      </form>
      {(localError || error) && <div className="form-error" role="alert"><AlertTriangle size={14} />{localError || error}</div>}
      <div className="privacy-note"><span className="status-dot" />Public GitHub repositories are supported</div>
      <div className="suggestion-heading"><span />What you’ll be able to ask<span /></div>
      <div className="suggestion-grid">
        {examples.map(([label, question]) => <div className="suggestion-card" key={question}><Sparkles size={15} /><span>{label}</span><p>{question}</p></div>)}
      </div>
      <div className="trust-row"><span><Code2 size={13} />Code-aware analysis</span><span><CircleDot size={13} />Source-grounded answers</span><span><GitBranch size={13} />Exact file and line references</span></div>
    </section>
  </main>
}

const pipeline: Array<{ status: Exclude<RepositoryStatus, 'queued' | 'ready' | 'failed'>; title: string; description: string }> = [
  { status: 'cloning', title: 'Clone repository', description: 'Fetch the latest repository snapshot' },
  { status: 'scanning', title: 'Scan supported files', description: 'Discover readable source and configuration files' },
  { status: 'chunking', title: 'Analyze source structure', description: 'Parse symbols and create code-aware chunks' },
  { status: 'embedding', title: 'Generate embeddings', description: 'Create semantic representations of each source chunk' },
  { status: 'indexing', title: 'Index repository', description: 'Store searchable context in an isolated namespace' },
]

const statusOrder: RepositoryStatus[] = ['queued', 'cloning', 'scanning', 'chunking', 'embedding', 'indexing', 'ready']

export function ProcessingView({ repository, error, onRetry, onRefresh, onChat, onOverview }: { repository: Repository; error: string | null; onRetry: () => void; onRefresh: () => void; onChat: () => void; onOverview: () => void }) {
  const failed = repository.status === 'failed'
  const ready = repository.status === 'ready'
  const currentIndex = statusOrder.indexOf(repository.status)
  const failedStepIndex = [12, 28, 48, 68, 91].findIndex((threshold) => repository.progress_percent <= threshold)
  if (ready) return <ReadyView repository={repository} onChat={onChat} onOverview={onOverview} />
  return <main className="processing-workspace page-scroll">
    <div className="page-container processing-container">
      <div className="page-heading">
        <div><p className="eyebrow">Repository analysis</p><h1>{failed ? 'Analysis needs attention' : `Preparing ${repository.repository_name}`}</h1><p>{failed ? 'The pipeline stopped before the repository was ready.' : repository.status_message}</p></div>
        <div className="progress-summary"><span>{failed ? 'Stopped' : `${repository.progress_percent}%`}</span><small>{failed ? statusLabel[repository.status] : `Step ${Math.max(1, currentIndex)} of 5`}</small></div>
      </div>
      {error && <div className="processing-alert" role="alert"><AlertTriangle size={16} /><div><strong>Backend connection interrupted</strong><span>{error}</span></div><button className="secondary-button compact" onClick={onRefresh}><RefreshCw size={14} />Check status</button></div>}
      <div className="processing-grid">
        <section className="pipeline-panel">
          <div className="pipeline-topline"><span>Analysis pipeline</span><span>{repository.repository_owner} / {repository.repository_name}</span></div>
          <div className="pipeline-list">
            {pipeline.map((step, index) => {
              const stepIndex = statusOrder.indexOf(step.status)
              const isFailed = failed && index === (failedStepIndex < 0 ? pipeline.length - 1 : failedStepIndex)
              const complete = !failed && currentIndex > stepIndex
              const active = !failed && repository.status === step.status
              return <div className={`pipeline-step ${complete ? 'complete' : ''} ${active ? 'active' : ''} ${isFailed ? 'failed' : ''}`} key={step.status}>
                <div className="step-rail"><span>{complete ? <Check size={13} /> : isFailed ? <X size={13} /> : active ? <LoaderCircle className="spin" size={13} /> : <span />}</span></div>
                <div className="step-copy"><strong>{step.title}</strong><p>{isFailed ? repository.error || repository.status_message : active ? repository.status_message : step.description}</p>
                  {active && <div className="inline-progress"><i style={{ width: `${repository.progress_percent}%` }} /></div>}
                  {isFailed && <button className="secondary-button compact" onClick={onRetry}><RefreshCw size={14} />Retry analysis</button>}
                </div>
                <small>{complete ? 'Complete' : active ? 'In progress' : isFailed ? 'Failed' : 'Waiting'}</small>
              </div>
            })}
          </div>
        </section>
        <RepositoryStats repository={repository} />
      </div>
      <p className="background-note"><Clock3 size={14} />You can leave this page while analysis continues.</p>
    </div>
  </main>
}

function ReadyView({ repository, onChat, onOverview }: { repository: Repository; onChat: () => void; onOverview: () => void }) {
  return <main className="processing-workspace page-scroll"><div className="page-container ready-container">
    <div className="ready-heading"><div className="ready-icon"><CheckCircle2 size={22} /></div><div><p className="eyebrow">Analysis complete</p><h1>{repository.repository_name} is ready</h1><p>Your repository is indexed and ready for grounded questions.</p></div></div>
    <div className="ready-body">
      <section className="pipeline-panel ready-panel">
        <div className="pipeline-topline"><span>Indexed pipeline</span><span>{repository.updated_at ? `Updated ${relativeTime(repository.updated_at)}` : 'Complete'}</span></div>
        {pipeline.map((step) => <div className="ready-step" key={step.status}><span><Check size={13} /></span><div><strong>{step.title}</strong><p>{step.description}</p></div><small>Complete</small></div>)}
      </section>
      <RepositoryStats repository={repository} />
    </div>
    <div className="ready-actions"><button className="primary-button" onClick={onChat}>Start asking questions<ArrowRight size={15} /></button><button className="secondary-button" onClick={onOverview}>View repository overview</button></div>
  </div></main>
}

export function RepositoryStats({ repository }: { repository: Repository }) {
  const summary = repository.scan_summary
  const languages = Object.entries(summary?.languages ?? {}).sort((a, b) => b[1] - a[1])
  const max = Math.max(...languages.map(([, count]) => count), 1)
  return <aside className="stats-panel">
    <div className="panel-label"><BarChart3 size={14} />Repository stats</div>
    <div className="stat-grid">
      <div><strong>{formatNumber(summary?.total_files)}</strong><span>Total files</span></div>
      <div><strong>{formatNumber(summary?.supported_files)}</strong><span>Indexed files</span></div>
      <div><strong>{formatNumber(repository.chunk_count)}</strong><span>Source chunks</span></div>
      <div><strong>{languages.length || '—'}</strong><span>File types</span></div>
    </div>
    <div className="language-list">
      <div className="panel-label">Language breakdown</div>
      {languages.length ? languages.slice(0, 6).map(([language, count]) => <div className="language-row" key={language}><div><span>{language}</span><strong>{count}</strong></div><i><b style={{ width: `${(count / max) * 100}%` }} /></i></div>) : <p className="muted-empty">Available after the file scan completes.</p>}
    </div>
  </aside>
}

export function ChatWorkspace({ repository, messages, streaming, error, onAsk, onSource }: {
  repository: Repository
  messages: ChatMessage[]
  streaming: boolean
  error: string | null
  onAsk: (question: string) => void
  onSource: (source: RagSource) => void
}) {
  const [question, setQuestion] = useState('')
  const suggestions = ['Explain the repository architecture', 'Where is authentication handled?', 'Trace a request through the codebase', 'Which files contain the core domain logic?']
  const submit = (event: FormEvent) => { event.preventDefault(); if (question.trim() && !streaming) { onAsk(question.trim()); setQuestion('') } }
  return <main className="chat-workspace">
    <div className="chat-scroll">
      {messages.length === 0 ? <section className="chat-empty">
        <div className="hero-glyph small"><Bot size={19} /></div><p className="eyebrow">Scoped to {repository.repository_name}</p>
        <h1>Ask the codebase</h1><p>Every answer is grounded in this repository’s indexed source.</p>
        <div className="chat-suggestions">{suggestions.map((item) => <button key={item} onClick={() => onAsk(item)} disabled={streaming}><Sparkles size={15} />{item}<ArrowRight size={14} /></button>)}</div>
      </section> : <div className="message-list">
        {messages.map((message) => <Message key={message.message_id} message={message} onSource={onSource} />)}
        {streaming && <div className="stream-status"><LoaderCircle className="spin" size={14} />Reading repository context…</div>}
      </div>}
    </div>
    <div className="composer-wrap">
      {error && <div className="composer-error" role="alert"><AlertTriangle size={14} />{error}</div>}
      <form className="composer" onSubmit={submit}>
        <label className="sr-only" htmlFor="question">Ask about this repository</label>
        <textarea id="question" rows={1} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder={`Ask about ${repository.repository_name}…`} disabled={streaming} />
        <button className="send-button" aria-label="Send question" disabled={!question.trim() || streaming}>{streaming ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />}</button>
      </form>
      <p>Answers are generated from indexed repository sources. Verify important details.</p>
    </div>
  </main>
}

function Message({ message, onSource }: { message: ChatMessage; onSource: (source: RagSource) => void }) {
  if (message.role === 'user') return <article className="message user-message"><div className="message-avatar">You</div><div className="message-body markdown-content"><ReactMarkdown skipHtml>{message.content}</ReactMarkdown></div></article>
  return <article className="message assistant-message"><div className="message-avatar"><Bot size={15} /></div><div className="message-body"><div className="answer-text markdown-content">{message.content ? <ReactMarkdown skipHtml>{message.content}</ReactMarkdown> : <span className="typing-caret" />}</div>
    {message.sources.length > 0 ? <div className="source-citations"><div className="source-citations-label">Sources</div>{message.sources.map((source) => <SourceCitation source={source} onClick={() => onSource(source)} key={`${message.message_id}-${source.citation_id}-${source.vector_id}`} />)}</div> : message.content && <div className="no-sources"><AlertTriangle size={13} />No source citations were returned for this answer.</div>}
  </div></article>
}

function SourceCitation({ source, onClick }: { source: RagSource; onClick: () => void }) {
  return <button className="source-citation" onClick={onClick}><FileCode2 size={14} /><span><strong>{fileName(source.file_path)}</strong><small>{source.symbol_name || source.chunk_type} · {formatRanges(source.source_ranges)}</small></span><ChevronRight size={14} /></button>
}

export function SourcePanel({ source, onClose }: { source: RagSource; onClose: () => void }) {
  const lines = source.content ? source.content.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n') : []
  const exactLineNumbers = source.source_ranges.flatMap((range) => Array.from({ length: range.end_line - range.start_line + 1 }, (_, index) => range.start_line + index))
  const canMapExactLines = exactLineNumbers.length === lines.length
  const firstLine = source.source_ranges[0]?.start_line ?? source.symbol_start_line ?? 1
  const lineNumberAt = (index: number) => canMapExactLines ? (source.source_ranges.length === 1 ? firstLine + index : exactLineNumbers[index]) : null
  return <aside className="source-panel" aria-label="Source code">
    <header className="source-header"><div><div className="source-title"><FileCode2 size={16} /><strong>{fileName(source.file_path)}</strong></div><p>{source.file_path}</p></div><button className="icon-button" onClick={onClose} aria-label="Close source panel"><X size={17} /></button></header>
    <div className="source-meta"><span>{source.language}</span><span>{source.chunk_type}</span>{source.symbol_name && <span>{source.symbol_name}</span>}</div>
    <div className="range-banner"><Layers3 size={14} /><div><strong>Exact source ranges</strong><span>{formatRanges(source.source_ranges)}</span></div>{source.symbol_start_line && <small>Symbol {source.symbol_start_line}–{source.symbol_end_line}</small>}</div>
    <div className="code-viewer">
      {lines.length ? <pre>{lines.map((line, index) => { const lineNumber = lineNumberAt(index); return <div className={`code-line ${lineNumber !== null ? 'selected' : ''}`} key={index}><span>{lineNumber ?? '·'}</span><code>{line || ' '}</code></div> })}</pre> : <div className="source-unavailable"><FileCode2 size={24} /><strong>Source unavailable</strong><p>The citation metadata is available, but this source did not include displayable code.</p></div>}
    </div>
    <footer className="source-footer"><span>{canMapExactLines ? source.citation_id : 'Combined non-contiguous excerpt'}</span><span>Relevance {Math.round(source.score * 100)}%</span></footer>
  </aside>
}

export function Overview({ repository, onChat }: { repository: Repository; onChat: () => void }) {
  const summary = repository.scan_summary
  return <main className="overview-workspace page-scroll"><div className="page-container overview-container">
    <div className="overview-heading"><div><p className="eyebrow">Repository overview</p><h1>{repository.repository_name}</h1><p><Github size={14} />{repository.repository_owner} / {repository.repository_name}</p></div><button className="primary-button" onClick={onChat}><MessageSquareText size={15} />Open chat</button></div>
    <div className="overview-stat-row">
      <Metric label="Total files" value={formatNumber(summary?.total_files)} />
      <Metric label="Indexed files" value={formatNumber(summary?.supported_files)} />
      <Metric label="Ignored files" value={formatNumber(summary?.ignored_files)} />
      <Metric label="Source chunks" value={formatNumber(repository.chunk_count)} />
    </div>
    <div className="overview-grid">
      <section className="structure-panel">
        <div className="section-heading"><div><Folder size={16} /><span>Repository structure</span></div><span className="api-note">Metadata only</span></div>
        <div className="fake-filter"><Search size={15} /><span>File search requires a repository tree endpoint</span></div>
        <div className="tree-unavailable"><Folder size={27} /><strong>Repository tree unavailable</strong><p>The current production API exposes indexed totals and language composition, but not safe file-tree data.</p></div>
      </section>
      <div className="overview-aside">
        <RepositoryStats repository={repository} />
        <section className="index-card"><div className="panel-label">AI index configuration</div><dl><div><dt>Status</dt><dd className="success-text">Indexed</dd></div><div><dt>Scope</dt><dd>Repository only</dd></div><div><dt>Documents</dt><dd>{formatNumber(repository.indexed_document_count)}</dd></div><div><dt>Last indexed</dt><dd>{relativeTime(repository.updated_at)}</dd></div></dl></section>
      </div>
    </div>
  </div></main>
}

function Metric({ label, value }: { label: string; value: string | number }) { return <div className="metric"><span>{label}</span><strong>{value}</strong></div> }

export function ConfirmDialog({ title, body, confirmLabel, danger, busy, onConfirm, onClose }: { title: string; body: ReactNode; confirmLabel: string; danger?: boolean; busy?: boolean; onConfirm: () => void; onClose: () => void }) {
  return <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose() }}><section className="dialog" role="alertdialog" aria-modal="true" aria-labelledby="dialog-title"><div className={`dialog-icon ${danger ? 'danger' : ''}`}>{danger ? <Trash2 size={19} /> : <RefreshCw size={19} />}</div><h2 id="dialog-title">{title}</h2><div className="dialog-body">{body}</div><div className="dialog-actions"><button className="secondary-button" onClick={onClose} disabled={busy}>Cancel</button><button className={danger ? 'danger-button' : 'primary-button'} onClick={onConfirm} disabled={busy}>{busy && <LoaderCircle className="spin" size={15} />}{confirmLabel}</button></div></section></div>
}

export function LoadingScreen() { return <main className="loading-workspace"><LoaderCircle className="spin" size={20} /><span>Loading repositories…</span></main> }

export function ErrorScreen({ message, onRetry }: { message: string; onRetry: () => void }) { return <main className="loading-workspace error-state"><AlertTriangle size={24} /><strong>Couldn’t load the workspace</strong><span>{message}</span><button className="secondary-button" onClick={onRetry}><RefreshCw size={14} />Try again</button></main> }

function formatNumber(value: number | null | undefined): string { return typeof value === 'number' ? new Intl.NumberFormat().format(value) : '—' }
function relativeTime(value: string): string { const date = new Date(value); if (Number.isNaN(date.getTime())) return 'Recently'; const minutes = Math.max(0, Math.round((Date.now() - date.getTime()) / 60000)); if (minutes < 1) return 'Just now'; if (minutes < 60) return `${minutes}m ago`; const hours = Math.round(minutes / 60); if (hours < 24) return `${hours}h ago`; return `${Math.round(hours / 24)}d ago` }
function fileName(path: string): string { return path.split('/').pop() || path }
function formatRanges(ranges: Array<{ start_line: number; end_line: number }>): string { return ranges.length ? ranges.map((range) => range.start_line === range.end_line ? `L${range.start_line}` : `L${range.start_line}–${range.end_line}`).join(', ') : 'Range unavailable' }
