import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError, streamQuestion } from './api'
import {
  ChatWorkspace, ConfirmDialog, ErrorScreen, LoadingScreen, NewRepository, Overview,
  ProcessingView, Sidebar, SourcePanel, TopBar,
} from './components'
import type { ChatMessage, RagSource, Repository, WorkspaceView } from './types'

type Dialog = 'delete' | 'reindex' | null

export function App() {
  const [repositories, setRepositories] = useState<Repository[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [view, setView] = useState<WorkspaceView>('new')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [selectedSource, setSelectedSource] = useState<RagSource | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [collapsed, setCollapsed] = useState(() => window.matchMedia('(max-width: 1050px)').matches)
  const [dialog, setDialog] = useState<Dialog>(null)
  const [dialogBusy, setDialogBusy] = useState(false)
  const activeIdRef = useRef<string | null>(null)
  const active = repositories.find((repository) => repository.repository_id === activeId) ?? null
  const activeRepositoryId = active?.repository_id
  const activeChatId = active?.chat_id
  const activeStatus = active?.status

  useEffect(() => { activeIdRef.current = activeId }, [activeId])

  useEffect(() => {
    const query = window.matchMedia('(max-width: 1050px)')
    const handleChange = (event: MediaQueryListEvent) => setCollapsed(event.matches)
    query.addEventListener('change', handleChange)
    return () => query.removeEventListener('change', handleChange)
  }, [])

  const loadRepositories = useCallback(async () => {
    setLoading(true); setLoadError(null)
    try {
      const data = await api.listRepositories()
      setRepositories(data)
      if (data.length && !activeIdRef.current) {
        const initial = data[0]
        setActiveId(initial.repository_id)
        setView(initial.status === 'ready' ? 'chat' : 'processing')
      } else if (!data.length) setView('new')
    } catch (error) { setLoadError(messageFor(error)) } finally { setLoading(false) }
  }, [])

  useEffect(() => { void loadRepositories() }, [loadRepositories])

  useEffect(() => {
    if (!activeRepositoryId || activeStatus === 'ready' || activeStatus === 'failed') return
    const timer = window.setInterval(async () => {
      try {
        const updated = await api.repository(activeRepositoryId)
        setRepositories((current) => current.map((item) => item.repository_id === updated.repository_id ? updated : item))
      } catch (error) { setActionError(messageFor(error)) }
    }, 1600)
    return () => window.clearInterval(timer)
  }, [activeRepositoryId, activeStatus])

  useEffect(() => {
    if (!activeChatId || activeStatus !== 'ready' || view !== 'chat') return
    let ignore = false
    api.chat(activeChatId).then((chat) => { if (!ignore) setMessages(chat.messages) }).catch((error) => { if (!ignore) setActionError(messageFor(error)) })
    return () => { ignore = true }
  }, [activeChatId, activeStatus, view])

  const selectRepository = (repository: Repository) => {
    setActiveId(repository.repository_id); setSelectedSource(null); setActionError(null); setMessages([])
    setView(repository.status === 'ready' ? 'chat' : 'processing')
  }

  const createRepository = async (url: string) => {
    setCreating(true); setActionError(null)
    try {
      const created = await api.createRepository(url)
      const record = await api.repository(created.repository_id)
      setRepositories((current) => [record, ...current])
      setActiveId(record.repository_id); setView('processing')
    } catch (error) {
      if (error instanceof ApiError && error.status === 409 && error.detail && typeof error.detail === 'object' && 'detail' in error.detail) {
        const detail = error.detail.detail
        if (detail && typeof detail === 'object' && 'repository_id' in detail) {
          const id = String(detail.repository_id); const existing = repositories.find((item) => item.repository_id === id)
          if (existing) selectRepository(existing)
        }
      }
      setActionError(messageFor(error))
    } finally { setCreating(false) }
  }

  const ask = async (question: string) => {
    if (!active || streaming) return
    setStreaming(true); setActionError(null)
    const stamp = Date.now().toString()
    const user: ChatMessage = { message_id: `local-user-${stamp}`, role: 'user', content: question, sources: [], created_at: new Date().toISOString() }
    const assistantId = `local-assistant-${stamp}`
    const assistant: ChatMessage = { message_id: assistantId, role: 'assistant', content: '', sources: [], created_at: new Date().toISOString() }
    setMessages((current) => [...current, user, assistant])
    try {
      await streamQuestion(active.repository_id, question, {
        onSources: (sources) => setMessages((current) => current.map((item) => item.message_id === assistantId ? { ...item, sources } : item)),
        onToken: (text) => setMessages((current) => current.map((item) => item.message_id === assistantId ? { ...item, content: item.content + text } : item)),
        onDone: () => undefined,
      })
    } catch (error) {
      setMessages((current) => current.filter((item) => item.message_id !== assistantId))
      setActionError(messageFor(error))
    } finally { setStreaming(false) }
  }

  const retry = async () => {
    if (!active) return
    setActionError(null)
    try { const updated = await api.retry(active.repository_id); setRepositories((current) => current.map((item) => item.repository_id === updated.repository_id ? updated : item)) }
    catch (error) { setActionError(messageFor(error)) }
  }

  const refreshActiveRepository = async () => {
    if (!active) return
    setActionError(null)
    try {
      const updated = await api.repository(active.repository_id)
      setRepositories((current) => current.map((item) => item.repository_id === updated.repository_id ? updated : item))
    } catch (error) { setActionError(messageFor(error)) }
  }

  const confirmDialog = async () => {
    if (!active || !dialog) return
    setDialogBusy(true); setActionError(null)
    try {
      if (dialog === 'delete') {
        await api.deleteChat(active.chat_id)
        const remaining = repositories.filter((item) => item.repository_id !== active.repository_id)
        setRepositories(remaining); setSelectedSource(null); setMessages([])
        if (remaining.length) selectRepository(remaining[0]); else { setActiveId(null); setView('new') }
      } else {
        const updated = await api.reindex(active.repository_id)
        setRepositories((current) => current.map((item) => item.repository_id === updated.repository_id ? updated : item))
        setView('processing'); setSelectedSource(null)
      }
      setDialog(null)
    } catch (error) { setActionError(messageFor(error)); setDialog(null) }
    finally { setDialogBusy(false) }
  }

  return <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''} ${selectedSource ? 'source-open' : ''}`}>
    <Sidebar repositories={repositories} activeId={activeId} collapsed={collapsed} onToggle={() => setCollapsed((value) => !value)} onNew={() => { setActiveId(null); setView('new'); setSelectedSource(null); setActionError(null) }} onSelect={selectRepository} />
    <div className="workspace-shell">
      {active && <TopBar repository={active} view={view} onView={(next) => { setView(next); setSelectedSource(null) }} onDelete={() => setDialog('delete')} onReindex={() => setDialog('reindex')} />}
      {loading ? <LoadingScreen /> : loadError ? <ErrorScreen message={loadError} onRetry={() => void loadRepositories()} /> : view === 'new' || !active ? <NewRepository onCreate={(url) => void createRepository(url)} busy={creating} error={actionError} /> : view === 'processing' ? <ProcessingView repository={active} error={actionError} onRetry={() => void retry()} onRefresh={() => void refreshActiveRepository()} onChat={() => setView('chat')} onOverview={() => setView('overview')} /> : view === 'overview' ? <Overview repository={active} onChat={() => setView('chat')} /> : <ChatWorkspace repository={active} messages={messages} streaming={streaming} error={actionError} onAsk={(question) => void ask(question)} onSource={setSelectedSource} />}
    </div>
    {selectedSource && <SourcePanel source={selectedSource} onClose={() => setSelectedSource(null)} />}
    {dialog && active && <ConfirmDialog title={dialog === 'delete' ? `Delete ${active.repository_name}?` : `Re-index ${active.repository_name}?`} body={dialog === 'delete' ? <>This removes the chat, local repository copy, indexed vectors, and metadata. This action cannot be undone.</> : <>The current index will remain available until a fresh repository analysis begins. Chat is unavailable while re-indexing.</>} confirmLabel={dialog === 'delete' ? 'Delete repository' : 'Start re-index'} danger={dialog === 'delete'} busy={dialogBusy} onClose={() => setDialog(null)} onConfirm={() => void confirmDialog()} />}
  </div>
}

function messageFor(error: unknown): string { return error instanceof Error ? error.message : 'Something went wrong. Try again.' }
