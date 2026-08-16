import type { ChatDetail, RagSource, Repository } from './types'

const API_ROOT = import.meta.env.VITE_API_URL ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly detail?: unknown,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    let detail: unknown
    try { detail = await response.json() } catch { detail = undefined }
    throw new ApiError(errorMessage(detail) || response.statusText, response.status, detail)
  }
  return response.json() as Promise<T>
}

function errorMessage(value: unknown): string | null {
  if (!value || typeof value !== 'object') return null
  const detail = 'detail' in value ? value.detail : value
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object' && 'message' in detail && typeof detail.message === 'string') {
    return detail.message
  }
  if (Array.isArray(detail) && detail[0] && typeof detail[0] === 'object' && 'msg' in detail[0]) {
    return String(detail[0].msg)
  }
  return null
}

export const api = {
  listRepositories: () => request<Repository[]>('/api/repositories'),
  repository: (id: string) => request<Repository>(`/api/repositories/${id}/status`),
  createRepository: (repository_url: string) =>
    request<Pick<Repository, 'repository_id' | 'chat_id' | 'repository_name' | 'repository_owner' | 'repository_url' | 'status'>>('/api/repositories', {
      method: 'POST', body: JSON.stringify({ repository_url }),
    }),
  retry: (id: string) => request<Repository>(`/api/repositories/${id}/retry`, { method: 'POST' }),
  reindex: (id: string) => request<Repository>(`/api/repositories/${id}/reindex`, { method: 'POST' }),
  deleteChat: (id: string) => request<{ deleted: boolean }>(`/api/chats/${id}`, { method: 'DELETE' }),
  chat: (id: string) => request<ChatDetail>(`/api/chats/${id}`),
}

interface StreamHandlers {
  onSources: (sources: RagSource[]) => void
  onToken: (text: string) => void
  onDone: () => void
}

export async function streamQuestion(
  repositoryId: string,
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_ROOT}/api/repositories/${repositoryId}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ question }),
    signal,
  })
  if (!response.ok) {
    let detail: unknown
    try { detail = await response.json() } catch { detail = undefined }
    throw new ApiError(errorMessage(detail) || response.statusText, response.status, detail)
  }
  if (!response.body) throw new Error('The answer stream is unavailable.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''
    for (const block of blocks) {
      const event = block.split('\n').find((line) => line.startsWith('event:'))?.slice(6).trim()
      const dataLine = block.split('\n').find((line) => line.startsWith('data:'))?.slice(5).trim()
      if (!event || !dataLine) continue
      const data = JSON.parse(dataLine) as Record<string, unknown>
      if (event === 'sources') handlers.onSources((data.sources ?? []) as RagSource[])
      if (event === 'token') handlers.onToken(String(data.text ?? ''))
      if (event === 'done') handlers.onDone()
      if (event === 'error') throw new Error(String(data.message ?? 'The answer stream failed.'))
    }
    if (done) break
  }
}
