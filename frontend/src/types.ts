export type RepositoryStatus =
  | 'queued'
  | 'cloning'
  | 'scanning'
  | 'chunking'
  | 'embedding'
  | 'indexing'
  | 'ready'
  | 'failed'

export interface ScanSummary {
  total_files: number
  supported_files: number
  ignored_files: number
  languages: Record<string, number>
}

export interface Repository {
  repository_id: string
  chat_id: string
  repository_name: string
  repository_owner: string
  repository_url: string
  status: RepositoryStatus
  progress_percent: number
  status_message: string
  created_at: string
  updated_at: string
  scan_summary: ScanSummary | null
  chunk_count: number | null
  indexed_document_count: number | null
  error: string | null
}

export interface SourceRange {
  start_line: number
  end_line: number
}

export interface RagSource {
  citation_id: string
  vector_id: string
  score: number
  vector_score: number
  lexical_score: number
  exact_match_score: number
  structural_score: number
  file_path: string
  language: string
  chunk_type: string
  symbol_name: string | null
  symbol_start_line: number | null
  symbol_end_line: number | null
  source_ranges: SourceRange[]
  content: string
}

export interface ChatMessage {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  sources: RagSource[]
  created_at: string
}

export interface ChatDetail {
  chat_id: string
  repository_id: string
  title: string
  repository_name: string
  repository_owner: string
  repository_url: string
  repository_status: RepositoryStatus
  message_count: number
  created_at: string
  updated_at: string
  messages: ChatMessage[]
}

export type WorkspaceView = 'new' | 'processing' | 'chat' | 'overview'
