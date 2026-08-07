# Codebase Intelligence Assistant — Agent Guide

## 1. Purpose
This file is the canonical project reference for coding agents working on the **Codebase Intelligence Assistant**. Read it before making architectural or implementation changes. Preserve the decisions below unless requirements explicitly change.

## 2. Product vision
The system is an AI-powered assistant for understanding GitHub codebases. A user submits a GitHub repository URL; the backend clones the repository locally, scans and parses useful files, creates RAG-ready chunks, stores vector representations in Pinecone, and creates a dedicated chat for that repository.

Typical questions:
- Where is authentication implemented?
- Explain the login flow.
- Which file defines the User model?
- What does this service depend on?
- Where is cart logic implemented?
- Explain the repository architecture.

Answers should be grounded in repository evidence and should ultimately cite file paths, symbol names, chunk types, and source line ranges.

## 3. Fundamental product rule
**One repository = one chat.**

Each chat is bound to exactly one repository. Repository retrieval must remain isolated between chats.

Example:
```text
Chat A -> Banking-System repository
Chat B -> Food-Delivery repository
Chat C -> Portfolio repository
```

## 4. User model
This application currently does **not** require:
- multi-user support
- authentication
- registration/login
- roles/permissions

Do not introduce authentication unless requirements explicitly change.

## 5. Repository lifecycle
When a GitHub URL is submitted:
```text
GitHub URL
  -> validate
  -> generate repository_id + chat_id
  -> clone locally
  -> scan files
  -> read supported files
  -> structurally chunk content
  -> create embeddings
  -> store vectors in Pinecone
  -> repository chat becomes ready
```

Repositories remain locally available while their chat exists.

Local storage convention:
```text
storage/repositories/<repository_id>/
```

Use generated repository IDs for storage, not repository names.

## 6. Chat deletion rule
Deleting a chat must eventually clean up all derived resources:
```text
Delete Chat
  -> delete local cloned repository
  -> delete Pinecone namespace/vectors
  -> delete chat messages
  -> delete repository metadata
  -> delete chat metadata
```

Do not leave orphaned repository folders or vectors.

Only backend-controlled paths under `storage/repositories/` may be deleted. Never delete arbitrary user-supplied paths.

## 7. Technology stack
### Frontend
- React
- TypeScript
- CSS

Responsibilities:
- GitHub URL submission
- repository/chat sidebar
- processing state UI
- chat interface
- streaming responses
- source references
- delete-chat action

### Backend
- FastAPI
- Python

Responsibilities:
- URL validation
- cloning
- local repository management
- scanning
- source extraction
- RAG orchestration
- Pinecone communication
- chat APIs
- streaming
- cleanup

### AI
- LangChain
- LangGraph
- LangSmith

LangChain is the main RAG/orchestration layer. LangGraph should be added only after core RAG works. LangSmith should be used later for tracing and evaluation.

### Vector database
Use **Pinecone**.

Reasons:
- industry relevance
- common in AI job descriptions
- managed vector database
- LangChain integration
- metadata filtering
- dense/sparse/hybrid retrieval support

Recommended index concept:
```text
index: codebase-assistant
namespace: <repository_id>
```

Repository isolation in Pinecone is mandatory.

### Relational database
Not required immediately.

Use SQLite first if persistence is needed. Migrate to PostgreSQL later only if deployment/application complexity justifies it.

## 8. Backend architecture
Current intended structure:
```text
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── repository_routes.py
│   ├── services/
│   │   ├── repository_service.py
│   │   └── repository_scanner.py
│   ├── models/
│   │   ├── repository.py
│   │   ├── repository_file.py
│   │   ├── repository_scan.py
│   │   └── code_chunk.py
│   ├── rag/
│   │   ├── document_processor.py
│   │   ├── tree_sitter_profiles.py
│   │   └── tree_sitter_chunker.py
│   └── core/
│       └── exceptions.py
├── storage/
│   └── repositories/
├── tests/
├── requirements.txt
└── .gitignore
```

Routes should remain thin. Complex cloning/scanning/chunking/RAG logic belongs in services or the `rag` package.

## 9. Existing API behavior
### Health
```text
GET /health
```
Expected:
```json
{"status":"ok"}
```

### Repository creation
```text
POST /api/repositories
```
Input:
```json
{"repository_url":"https://github.com/example/project"}
```

The endpoint currently:
1. validates the GitHub URL
2. extracts repository owner/name
3. generates repository and chat IDs
4. clones the repository
5. scans the repository
6. returns scan information

Observed working example:
```json
{
  "repository_id": "repo_a7890eef",
  "chat_id": "chat_96636fdf",
  "repository_name": "QuickBite-WebApp",
  "repository_owner": "HarshaJayaweera21",
  "repository_url": "https://github.com/HarshaJayaweera21/QuickBite-WebApp.git",
  "local_path": "storage\\repositories\\repo_a7890eef",
  "status": "scanned",
  "scan_summary": {
    "total_files": 92,
    "supported_files": 38,
    "ignored_files": 54,
    "languages": {
      "Java": 19,
      "CSS": 15,
      "Markdown": 2,
      "Maven Configuration": 1,
      "XML": 1
    }
  }
}
```

## 10. Atomic repository creation
Repository creation should behave atomically.

Conceptual flow:
```text
validate
 -> generate temporary IDs
 -> clone
 -> if clone fails: remove partial folder + return error
 -> scan
 -> if scan fails: remove cloned folder + return error
 -> return success
```

When persistence is later added, avoid leaving database records if repository creation fails.

## 11. Git cloning
Current approach uses Git through Python `subprocess`, conceptually:
```text
git clone --depth 1 <url> <destination>
```

`--depth 1` is intentional because the MVP analyzes the current repository state rather than full Git history.

Private repositories are not officially supported yet. On a developer machine, cached Git credentials may make a private clone succeed; do not treat this as production private-repository support.

## 12. Repository scanner
The scanner recursively discovers useful files and skips generated/dependency/cache directories.

Common ignored directories include:
```text
.git
node_modules
dist
build
target
coverage
.venv
venv
__pycache__
.idea
.vscode
.next
.nuxt
.cache
.pytest_cache
.gradle
bin
obj
vendor
```

Sensitive/unnecessary files should not be indexed:
```text
.env
.env.local
.env.production
*.pem
*.key
package-lock.json
yarn.lock
pnpm-lock.yaml
composer.lock
poetry.lock
```

## 13. Supported files
Current extension mapping should cover:
```text
.py  -> Python
.java -> Java
.js -> JavaScript
.jsx -> JavaScript JSX
.ts -> TypeScript
.tsx -> TypeScript TSX
.c -> C
.h -> C Header
.cpp/.cc -> C++
.hpp -> C++ Header
.cs -> C#
.go -> Go
.rs -> Rust
.php -> PHP
.rb -> Ruby
.kt -> Kotlin
.kts -> Kotlin Script
.swift -> Swift
.html/.htm -> HTML
.css -> CSS
.scss -> SCSS
.md -> Markdown
.json -> JSON
.yaml/.yml -> YAML
.xml -> XML
.sql -> SQL
.sh -> Shell
```

Important filename-specific formats include:
```text
Dockerfile
Makefile
requirements.txt
pyproject.toml
Pipfile
package.json
tsconfig.json
pom.xml
build.gradle
build.gradle.kts
settings.gradle
settings.gradle.kts
docker-compose.yml
docker-compose.yaml
```

## 14. RepositoryFile model
A supported readable file is represented internally as:
```text
RepositoryFile
- relative_path
- language
- size_bytes
- content
```

Store repository-relative POSIX-style paths such as:
```text
src/main/java/controller/CartServlet.java
```

Avoid exposing absolute local paths.

File reading should:
- skip empty files
- skip files above configured maximum size
- safely decode text
- skip unreadable files

Current approximate max size: `1_000_000` bytes.

## 15. LangChain Document layer
Repository files can be converted into LangChain Documents:
```python
Document(
    page_content=repository_file.content,
    metadata={
        "repository_id": repository_id,
        "file_path": repository_file.relative_path,
        "language": repository_file.language,
        "size_bytes": repository_file.size_bytes,
    },
)
```

Metadata must survive later chunking and retrieval.

## 16. Chunking strategy
Do **not** use one fixed character splitter for every file.

Preferred architecture:
```text
RepositoryFile
   -> determine type
      -> programming code -> Tree-sitter structural parsing
      -> docs/config      -> format-aware splitting
   -> validate resulting chunk size
   -> oversized chunk? -> recursive fallback
```

Principle: **structure first, size enforcement second**.

## 17. Why Tree-sitter
Tree-sitter is the primary parser for programming languages because it recognizes real syntax structures instead of arbitrary character ranges.

Example:
```text
CartServlet.java
- class CartServlet
- init()
- doGet()
- doPost()
```

This produces meaningful RAG chunks such as:
```text
chunk_type = method
symbol_name = doPost
```

GitHub also uses the open-source Tree-sitter library for code navigation and symbol/reference extraction, so this is an industry-relevant architectural direction.

## 18. Generic Tree-sitter architecture
The project started with Java-specific parsing, then generalized it.

Avoid creating fully duplicated modules such as:
```text
java_chunker.py
python_chunker.py
javascript_chunker.py
...
```

Preferred design:
```text
tree_sitter_chunker.py      -> common parsing/extraction engine
tree_sitter_profiles.py     -> language-specific profiles/configuration
```

Current direction uses `tree-sitter-language-pack` so parsers can be obtained through a common interface such as:
```python
get_parser("java")
get_parser("python")
get_parser("javascript")
get_parser("typescript")
get_parser("cpp")
```

## 19. CodeChunk model and exact source ranges
Use a model equivalent to:
```python
@dataclass(frozen=True)
class SourceRange:
    start_line: int
    end_line: int

@dataclass(frozen=True)
class CodeChunk:
    repository_id: str
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None
    symbol_start_line: int
    symbol_end_line: int
    source_ranges: tuple[SourceRange, ...]
    content: str
```

Why both symbol range and source ranges?

A class may occupy lines 16-88, while its class-context chunk may contain only lines 16 and 18 because methods are excluded.

Example:
```text
symbol_start_line = 16
symbol_end_line = 88
source_ranges = [16-16, 18-18]
```

Do not collapse this back into a misleading single line range.

## 20. Java structural chunking status
Java structural parsing has been tested successfully through the generic chunker.

Example output from `CartServlet.java`:
```text
class_context -> CartServlet
method -> init
method -> doGet
method -> doPost
```

Example method metadata:
```json
{
  "chunk_type": "method",
  "symbol_name": "doPost",
  "symbol_start_line": 52,
  "symbol_end_line": 87,
  "source_ranges": [
    {"start_line":52,"end_line":87}
  ]
}
```

## 21. Avoid class/method duplication
Do not embed an entire class including methods **and** then embed the methods separately.

Preferred:
```text
class_context
- annotations
- class declaration/header
- fields/direct context

method
- method body

constructor
- constructor body
```

Apply the same principle to interfaces, structs, traits, namespaces, modules, etc.

## 22. JavaScript/TypeScript limitation discovered
Initial JS/TS profiles only targeted traditional nodes such as:
```text
function_declaration
class_declaration
method_definition
```

Modern JS/TS frequently uses variable-based declarations.

Example controller pattern:
```javascript
const verifyScan = async (req, res) => {
    ...
};
```

Tree-sitter sees this approximately as:
```text
lexical_declaration
  -> variable_declarator
     -> name: identifier
     -> value: arrow_function
```

This must be captured as:
```text
chunk_type = function
symbol_name = verifyScan
```

## 23. Important JS/TS patterns to support
### Arrow functions
```javascript
const verifyScan = async (...) => {...};
```
-> `function`

### Function expressions
```javascript
const login = function(...) {...};
```
-> `function`

### Important top-level declarations
```javascript
const userSchema = new mongoose.Schema({...});
```
-> `declaration`, symbol `userSchema`

```javascript
const User = mongoose.model("User", userSchema);
```
-> `declaration`, symbol `User`

### React Native / TSX style declarations
```typescript
const styles = StyleSheet.create({...});
```
-> `declaration`, symbol `styles`

The full meaningful top-level declaration should be searchable.

## 24. Do not capture every local variable
Inside functions there may be:
```javascript
const data = ...
const result = ...
const user = ...
```

These generally should not become independent chunks.

Rule: capture important variable declarations primarily at module/top-level scope.

Useful top-level examples:
- exported functions
- schemas
- models
- major constants
- configuration objects
- style objects
- API clients
- components

## 25. JS/TS classification baseline
Current intended classification:
```text
arrow_function      -> function
function_expression -> function
call_expression     -> declaration
new_expression      -> declaration
object               -> declaration
array                -> declaration
```

Keep the logic conservative to avoid vector explosion.

## 26. Tree-sitter Queries — planned refinement
The long-term extractor should increasingly use Tree-sitter Queries instead of endlessly adding Python `if node.type == ...` rules.

Conceptual approach:
```text
Tree-sitter parser
   -> language-specific query
      -> @definition.function
      -> @definition.class
      -> @definition.method
      -> @name
   -> common extraction engine
```

Potential organization:
```text
app/rag/queries/
├── java.scm
├── python.scm
├── javascript.scm
├── typescript.scm
├── tsx.scm
├── c.scm
├── cpp.scm
├── c_sharp.scm
├── go.scm
├── rust.scm
└── ...
```

Do not migrate blindly; preserve tested behavior while converting languages gradually.

## 27. Intended programming-language coverage
Primary structural support target:
```text
Python
Java
JavaScript
JSX
TypeScript
TSX
C
C++
C#
Go
Rust
PHP
Ruby
Kotlin
Swift
Bash/Shell
```

HTML/CSS/SCSS/SQL may use structural parsers where useful, but should be evaluated separately.

## 28. Structured non-code file strategy
Not every file should use code-symbol parsing.

### Markdown
Heading-aware splitting.

### JSON
Object/key-aware splitting.

### YAML
Structure-aware sections, especially for Docker Compose and CI/CD.

### XML
Element-aware splitting, useful for `pom.xml` and configuration.

### TOML
Table/section-aware splitting, useful for `pyproject.toml` and Pipfile.

### Small cohesive files
Small `requirements.txt`, `Dockerfile`, and `Makefile` may remain whole unless too large.

## 29. Oversized structural chunks
A Tree-sitter function may still be hundreds of lines.

Use:
```text
Tree-sitter structural chunk
  -> if within size limit: keep
  -> if too large: recursive text splitter fallback
```

`RecursiveCharacterTextSplitter` is a fallback, not the primary code chunker.

Approximate starting fallback settings discussed:
```text
chunk_size ≈ 1500 chars
chunk_overlap ≈ 200 chars
```

These must later be evaluated rather than treated as permanent truth.

## 30. Duplicate chunk prevention
Avoid duplicate source chunks caused by overlapping extraction strategies.

Potential deduplication keys:
```text
file_path
chunk_type
symbol_name
symbol_start_line
symbol_end_line
```

Source byte ranges can be added later if useful.

Chunks should be sorted in source order for deterministic testing and debugging.

## 31. Development-only preview endpoints
Temporary endpoints have been used to inspect intermediate stages, including concepts such as:
```text
GET /api/repositories/preview-files
GET /api/repositories/preview-documents
GET /api/repositories/preview-code-chunks
```

These are development tools and need not remain in the final API.

Do not expose arbitrary filesystem paths in final endpoints. Use backend-generated `repository_id` / `chat_id` values.

## 32. Current development status
Completed and working:
```text
[done] FastAPI project foundation
[done] health endpoint
[done] GitHub URL validation
[done] repository ID generation
[done] chat ID generation
[done] repository cloning
[done] clone failure handling
[done] partial clone cleanup
[done] repository scanning
[done] supported-file detection
[done] ignored file/directory rules
[done] language detection
[done] scan summary
[done] safe source-file reading
[done] RepositoryFile model
[done] LangChain Document conversion test
[done] Tree-sitter Java parsing
[done] Java class/method extraction
[done] Java class-context de-duplication
[done] exact SourceRange model
[done] generic Tree-sitter architecture
[done] Java tested successfully through generic chunker
[in progress] JavaScript/TypeScript/TSX structural coverage
```

## 33. Immediate next tasks
### A. Finish Tree-sitter structural extraction
1. JS arrow functions
2. JS function expressions
3. top-level JS declarations
4. TypeScript/TSX equivalents
5. JSX/React component patterns
6. Python
7. C/C++
8. C#
9. Go
10. Rust
11. PHP
12. Ruby
13. Kotlin
14. Swift
15. Shell/Bash
16. graceful parser fallback

### B. Move toward Tree-sitter queries
Add query-based extraction where it improves correctness and maintainability.

### C. Structured non-code chunkers
Implement Markdown, JSON, YAML, XML, TOML, then special handling for Dockerfile/Makefile/requirements.

### D. Size validation
Add recursive fallback for oversized chunks.

### E. Final LangChain Documents
Final chunk metadata should include:
```text
repository_id
file_path
language
chunk_type
symbol_name
symbol_start_line
symbol_end_line
source_ranges
```

### F. Embeddings
Choose one embedding model and use the same model for indexing and querying. Batch embedding calls when possible.

### G. Pinecone
Implement:
- index/config setup
- one repository namespace or equivalent isolation
- vector upsert
- metadata storage
- semantic search
- delete repository vectors/namespace

### H. Retrieval testing
Before LLM RAG, directly test whether top-k retrieval returns the right chunks.

Example questions:
```text
Where is authentication handled?
Where is the cart updated?
Which file defines the User model?
Where are attendance logs retrieved?
```

### I. Basic RAG
```text
question
 -> query embedding
 -> Pinecone retrieval
 -> relevant chunks
 -> prompt
 -> LLM
 -> grounded answer + sources
```

### J. Chat persistence
Store chat/repository relationship and messages. SQLite is sufficient initially.

### K. React frontend
Build URL input, processing UI, chat sidebar, chat interface, citations, delete chat.

### L. Streaming
Use FastAPI streaming/SSE-style response flow.

### M. Cleanup
Deleting a chat must remove local repository + Pinecone vectors + messages + metadata.

### N. LangGraph
Only after core RAG is reliable.

### O. LangSmith
Add tracing and evaluation after the system has meaningful retrieval behavior.

## 34. RAG quality principles
Optimize for:
```text
retrieval correctness
grounded answers
source traceability
low duplicate context
repository isolation
useful structural chunks
```

Do not optimize merely for “the chatbot returns some answer.”

## 35. Metadata is first-class
Never casually discard:
```text
repository_id
file_path
language
chunk_type
symbol_name
source ranges
```

These are needed for citations, filtering, deletion, debugging, evaluation, code navigation, and future tools.

## 36. Future code-graph direction
Once structural extraction is reliable, definitions/references can support relationships such as:
```text
controller -> service
service -> repository
function -> function call
class -> class
file -> import
module -> dependency
```

Potential questions:
- Explain the complete login flow.
- What calls UserService?
- Where is CartService used?
- Trace this API request to the database.

Do not build the graph before core RAG works.

## 37. Error-handling philosophy
External operations can fail:
```text
Git clone
file read
Tree-sitter parse
embedding API
Pinecone upsert
LLM API
cleanup
```

The application should:
- fail clearly
- return useful errors
- clean partial resources when appropriate
- avoid orphaned state
- never return success for incomplete operations

## 38. Development philosophy
Build one layer at a time:
```text
clone -> test
scan -> test
chunk -> inspect
embed -> test
Pinecone retrieval -> inspect
RAG -> evaluate
agents -> later
```

Do not hide weak retrieval behind an LLM.

## 39. Code-quality expectations
Prefer:
- type hints
- clear names
- small functions
- explicit models
- thin routes
- separated services
- dedicated exceptions
- deterministic behavior
- comments for non-obvious design decisions
- tests for grammar-specific assumptions

Avoid:
- giant route handlers
- duplicated parser implementations
- hard-coded absolute paths everywhere
- arbitrary shell execution
- secret leakage
- premature microservices
- premature authentication
- premature PostgreSQL
- premature LangGraph complexity

## 40. Important decisions not to reverse accidentally
1. One repository = one chat.
2. No user authentication/multi-user requirement.
3. Repository remains local until chat deletion.
4. Chat deletion cleans repository + Pinecone vectors + metadata.
5. Pinecone is the intended vector database.
6. FastAPI is the backend.
7. LangChain is the RAG orchestration layer.
8. LangGraph comes after basic RAG.
9. Tree-sitter is the primary programming-code structural parser.
10. Structural boundaries come before arbitrary character splitting.
11. Avoid whole-class + method duplication.
12. Preserve exact source metadata.
13. JS/TS must support modern patterns such as arrow functions.
14. Different file formats use appropriate chunking strategies.
15. Retrieval is tested before relying on final LLM answers.

## 41. Agent workflow guidance
Before major changes:
1. Read this file.
2. Inspect the current implementation.
3. Extend the existing architecture rather than replacing it casually.
4. Preserve tested behavior.
5. When assuming a Tree-sitter node type, inspect actual parser output instead of guessing.
6. Add focused tests for language-specific syntax patterns.
7. Keep APIs backward-compatible where practical.

## 42. Final target architecture
```text
React + TypeScript
        |
      FastAPI
        |
  +-----+--------------------+
  |                          |
Repository Service        RAG Service
  |                          |
Local Git Repo               |
  |                          |
Scanner                       |
  |                          |
Chunking Router               |
 /          \                 |
Tree-sitter  Format parsers   |
 \          /                 |
  Final Chunks                |
      |                       |
  Embeddings                  |
      |                       |
   Pinecone <---- Retriever --+
                     |
                    LLM
                     |
           Grounded answer + sources
```

Later:
```text
FastAPI
  -> LangGraph
     -> semantic search tool
     -> read-file tool
     -> find-symbol tool
     -> dependency inspection
     -> repository analysis
```

## 43. Successful MVP definition
The MVP is successful when:
1. User enters a public GitHub repository URL.
2. Repository clones locally.
3. Useful files are scanned and read.
4. Code is structurally chunked.
5. Chunks are embedded.
6. Vectors are stored in Pinecone.
7. User asks a repository-specific question.
8. Correct chunks are retrieved.
9. LLM answers using repository evidence.
10. Answer includes useful source references.
11. Repositories stay isolated by chat.
12. Deleting a chat removes local repository + vectors + metadata.

## 44. Final project identity
Do not treat this as a generic “chat with files” project.

It is a **Codebase Intelligence Assistant**.

The repository-intelligence layer should be strong enough to support future capabilities such as:
- semantic code search
- symbol search
- architecture understanding
- dependency tracing
- bug investigation
- code navigation
- documentation generation

The quality of parsing, chunking, metadata, retrieval, and grounded explanations matters more than merely generating chatbot responses.
