from collections import Counter
from pathlib import Path

from app.models.repository_file import RepositoryFile
from app.models.repository_scan import (
    RepositoryScanResult,
    RepositoryScanSummary,
)

MAX_FILE_SIZE_BYTES = 1_000_000

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "Python",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript JSX",
    ".ts": "TypeScript",
    ".tsx": "TypeScript TSX",
    ".c": "C",
    ".h": "C Header",
    ".cpp": "C++",
    ".cc": "C++",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".php": "PHP",
    ".rb": "Ruby",
    ".kt": "Kotlin",
    ".kts": "Kotlin Script",
    ".swift": "Swift",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".xml": "XML",
    ".sql": "SQL",
    ".sh": "Shell",
}


SUPPORTED_FILENAMES: dict[str, str] = {
    "Dockerfile": "Dockerfile",
    "Makefile": "Makefile",
    "requirements.txt": "Python Requirements",
    "pyproject.toml": "Python Project Configuration",
    "Pipfile": "Python Pipfile",
    "package.json": "Node Package Configuration",
    "tsconfig.json": "TypeScript Configuration",
    "pom.xml": "Maven Configuration",
    "build.gradle": "Gradle Configuration",
    "build.gradle.kts": "Gradle Kotlin Configuration",
    "settings.gradle": "Gradle Configuration",
    "settings.gradle.kts": "Gradle Kotlin Configuration",
    "docker-compose.yml": "Docker Compose",
    "docker-compose.yaml": "Docker Compose",
}


IGNORED_DIRECTORIES: set[str] = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
    ".next",
    ".nuxt",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".gradle",
    "bin",
    "obj",
    "vendor",
}


IGNORED_FILENAMES: set[str] = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "poetry.lock",
}


IGNORED_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".jar",
    ".war",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".pem",
    ".key",
    ".crt",
}


def scan_repository(
    repository_path: Path,
) -> RepositoryScanResult:
    if not repository_path.exists():
        raise FileNotFoundError(
            f"Repository path does not exist: {repository_path}"
        )

    if not repository_path.is_dir():
        raise NotADirectoryError(
            f"Repository path is not a directory: {repository_path}"
        )

    total_files = 0
    supported_files = 0
    language_counter: Counter[str] = Counter()
    repository_files: list[RepositoryFile] = []

    for file_path in repository_path.rglob("*"):
        if should_ignore_path(file_path, repository_path):
            continue

        if not file_path.is_file():
            continue

        total_files += 1

        language = detect_language(file_path)

        if language is None:
            continue

        repository_file = create_repository_file(
            file_path=file_path,
            repository_root=repository_path,
            language=language,
        )

        if repository_file is None:
            continue

        repository_files.append(repository_file)
        supported_files += 1
        language_counter[language] += 1

    ignored_files = total_files - supported_files

    summary = RepositoryScanSummary(
        total_files=total_files,
        supported_files=supported_files,
        ignored_files=ignored_files,
        languages=dict(
            sorted(
                language_counter.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    )

    return RepositoryScanResult(
        summary=summary,
        files=repository_files,
    )

def create_repository_file(
    file_path: Path,
    repository_root: Path,
    language: str,
) -> RepositoryFile | None:
    try:
        size_bytes = file_path.stat().st_size
    except OSError:
        return None

    if size_bytes == 0:
        return None

    if size_bytes > MAX_FILE_SIZE_BYTES:
        return None

    content = read_text_file(file_path)

    if content is None:
        return None

    if not content.strip():
        return None

    relative_path = file_path.relative_to(repository_root)

    return RepositoryFile(
        relative_path=relative_path.as_posix(),
        language=language,
        size_bytes=size_bytes,
        content=content,
    )


def should_ignore_path(
    file_path: Path,
    repository_root: Path,
) -> bool:
    relative_path = file_path.relative_to(repository_root)

    if any(
        path_part in IGNORED_DIRECTORIES
        for path_part in relative_path.parts[:-1]
    ):
        return True

    if file_path.name in IGNORED_FILENAMES:
        return True

    if file_path.suffix.lower() in IGNORED_EXTENSIONS:
        return True

    return False


def detect_language(file_path: Path) -> str | None:
    if file_path.name in SUPPORTED_FILENAMES:
        return SUPPORTED_FILENAMES[file_path.name]

    return SUPPORTED_EXTENSIONS.get(
        file_path.suffix.lower()
    )


def read_text_file(file_path: Path) -> str | None:
    encodings = (
        "utf-8",
        "utf-8-sig",
        "latin-1",
    )

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)

        except UnicodeDecodeError:
            continue

        except OSError:
            return None

    return None


def create_file_previews(
    repository_path: Path,
    preview_length: int = 200,
) -> list[dict[str, str | int]]:
    scan_result = scan_repository(repository_path)

    return [
        {
            "relative_path": repository_file.relative_path,
            "language": repository_file.language,
            "size_bytes": repository_file.size_bytes,
            "content_preview": repository_file.content[:preview_length],
        }
        for repository_file in scan_result.files
    ]