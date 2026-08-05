from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse
from uuid import uuid4

from app.core.exceptions import (
    RepositoryCloneError,
    RepositoryScanError,
)
from app.models.repository_scan import RepositoryScanSummary
from app.services.repository_scanner import scan_repository


REPOSITORY_STORAGE_PATH = Path("storage/repositories")


@dataclass(frozen=True)
class RepositoryDetails:
    repository_id: str
    chat_id: str
    repository_name: str
    repository_owner: str
    repository_url: str
    local_path: str
    status: str
    scan_summary: RepositoryScanSummary


def create_repository(repository_url: str) -> RepositoryDetails:
    normalized_url = repository_url.rstrip("/")

    parsed_url = urlparse(normalized_url)
    path_parts = [part for part in parsed_url.path.split("/") if part]

    repository_owner = path_parts[0]
    repository_name = remove_git_suffix(path_parts[1])

    repository_id = f"repo_{uuid4().hex[:8]}"
    chat_id = f"chat_{uuid4().hex[:8]}"

    local_path = REPOSITORY_STORAGE_PATH / repository_id

    clone_repository(
        repository_url=normalized_url,
        destination=local_path,
    )

    try:
        scan_summary = scan_repository(local_path)

    except (
        OSError,
        FileNotFoundError,
        NotADirectoryError,
    ) as error:
        shutil.rmtree(local_path, ignore_errors=True)

        raise RepositoryScanError(
            "The repository was cloned, but its files could not be scanned."
        ) from error

    return RepositoryDetails(
        repository_id=repository_id,
        chat_id=chat_id,
        repository_name=repository_name,
        repository_owner=repository_owner,
        repository_url=normalized_url,
        local_path=str(local_path),
        status="scanned",
        scan_summary=scan_summary,
    )


def clone_repository(repository_url: str, destination: Path) -> None:
    REPOSITORY_STORAGE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        raise RepositoryCloneError(
            "The repository storage directory already exists."
        )

    command = [
        "git",
        "clone",
        "--depth",
        "1",
        repository_url,
        str(destination),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.returncode != 0:
            error_message = result.stderr.strip()

            raise RepositoryCloneError(
                format_clone_error(error_message)
            )

    except subprocess.TimeoutExpired as error:
        raise RepositoryCloneError(
            "Repository cloning timed out. The repository may be too large "
            "or the network connection may be slow."
        ) from error

    except FileNotFoundError as error:
        raise RepositoryCloneError(
            "Git is not installed or is not available in the system PATH."
        ) from error

    finally:
        # If Git created the directory but cloning did not complete,
        # remove all partially downloaded files.
        if destination.exists() and not is_valid_git_repository(destination):
            shutil.rmtree(destination, ignore_errors=True)


def is_valid_git_repository(repository_path: Path) -> bool:
    git_directory = repository_path / ".git"

    return (
        repository_path.is_dir()
        and git_directory.is_dir()
    )


def format_clone_error(git_error: str) -> str:
    lower_error = git_error.lower()

    if "repository not found" in lower_error:
        return (
            "The GitHub repository was not found. "
            "It may not exist or may be private."
        )

    if "authentication failed" in lower_error:
        return (
            "Authentication failed. "
            "Only public GitHub repositories are currently supported."
        )

    if "could not resolve host" in lower_error:
        return (
            "GitHub could not be reached. "
            "Check your internet connection and try again."
        )

    if "unable to access" in lower_error:
        return (
            "The repository could not be accessed. "
            "Check the URL and your internet connection."
        )

    return "The repository could not be cloned."


def remove_git_suffix(repository_name: str) -> str:
    if repository_name.lower().endswith(".git"):
        return repository_name[:-4]

    return repository_name