"""Context retrieval — reads configured sources and fits to token budget."""

import glob
from pathlib import Path

from tokens import count_tokens


class ContextRetriever:
    """Reads context sources from configured paths/globs.

    MVP: reads all sources every turn, truncates to fit budget.
    Future: keyword/semantic relevance scoring per turn.
    """

    def __init__(self, sources: list[str]):
        self.sources = sources

    def _resolve_paths(self) -> list[Path]:
        """Expand globs and resolve all source paths."""
        paths = []
        for source in self.sources:
            expanded = Path(source).expanduser()
            if "*" in source or "?" in source:
                for match in sorted(glob.glob(str(expanded), recursive=True)):
                    p = Path(match)
                    if p.is_file():
                        paths.append(p)
            elif expanded.is_file():
                paths.append(expanded)
            elif expanded.is_dir():
                for p in sorted(expanded.rglob("*.md")):
                    paths.append(p)
        return paths

    def retrieve(self, query: str, token_budget: int) -> str:
        """Retrieve context from sources, fitting within token budget.

        Args:
            query: the current user turn (unused in MVP, for future relevance scoring)
            token_budget: max tokens for the assembled context

        Returns:
            assembled context string
        """
        paths = self._resolve_paths()
        if not paths:
            return ""

        chunks = []
        total_tokens = 0

        for path in paths:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            # Header with source path for traceability
            chunk = f"--- {path} ---\n{content}\n"
            chunk_tokens = count_tokens(chunk)

            if total_tokens + chunk_tokens > token_budget:
                # Fit what we can from this file
                remaining = token_budget - total_tokens
                if remaining > 100:  # worth including a partial
                    truncated = content[: remaining * 4]  # rough char estimate
                    chunks.append(f"--- {path} (truncated) ---\n{truncated}\n")
                break

            chunks.append(chunk)
            total_tokens += chunk_tokens

        return "\n".join(chunks)
