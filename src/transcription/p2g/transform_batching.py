from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BatchTransformResult:
    outputs: list[Optional[str]]
    event_counts: Counter[str] = field(default_factory=Counter)


class LanguageBatchCollector:
    def __init__(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self._buffers: dict[str, list[str]] = defaultdict(list)

    def add(self, language: str, text: str) -> None:
        self._buffers[language].append(text)

    def should_flush(self, language: str, remaining_quota: int) -> bool:
        current = len(self._buffers.get(language, []))
        if current == 0:
            return False
        if current >= self.batch_size:
            return True
        return remaining_quota > 0 and current >= remaining_quota

    def pop(self, language: str) -> list[str]:
        texts = self._buffers.pop(language, [])
        return texts

    def pop_all(self, languages: Optional[list[str]] = None) -> list[tuple[str, list[str]]]:
        if languages is None:
            keys = list(self._buffers.keys())
        else:
            keys = [language for language in languages if language in self._buffers]

        popped: list[tuple[str, list[str]]] = []
        for language in keys:
            texts = self.pop(language)
            if texts:
                popped.append((language, texts))
        return popped


class PhonemizeBatchRunner:
    def __init__(self, supervisor):
        self.supervisor = supervisor

    def run_batch(self, texts: list[str], language: str) -> BatchTransformResult:
        outputs = self.supervisor.transcribe_batch(texts, language)
        event_counts = Counter()

        if self.supervisor.last_batch_used_fallback:
            event_counts["phonemizer_batch_fallback"] += 1
        if self.supervisor.last_batch_failure_kind is not None:
            event_counts[f"phonemizer_{self.supervisor.last_batch_failure_kind}"] += 1
        event_counts.update({
            f"phonemizer_{key}": count
            for key, count in self.supervisor.last_item_failure_counts.items()
        })

        return BatchTransformResult(outputs=outputs, event_counts=event_counts)


class RomanizeBatchRunner:
    def __init__(self, uroman_path: Path, perl_path: str):
        self.uroman_path = str(uroman_path)
        self.perl_path = perl_path

    def run_batch(self, texts: list[str], language: str) -> BatchTransformResult:
        del language
        from transcription.g2p import phonemize_romanize

        try:
            return BatchTransformResult(
                outputs=phonemize_romanize.uromanize_batch(texts, self.uroman_path, self.perl_path)
            )
        except Exception:
            outputs: list[Optional[str]] = []
            event_counts = Counter({"romanize_batch_fallback": 1})
            for text in texts:
                try:
                    outputs.append(
                        phonemize_romanize.uromanize_batch([text], self.uroman_path, self.perl_path)[0]
                    )
                except Exception:
                    outputs.append(None)
                    event_counts["romanize_transform_error"] += 1
            return BatchTransformResult(outputs=outputs, event_counts=event_counts)
