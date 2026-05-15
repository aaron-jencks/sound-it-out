from logging import Logger
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

from phonemizer.backend.espeak.espeak import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation


DEFAULT_PUNCTUATION = Punctuation.default_marks()
PLACEHOLDER_PUNCTUATION = "<>/,`~"
LEGACY_LATIN_ONLY_PUNCT = r"[^'a-zA-Z0-9\s]"


def _dedupe_chars(chars: str) -> str:
    return "".join(dict.fromkeys(chars))


BASE_PUNCTUATION = _dedupe_chars(DEFAULT_PUNCTUATION + PLACEHOLDER_PUNCTUATION)

LANGUAGE_PUNCTUATION: Dict[str, str] = {
    "en": BASE_PUNCTUATION,
    "es": _dedupe_chars(BASE_PUNCTUATION + "¡¿"),
    "ru": _dedupe_chars(BASE_PUNCTUATION + "„‚‹›"),
    "pl": _dedupe_chars(BASE_PUNCTUATION + "„‚‹›"),
    "ta": _dedupe_chars(BASE_PUNCTUATION + "।॥"),
    "ml": _dedupe_chars(BASE_PUNCTUATION + "।॥"),
    "hi": _dedupe_chars(BASE_PUNCTUATION + "।॥"),
    "ur": _dedupe_chars(BASE_PUNCTUATION + "،؛؟۔"),
}

SCRIPT_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "latin": (
        (0x0041, 0x005A),
        (0x0061, 0x007A),
        (0x00C0, 0x00FF),
        (0x0100, 0x024F),
        (0x1E00, 0x1EFF),
        (0x2C60, 0x2C7F),
        (0xA720, 0xA7FF),
        (0xAB30, 0xAB6F),
    ),
    "cyrillic": (
        (0x0400, 0x04FF),
        (0x0500, 0x052F),
        (0x1C80, 0x1C8F),
        (0x2DE0, 0x2DFF),
        (0xA640, 0xA69F),
    ),
    "greek": (
        (0x0370, 0x03FF),
        (0x1F00, 0x1FFF),
    ),
    "arabic": (
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x0870, 0x089F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ),
    "devanagari": (
        (0x0900, 0x097F),
        (0xA8E0, 0xA8FF),
    ),
    "bengali": (
        (0x0980, 0x09FF),
    ),
    "tamil": (
        (0x0B80, 0x0BFF),
    ),
    "malayalam": (
        (0x0D00, 0x0D7F),
    ),
    "han": (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x30000, 0x3134F),
        (0x31350, 0x323AF),
    ),
}

LANGUAGE_TO_SCRIPT = {
    "en": "latin",
    "es": "latin",
    "fr": "latin",
    "pl": "latin",
    "ru": "cyrillic",
    "bg": "cyrillic",
    "el": "greek",
    "ar": "arabic",
    "ur": "arabic",
    "hi": "devanagari",
    "bn": "bengali",
    "ta": "tamil",
    "ml": "malayalam",
    "cmn": "han",
    "zh": "han",
}


def int_to_token(i: int, vocab: str) -> str:
    if i < 0:
        raise ValueError("Only non-negative integers are allowed.")
    base = len(vocab)
    if i < base:
        return vocab[i]
    result = []
    while i >= base:
        result.append(vocab[i % base])
        i = i // base - 1
    result.append(vocab[i])
    return "".join(reversed(result))


def normalize_language(language: str) -> str:
    return language.lower().split("-", 1)[0].split("_", 1)[0]


def resolve_script(language: str) -> Optional[str]:
    return LANGUAGE_TO_SCRIPT.get(normalize_language(language))


def char_in_script(ch: str, script: Optional[str]) -> bool:
    if not script:
        return False
    codepoint = ord(ch)
    for start, end in SCRIPT_RANGES[script]:
        if start <= codepoint <= end:
            return True
    return False


def needs_rtl_isolation(language: str) -> bool:
    return resolve_script(language) == "arabic"


class CustomEspeakBackend(EspeakBackend):
    def __init__(
        self,
        language: str,
        punct: Optional[str] = None,
        preserve_regex: Optional[List[str]] = None,
        token_vocab: str = ",`~",
        preserve_punctuation: bool = False,
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
        logger: Optional[Logger] = None,
        rtl_preserve: Optional[bool] = None,
        preserve_non_target_script: bool = True,
        preserve_numbers: bool = True,
        preserve_symbols: bool = True,
    ):
        self.language_code = language
        self.script = resolve_script(language)
        self.token_index = 0
        self.token_vocab = token_vocab
        self.token_prefix = "<<<{}>"
        self.token_suffix = "</{}>>>"
        self.regex = re.compile("|".join([f"({pattern})" for pattern in preserve_regex])) if preserve_regex else None
        self.mappings: Dict[str, int] = {}
        self.rtl_preserve = needs_rtl_isolation(language) if rtl_preserve is None else rtl_preserve
        self.preserve_non_target_script = preserve_non_target_script
        self.preserve_numbers = preserve_numbers
        self.preserve_symbols = preserve_symbols

        punctuation_marks = punct
        if punctuation_marks is None or punctuation_marks == LEGACY_LATIN_ONLY_PUNCT:
            punctuation_marks = LANGUAGE_PUNCTUATION.get(
                normalize_language(language),
                BASE_PUNCTUATION,
            )

        super().__init__(
            language,
            punctuation_marks=punctuation_marks,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress,
            tie=tie,
            language_switch=language_switch,
            words_mismatch=words_mismatch,
            logger=logger,
        )

    def phonemize(self, text: List[str]) -> List[str]:
        result = []
        for txt in text:
            self.mappings = {}
            self.token_index = 0
            pre_processed = self.pre_process(txt)
            phonemized = super(CustomEspeakBackend, self).phonemize([pre_processed])[0]
            result.append(self.post_process(phonemized))
        return result

    def _register_token(self, preserved_text: str) -> str:
        if preserved_text in self.mappings:
            encoded_token = int_to_token(self.mappings[preserved_text], self.token_vocab)
            return f"{self.token_prefix.format(encoded_token)} {preserved_text} {self.token_suffix.format(encoded_token)}"

        encoded_token = int_to_token(self.token_index, self.token_vocab)
        token = f"{self.token_prefix.format(encoded_token)} {preserved_text} {self.token_suffix.format(encoded_token)}"
        self.mappings[preserved_text] = self.token_index
        self.token_index += 1
        return token

    def _should_preserve_char(self, ch: str, previous_preserved: bool) -> bool:
        if ch.isspace():
            return False

        category = unicodedata.category(ch)
        if self.preserve_numbers and category.startswith("N"):
            return True
        if category.startswith("M"):
            return previous_preserved
        if category == "Cf":
            return previous_preserved
        if category.startswith("P"):
            return ch not in {"'", "’", "-", "‐", "‑"}
        if self.preserve_symbols and category.startswith("S"):
            return True
        if category.startswith("L"):
            if not self.preserve_non_target_script:
                return False
            return not char_in_script(ch, self.script)
        return False

    def _automatic_preserve_spans(self, txt: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        start: Optional[int] = None
        previous_preserved = False

        for idx, ch in enumerate(txt):
            preserve = self._should_preserve_char(ch, previous_preserved)
            if preserve and start is None:
                start = idx
            elif not preserve and start is not None:
                spans.append((start, idx))
                start = None
            previous_preserved = preserve

        if start is not None:
            spans.append((start, len(txt)))

        return spans

    def _regex_preserve_spans(self, txt: str) -> List[Tuple[int, int]]:
        if not self.regex:
            return []
        return [(match.start(), match.end()) for match in self.regex.finditer(txt)]

    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not spans:
            return []
        spans = sorted(spans)
        merged: List[Tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def pre_process(self, txt: str) -> str:
        spans = self._merge_spans(self._regex_preserve_spans(txt) + self._automatic_preserve_spans(txt))
        if not spans:
            return txt

        pieces: List[str] = []
        cursor = 0
        for start, end in spans:
            if cursor < start:
                pieces.append(txt[cursor:start])
            pieces.append(self._register_token(txt[start:end]))
            cursor = end
        if cursor < len(txt):
            pieces.append(txt[cursor:])
        return "".join(pieces)

    def post_process(self, phoneme: str) -> str:
        for token, index in self.mappings.items():
            index_str = int_to_token(index, self.token_vocab)
            prefix = self.token_prefix.format(index_str)
            suffix = self.token_suffix.format(index_str)

            if prefix not in phoneme:
                raise Exception(f"phoneme {prefix} not in {phoneme}")
            if suffix not in phoneme:
                raise Exception(f"phoneme {suffix} not in {phoneme}")

            replacement = token
            if self.rtl_preserve:
                replacement = f"\u2067{token}\u2069"

            phoneme = re.sub(f"{re.escape(prefix)}.*?{re.escape(suffix)}", lambda _: replacement, phoneme)

        return phoneme
