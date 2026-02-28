from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    words: Tuple[str, ...]
    groups: Tuple[Tuple[int, int, int, int], ...]
    labels: Tuple[str, ...]


@dataclass(frozen=True)
class PuzzleBank:
    puzzles: Tuple[Puzzle, ...]
    dataset_hash_full: str


def _validate_word(word: str) -> str:
    clean = word.strip().lower()
    if not clean or not clean.isalpha():
        raise ValueError(f"Invalid puzzle word: {word!r}. Words must be non-empty alphabetic tokens.")
    return clean


def _normalize_puzzle(raw: Dict[str, object]) -> Puzzle:
    if "id" not in raw:
        raise ValueError("Puzzle missing 'id'.")
    puzzle_id = str(raw["id"]).strip()
    if not puzzle_id:
        raise ValueError("Puzzle id cannot be empty.")

    raw_groups = raw.get("groups")
    if not isinstance(raw_groups, list) or len(raw_groups) != 4:
        raise ValueError(f"Puzzle {puzzle_id}: expected exactly 4 groups.")

    words: List[str] = []
    labels: List[str] = []
    groups: List[Tuple[int, int, int, int]] = []
    word_to_index: Dict[str, int] = {}

    for group_idx, group_obj in enumerate(raw_groups):
        if not isinstance(group_obj, dict):
            raise ValueError(f"Puzzle {puzzle_id}: group {group_idx} must be an object.")
        label = str(group_obj.get("label", f"group_{group_idx}")).strip()
        labels.append(label)

        raw_words = group_obj.get("words")
        if not isinstance(raw_words, list) or len(raw_words) != 4:
            raise ValueError(f"Puzzle {puzzle_id}: each group must contain exactly 4 words.")

        group_indices: List[int] = []
        for word in raw_words:
            clean = _validate_word(str(word))
            if clean in word_to_index:
                raise ValueError(f"Puzzle {puzzle_id}: duplicate word {clean!r}.")
            word_to_index[clean] = len(words)
            words.append(clean)
            group_indices.append(word_to_index[clean])

        groups.append(tuple(group_indices))

    if len(words) != 16:
        raise ValueError(f"Puzzle {puzzle_id}: expected 16 total words, got {len(words)}.")

    return Puzzle(
        puzzle_id=puzzle_id,
        words=tuple(words),
        groups=tuple(groups),
        labels=tuple(labels),
    )


def load_default_puzzle_bank() -> PuzzleBank:
    data_dir = files("connections_rl.data")
    raw_text = data_dir.joinpath("puzzles.json").read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("connections_rl.data/puzzles.json must contain a non-empty list.")

    puzzles = tuple(_normalize_puzzle(item) for item in parsed)
    seen = set()
    for puzzle in puzzles:
        if puzzle.puzzle_id in seen:
            raise ValueError(f"Duplicate puzzle id: {puzzle.puzzle_id}")
        seen.add(puzzle.puzzle_id)

    digest = sha256(raw_text.encode("utf-8")).hexdigest()
    return PuzzleBank(puzzles=puzzles, dataset_hash_full=digest)


def make_puzzle_bank(puzzles: Sequence[Dict[str, object]]) -> PuzzleBank:
    if not puzzles:
        raise ValueError("puzzles must not be empty.")
    normalized = tuple(_normalize_puzzle(item) for item in puzzles)
    seen = set()
    for puzzle in normalized:
        if puzzle.puzzle_id in seen:
            raise ValueError(f"Duplicate puzzle id: {puzzle.puzzle_id}")
        seen.add(puzzle.puzzle_id)

    canonical = json.dumps(puzzles, sort_keys=True, separators=(",", ":"))
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    return PuzzleBank(puzzles=normalized, dataset_hash_full=digest)
