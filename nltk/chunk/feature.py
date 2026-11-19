# Natural Language Toolkit: Feature-based Chunkers
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Feature-based and cascaded chunk parsers.

``RegexpChunkParser`` is a good fit when a chunk can be described by a
regular expression over part-of-speech tags.  Domain entities often need
lexical, morphological, and neighbouring-token evidence instead.  This
module provides:

* :func:`context_features`, which builds a dictionary of word, tag, shape,
  affix, and neighbouring-token features for one token;
* :class:`FeatureChunkRule`, whose predicate can inspect any of those
  features (and the output of earlier cascade stages);
* :class:`FeatureChunkStage` and :class:`CascadedChunker`, which apply
  rule sets in order while preserving chunks found by earlier stages; and
* :func:`evaluate_chunkers`, a small wrapper around ``ChunkScore`` for
  precision/recall/F1 comparisons on annotated corpora.

The parser implements ``ChunkParserI``, so tagged output from a POS tagger
can be passed directly to ``parse``::

    >>> from nltk.chunk.feature import CascadedChunker, FeatureChunkRule
    >>> from nltk.chunk.feature import FeatureChunkStage
    >>> drugs = FeatureChunkStage(
    ...     "DRUG",
    ...     [FeatureChunkRule(lambda f: f["lower"] in {"aspirin", "metformin"})],
    ... )
    >>> chunker = CascadedChunker([drugs])
    >>> tree = chunker.parse(
    ...     [("Patient", "NN"), ("took", "VBD"), ("aspirin", "NN")]
    ... )
    >>> tree
    Tree('S', [('Patient', 'NN'), ('took', 'VBD'), Tree('DRUG', [('aspirin', 'NN')])])

Rules are evaluated left to right and may be repeated in a stage.  A rule
creates at most one chunk per matching span, and spans never overlap an
existing chunk.  A later stage can use ``chunk_label``,
``left_chunk_label``, or ``right_chunk_label`` to refer to earlier output.
"""

from collections.abc import Mapping

from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.termsec import safe_print
from nltk.tree import Tree

_ACTIONS = frozenset({"chunk", "expand_left", "expand_right", "merge"})


def _word_and_tag(token):
    if not isinstance(token, (tuple, list)) or len(token) < 2:
        raise TypeError(
            "tagged tokens must be (word, tag) pairs, "
            f"got {type(token).__name__}: {token!r}"
        )
    return token[0], token[1]


def _as_text(value):
    return value if isinstance(value, str) else str(value)


def _word_shape(word):
    """Return a compact character-shape representation (e.g. ``Xxxx-dd``)."""
    chars = []
    for char in word:
        if char.isupper():
            chars.append("X")
        elif char.islower():
            chars.append("x")
        elif char.isdigit():
            chars.append("d")
        else:
            chars.append(char)
    return "".join(chars)


def _shape_class(word):
    """Return a coarse shape class, mirroring the named-entity chunker."""
    if not word:
        return "other"
    if all(char.isdigit() for char in word):
        return "number"
    if all(not char.isalnum() for char in word):
        return "punct"
    if word.istitle():
        return "upcase"
    if word.islower():
        return "downcase"
    if word.isupper():
        return "allcaps"
    return "mixedcase"


def _chunk_metadata(chunk_labels, chunk_spans=None):
    """Describe the chunk (if any) containing each token."""
    if chunk_spans is None:
        chunk_spans = []
        start = 0
        while start < len(chunk_labels):
            label = chunk_labels[start]
            end = start + 1
            while end < len(chunk_labels) and chunk_labels[end] == label:
                end += 1
            if label is not None:
                chunk_spans.append((start, end, label))
            start = end

    metadata = [None] * len(chunk_labels)
    for start, end, label in chunk_spans:
        for index in range(start, end):
            metadata[index] = {
                "label": label,
                "start": start,
                "end": end,
                "length": end - start,
            }
    return metadata


def _token_at(tokens, index):
    if index < 0 or index >= len(tokens):
        return None
    return tokens[index]


def _feature_for_token(tokens, index):
    word, tag = _word_and_tag(tokens[index])
    word_text = _as_text(word)
    lower = word_text.lower()
    return {
        "bias": True,
        "word": word,
        "lower": lower,
        "tag": tag,
        "shape": _shape_class(word_text),
        "word_shape": _word_shape(word_text),
        "word_length": len(word_text),
        "is_title": word_text.istitle(),
        "is_upper": word_text.isupper(),
        "is_lower": word_text.islower(),
        "is_digit": word_text.isdigit(),
        "is_punct": bool(word_text) and not any(c.isalnum() for c in word_text),
        "prefix1": word_text[:1],
        "prefix2": word_text[:2],
        "prefix3": word_text[:3],
        "prefix4": word_text[:4],
        "suffix1": word_text[-1:],
        "suffix2": word_text[-2:],
        "suffix3": word_text[-3:],
        "suffix4": word_text[-4:],
        "index": index,
        "sentence_length": len(tokens),
        "distance_from_end": len(tokens) - index - 1,
        "is_first": index == 0,
        "is_last": index == len(tokens) - 1,
    }


def _add_neighbour_features(features, tokens, index, metadata):
    for offset, prefix in ((-2, "prev2"), (-1, "prev"), (1, "next"), (2, "next2")):
        token = _token_at(tokens, index + offset)
        if token is None:
            features[f"{prefix}_word"] = None
            features[f"{prefix}_lower"] = None
            features[f"{prefix}_tag"] = None
            features[f"{prefix}_shape"] = None
            continue
        word, tag = _word_and_tag(token)
        word_text = _as_text(word)
        features[f"{prefix}_word"] = word
        features[f"{prefix}_lower"] = word_text.lower()
        features[f"{prefix}_tag"] = tag
        features[f"{prefix}_shape"] = _shape_class(word_text)

    current = metadata[index]
    features["in_chunk"] = current is not None
    features["chunk_label"] = current["label"] if current else None
    features["chunk_start"] = current["start"] if current else None
    features["chunk_end"] = current["end"] if current else None
    features["chunk_length"] = current["length"] if current else 0
    features["chunk_offset"] = index - current["start"] if current else 0
    if current:
        features["chunk_text"] = " ".join(
            _as_text(_word_and_tag(token)[0])
            for token in tokens[current["start"] : current["end"]]
        )
    else:
        features["chunk_text"] = None

    chunks = [
        (item["start"], item["end"], item["label"])
        for item in metadata
        if item is not None
    ]
    chunks = sorted(set(chunks))

    left = [chunk for chunk in chunks if chunk[1] <= index]
    right = [chunk for chunk in chunks if chunk[0] > index]
    left_chunk = left[-1] if left else None
    right_chunk = right[0] if right else None
    for prefix, chunk in (
        ("left", left_chunk),
        ("right", right_chunk),
        ("prev", left_chunk),
        ("next", right_chunk),
    ):
        features[f"{prefix}_chunk_label"] = chunk[2] if chunk else None
        if chunk:
            features[f"{prefix}_chunk_text"] = " ".join(
                _as_text(_word_and_tag(token)[0])
                for token in tokens[chunk[0] : chunk[1]]
            )
        else:
            features[f"{prefix}_chunk_text"] = None


def context_features(tagged_tokens, index, chunk_labels=None):
    """
    Return contextual features for ``tagged_tokens[index]``.

    The dictionary contains the current token's ``word``, ``tag``,
    ``lower``, ``shape``, ``word_shape``, prefixes/suffixes (1--4
    characters), and boolean properties; the two preceding and following
    tokens' word/tag/shape; and chunk fields (``chunk_label``,
    ``chunk_text``, ``left_chunk_label``, ``right_chunk_label``, and
    aligned start/end/length/offset values).

    :param tagged_tokens: A sentence of ``(word, tag)`` pairs.
    :param index: Index of the token to describe.
    :param chunk_labels: Optional sequence aligned with ``tagged_tokens``.
        Each item is a chunk label or ``None``.  This is useful when a
        feature rule needs to inspect chunks produced by an earlier stage.
    :return: A feature dictionary suitable for a ``FeatureChunkRule``.
    :rtype: dict
    """
    tokens = tuple(tagged_tokens)
    if not 0 <= index < len(tokens):
        raise IndexError(f"token index {index} out of range")
    if chunk_labels is None:
        chunk_labels = [None] * len(tokens)
    else:
        chunk_labels = list(chunk_labels)
        if len(chunk_labels) != len(tokens):
            raise ValueError("chunk_labels must align with tagged_tokens")

    metadata = _chunk_metadata(chunk_labels)
    features = _feature_for_token(tokens, index)
    _add_neighbour_features(features, tokens, index, metadata)
    return features


token_features = context_features
extract_features = context_features


def _condition_matches(condition, value):
    if callable(condition):
        return bool(condition(value))
    if hasattr(condition, "match") and not isinstance(condition, str):
        return bool(condition.match(value))
    if isinstance(condition, Mapping):
        return value in condition
    if isinstance(condition, (set, frozenset, list, tuple, range)):
        try:
            return value in condition
        except TypeError:
            return False
    return value == condition


class FeatureChunkRule:
    """
    A chunk rule backed by token/context features rather than tag regexes.

    ``predicate`` receives the feature dictionary returned by
    :func:`context_features` for the first token of a candidate span.
    ``continuation`` is checked for every following token in that span; if
    omitted, ``predicate`` is used.  ``max_length`` bounds each match
    (default: one token).

    ``action`` controls what a match does:

    ``"chunk"``
        Create a new chunk labeled by ``label`` (or the enclosing stage).
    ``"expand_right"``
        Append the match to the chunk immediately to its left.
    ``"expand_left"``
        Prepend the match to the chunk immediately to its right.
    ``"merge"``
        Merge adjacent left and right chunks with the match; if only one
        neighbour exists, expand it, and if neither exists, create a chunk.

    Expansion and merge preserve an existing chunk's label.  Set an explicit
    ``label`` if the expanded or merged chunk should be relabeled.
    """

    def __init__(
        self,
        predicate=None,
        description="",
        *,
        continuation=None,
        max_length=1,
        action="chunk",
        label=None,
        conditions=None,
    ):
        if predicate is None and conditions is None:
            raise TypeError("FeatureChunkRule requires a predicate or conditions")
        if action not in _ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_ACTIONS)}, got {action!r}"
            )
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive or None")
        if predicate is not None and not callable(predicate):
            raise TypeError("predicate must be callable")
        if continuation is not None and not callable(continuation):
            raise TypeError("continuation must be callable")
        if label is not None and not isinstance(label, str):
            raise TypeError("label must be a string or None")

        self.predicate = predicate
        self.description = description
        self.continuation = continuation
        self.max_length = max_length
        self.action = action
        self.label = label
        self.conditions = dict(conditions or {})

    @classmethod
    def from_features(cls, conditions, **kwargs):
        """
        Build a rule from exact values, collections, callables, or compiled
        patterns.  For example::

            FeatureChunkRule.from_features(
                {"lower": {"aspirin", "metformin"}, "tag": {"NN", "NNP"}}
            )
        """
        if not isinstance(conditions, Mapping):
            raise TypeError("conditions must be a mapping")
        return cls(conditions=conditions, **kwargs)

    def matches(self, features):
        """Return true when all configured conditions hold."""
        if self.predicate is not None and not self.predicate(features):
            return False
        for feature, expected in self.conditions.items():
            if not _condition_matches(expected, features.get(feature)):
                return False
        return True

    def continues(self, features):
        """Return true when a candidate chunk may include this token."""
        if self.continuation is not None:
            return bool(self.continuation(features))
        return self.matches(features)

    def descr(self):
        return self.description or f"{self.action} rule"

    def __repr__(self):
        details = []
        if self.label is not None:
            details.append(f"label={self.label!r}")
        details.append(f"action={self.action!r}")
        if self.max_length is not None:
            details.append(f"max_length={self.max_length!r}")
        return f"<FeatureChunkRule {self.descr()!r} ({', '.join(details)})>"


class FeatureChunkStage:
    """An ordered set of feature rules that emits one default chunk label."""

    def __init__(self, chunk_label, rules, name=None, loop=1):
        if not isinstance(chunk_label, str) or not chunk_label:
            raise TypeError("chunk_label must be a non-empty string")
        self.chunk_label = chunk_label
        self.rules = tuple(rules)
        if not all(isinstance(rule, FeatureChunkRule) for rule in self.rules):
            raise TypeError("rules must contain FeatureChunkRule instances")
        if not isinstance(loop, int) or loop < 1:
            raise ValueError("loop must be a positive integer")
        self.name = name or chunk_label
        self.loop = loop

    def __repr__(self):
        return f"<FeatureChunkStage {self.name!r}: {len(self.rules)} rules>"


class CascadedChunker(ChunkParserI):
    """
    Apply feature-rule stages in order, preserving earlier chunks.

    ``stages`` may be a sequence of :class:`FeatureChunkStage` objects,
    ``(chunk_label, rules)`` pairs, or other ``ChunkParserI`` instances.
    A mapping from label to rules is also accepted.  Input to :meth:`parse`
    is either tagged tokens or an existing chunk tree; the latter lets the
    chunker participate in a multi-stage annotation pipeline.
    """

    def __init__(self, stages, root_label="S", trace=0):
        if isinstance(stages, Mapping):
            stages = stages.items()
        converted = []
        for stage in stages:
            if isinstance(stage, (FeatureChunkStage, ChunkParserI)):
                converted.append(stage)
            elif isinstance(stage, (tuple, list)) and len(stage) == 2:
                converted.append(FeatureChunkStage(stage[0], stage[1]))
            else:
                raise TypeError(
                    "stages must be FeatureChunkStage objects, "
                    "ChunkParserI instances, or (chunk_label, rules) pairs"
                )
        self._stages = tuple(converted)
        self._root_label = root_label
        self._trace = trace

    def stages(self):
        return self._stages

    def parse(self, chunk_struct, trace=None):
        """
        Chunk tagged tokens or further chunk an existing tree.

        :rtype: Tree
        """
        if isinstance(chunk_struct, Tree):
            tree = Tree(chunk_struct.label(), list(chunk_struct))
        else:
            tokens = list(chunk_struct)
            tree = Tree(self._root_label, tokens)

        if trace is None:
            trace = self._trace
        if not tree:
            return Tree(self._root_label, [])

        self._validate_tokens(tree)
        for stage in self._stages:
            if not isinstance(stage, FeatureChunkStage):
                tree = stage.parse(tree)
                if trace:
                    safe_print(f"# {stage!r}:")
                    safe_print(tree)
                continue
            if trace:
                safe_print(f"# Stage: {stage.name}")
            for _ in range(stage.loop):
                for rule in stage.rules:
                    tree = self._apply_rule(tree, stage, rule)
                    if trace:
                        safe_print(f"# {rule.descr()}:")
                        safe_print(tree)
        return tree

    def parse_sents(self, sentences, *args, **kwargs):
        """Apply :meth:`parse` to each sentence (pipeline-friendly)."""
        return [self.parse(sentence, *args, **kwargs) for sentence in sentences]

    def _validate_tokens(self, tree):
        for token in tree.leaves():
            _word_and_tag(token)

    @staticmethod
    def _units(tree):
        units = []
        position = 0
        for child in tree:
            length = len(child.leaves()) if isinstance(child, Tree) else 1
            units.append([position, position + length, child])
            position += length
        return units

    @staticmethod
    def _labels_and_spans(units, token_count):
        labels = [None] * token_count
        spans = []
        for start, end, child in units:
            if isinstance(child, Tree):
                label = child.label()
                spans.append((start, end, label))
                for index in range(start, end):
                    labels[index] = label
        return labels, spans

    def _features(self, tokens, index, labels, spans):
        metadata = _chunk_metadata(labels, spans)
        features = _feature_for_token(tokens, index)
        _add_neighbour_features(features, tokens, index, metadata)
        return features

    @staticmethod
    def _candidate_end(rule, tokens, start, labels, feature_fn):
        if rule.max_length == 1:
            return start + 1
        stop = len(tokens)
        if rule.max_length is not None:
            stop = min(stop, start + rule.max_length)
        end = start + 1
        while end < stop and labels[end] is None:
            if not rule.continues(feature_fn(end)):
                break
            end += 1
        return end

    def _find_match(self, rule, units, tokens, labels, spans, start_at=0):
        for start in range(start_at, len(tokens)):
            if labels[start] is not None:
                continue
            features = self._features(tokens, start, labels, spans)
            if not rule.matches(features):
                continue
            end = self._candidate_end(
                rule,
                tokens,
                start,
                labels,
                lambda index: self._features(tokens, index, labels, spans),
            )
            left = next(
                (index for index, unit in enumerate(units) if unit[1] == start),
                None,
            )
            right = next(
                (index for index, unit in enumerate(units) if unit[0] == end),
                None,
            )
            if rule.action == "expand_right" and left is None:
                continue
            if rule.action == "expand_left" and right is None:
                continue
            return start, end, left, right
        return None

    @staticmethod
    def _unit_indices_for_span(units, start, end):
        indices = set()
        for index, unit in enumerate(units):
            if unit[0] >= start and unit[1] <= end and unit[0] < unit[1]:
                indices.add(index)
        return indices

    def _apply_match(self, rule, stage, units, match):
        start, end, left_index, right_index = match
        candidate_indices = self._unit_indices_for_span(units, start, end)
        candidate = [units[index][2] for index in sorted(candidate_indices)]
        label = rule.label or stage.chunk_label
        action = rule.action
        left_chunk = units[left_index][2] if left_index is not None else None
        right_chunk = units[right_index][2] if right_index is not None else None

        if action == "merge" and left_chunk and right_chunk:
            label = rule.label or left_chunk.label()
            node = Tree(label, list(left_chunk) + candidate + list(right_chunk))
            resume = units[right_index][1]
            units[left_index] = [units[left_index][0], resume, node]
            drop = candidate_indices | {right_index}
            return (
                [unit for index, unit in enumerate(units) if index not in drop],
                resume,
            )

        if action == "expand_right" or (action == "merge" and left_chunk):
            label = rule.label or left_chunk.label()
            node = Tree(label, list(left_chunk) + candidate)
            units[left_index] = [units[left_index][0], end, node]
            return (
                [
                    unit
                    for index, unit in enumerate(units)
                    if index not in candidate_indices
                ],
                end,
            )

        if action == "expand_left" or (action == "merge" and right_chunk):
            label = rule.label or right_chunk.label()
            node = Tree(label, candidate + list(right_chunk))
            resume = units[right_index][1]
            units[right_index] = [start, units[right_index][1], node]
            return (
                [
                    unit
                    for index, unit in enumerate(units)
                    if index not in candidate_indices
                ],
                resume,
            )

        node = Tree(label, candidate)
        units = [
            unit for index, unit in enumerate(units) if index not in candidate_indices
        ]
        units.append([start, end, node])
        units.sort(key=lambda unit: unit[0])
        return units, end

    def _apply_rule(self, tree, stage, rule):
        units = self._units(tree)
        tokens = tree.leaves()
        search_from = 0
        while True:
            labels, spans = self._labels_and_spans(units, len(tokens))
            match = self._find_match(
                rule, units, tokens, labels, spans, start_at=search_from
            )
            if match is None:
                break
            units, search_from = self._apply_match(rule, stage, units, match)
        return Tree(tree.label(), [unit[2] for unit in units])

    def __repr__(self):
        names = [
            stage.name if isinstance(stage, FeatureChunkStage) else repr(stage)
            for stage in self._stages
        ]
        return (
            f"<CascadedChunker with {len(self._stages)} stages: "
            + ", ".join(names)
            + ">"
        )


FeatureChunker = CascadedChunker
FeatureChunkParser = CascadedChunker
CascadeChunker = CascadedChunker
FeatureRule = FeatureChunkRule


def evaluate_chunkers(chunkers, gold, chunk_label=".*"):
    """
    Compare chunkers on an annotated corpus and return precision/recall/F1.

    :param chunkers: Mapping of report name to a ``ChunkParserI`` instance.
    :param gold: Sequence of gold chunk trees.
    :param chunk_label: Regex selecting which chunk labels to score.
    :return: ``{name: {"precision": ..., "recall": ..., "f1": ...}}``.
    """
    results = {}
    for name, chunker in chunkers.items():
        score = ChunkScore(chunk_label=chunk_label)
        for correct in gold:
            score.score(correct, chunker.parse(correct.leaves()))
        results[name] = {
            "precision": score.precision(),
            "recall": score.recall(),
            "f1": score.f_measure(),
        }
    return results
