from pathlib import Path

from nltk import RegexpParser
from nltk.chunk import (
    CascadedChunker,
    ChunkParserI,
    FeatureChunkRule,
    FeatureChunkStage,
    context_features,
    evaluate_chunkers,
)
from nltk.chunk.util import conllstr2tree
from nltk.tree import Tree

_CORPUS = Path(__file__).parent / "files" / "domain_entity_chunks.iob"

_DRUG_WORDS = {
    "agonists",
    "aspirin",
    "inhibitors",
    "metformin",
    "statins",
    "warfarin",
}

_DISEASE_WORDS = {
    "bleeding",
    "cancer",
    "dermatitis",
    "diabetes",
    "disease",
    "hypertension",
    "leukemia",
    "pain",
}


def _read_corpus():
    text = _CORPUS.read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")
    return [conllstr2tree(block, chunk_types=("DRUG", "DISEASE")) for block in blocks]


def _drug_stage():
    def drug_core(features):
        return (
            features["lower"] in _DRUG_WORDS
            or features["suffix3"] in {"mab", "nib"}
            or features["lower"].endswith("statin")
        )

    return FeatureChunkStage(
        "DRUG",
        [
            FeatureChunkRule(
                drug_core,
                "domain drug lexicon and pharmacological suffixes",
            ),
            FeatureChunkRule(
                lambda f: f["lower"] == "low" and f["right_chunk_label"] == "DRUG",
                "low dose modifier",
                continuation=lambda f: f["lower"] == "dose",
                max_length=2,
                action="expand_left",
            ),
            FeatureChunkRule(
                lambda f: f["lower"] == "ace" and f["right_chunk_label"] == "DRUG",
                "ACE inhibitor prefix",
                action="expand_left",
            ),
            FeatureChunkRule(
                lambda f: f["lower"] == "glp" and f["right_chunk_label"] == "DRUG",
                "GLP-1 agonist prefix",
                continuation=lambda f: f["lower"] in {"-", "1"},
                max_length=3,
                action="expand_left",
            ),
        ],
    )


def _disease_stage():
    disease_modifiers = {
        "chronic",
        "gastrointestinal",
        "muscle",
        "non",
        "severe",
        "type",
    }
    modifier_continuations = {
        "-",
        "2",
        "atopic",
        "cell",
        "kidney",
        "lung",
        "myeloid",
        "small",
    }
    return FeatureChunkStage(
        "DISEASE",
        [
            FeatureChunkRule(
                lambda f: f["lower"] in _DISEASE_WORDS,
                "domain disease lexicon",
            ),
            FeatureChunkRule(
                lambda f: (
                    f["lower"] in disease_modifiers
                    and f["right_chunk_label"] == "DISEASE"
                ),
                "left modifier cascade over earlier disease chunk",
                continuation=lambda f: f["lower"] in modifier_continuations,
                max_length=5,
                action="expand_left",
            ),
        ],
    )


def _feature_chunker():
    return CascadedChunker([_drug_stage(), _disease_stage()])


def _regex_baseline():
    return RegexpParser(
        r"""
        DISEASE:
          {<JJ>+<NN.*>+}
        DRUG:
          {<NN.*>+}
        """
    )


def test_context_features_include_neighbours_and_cascade_chunks():
    tokens = [
        ("Patient", "NN"),
        ("took", "VBD"),
        ("low", "JJ"),
        ("dose", "NN"),
        ("aspirin", "NN"),
    ]
    features = context_features(tokens, 4)

    assert features["word"] == "aspirin"
    assert features["prev_word"] == "dose"
    assert features["next_word"] is None
    assert features["suffix3"] == "rin"
    assert features["is_last"] is True

    labeled = context_features(
        tokens,
        4,
        chunk_labels=[None, None, "DRUG", "DRUG", "DRUG"],
    )
    assert labeled["chunk_label"] == "DRUG"
    assert labeled["chunk_start"] == 2
    assert labeled["chunk_length"] == 3
    assert labeled["chunk_text"] == "low dose aspirin"


def test_cascade_preserves_and_expands_earlier_chunks():
    tokens = [
        ("GLP", "NNP"),
        ("-", ":"),
        ("1", "CD"),
        ("agonists", "NNS"),
        ("improved", "VBD"),
        ("control", "NN"),
    ]

    tree = _feature_chunker().parse(tokens)

    assert tree == Tree(
        "S",
        [
            Tree(
                "DRUG",
                [
                    ("GLP", "NNP"),
                    ("-", ":"),
                    ("1", "CD"),
                    ("agonists", "NNS"),
                ],
            ),
            ("improved", "VBD"),
            ("control", "NN"),
        ],
    )


def test_expand_and_merge_actions_use_neighbour_chunks():
    chunks = FeatureChunkStage(
        "ORG",
        [FeatureChunkRule(lambda f: f["word"] in {"Acme", "Globex"})],
    )
    expand = FeatureChunkStage(
        "ORG",
        [
            FeatureChunkRule(
                lambda f: f["word"] == "Inc." and f["left_chunk_label"] == "ORG",
                action="expand_right",
            )
        ],
    )
    merged = FeatureChunkStage(
        "ORG",
        [
            FeatureChunkRule.from_features(
                {
                    "lower": "and",
                    "left_chunk_label": "ORG",
                    "right_chunk_label": "ORG",
                },
                action="merge",
            )
        ],
    )

    expanded = CascadedChunker([chunks, expand]).parse(
        [("Acme", "NNP"), ("Inc.", "NNP"), ("reported", "VBD")]
    )
    assert expanded[0] == Tree("ORG", [("Acme", "NNP"), ("Inc.", "NNP")])

    merged_tree = CascadedChunker([chunks, merged]).parse(
        [("Acme", "NNP"), ("and", "CC"), ("Globex", "NNP")]
    )
    assert merged_tree[0] == Tree(
        "ORG", [("Acme", "NNP"), ("and", "CC"), ("Globex", "NNP")]
    )


def test_chunker_is_pipeline_compatible():
    chunker = _feature_chunker()
    corpus = _read_corpus()

    assert isinstance(chunker, ChunkParserI)
    assert chunker.parse([]) == Tree("S", [])

    tagged_sentences = [tree.leaves() for tree in corpus]
    parsed = chunker.parse_sents(tagged_sentences)
    assert len(parsed) == len(corpus)
    assert parsed[0] == chunker.parse(tagged_sentences[0])

    # A pre-chunked tree can be fed back into a later pipeline stage.
    prefixed = Tree("S", [Tree("MENTION", tagged_sentences[0])])
    reparsed = chunker.parse(prefixed)
    assert reparsed[0].label() == "MENTION"
    assert any(isinstance(child, Tree) for child in reparsed)

    mixed = CascadedChunker(
        [
            RegexpParser("MENTION: {<NNP>}"),
            _drug_stage(),
        ]
    )
    mixed_tree = mixed.parse([("Acme", "NNP"), ("took", "VBD"), ("aspirin", "NN")])
    assert mixed_tree[0] == Tree("MENTION", [("Acme", "NNP")])
    assert mixed_tree[2] == Tree("DRUG", [("aspirin", "NN")])


def test_f1_comparison_on_annotated_domain_corpus():
    corpus = _read_corpus()
    results = evaluate_chunkers(
        {
            "tag-regex baseline": _regex_baseline(),
            "cascaded feature rules": _feature_chunker(),
        },
        corpus,
    )

    baseline = results["tag-regex baseline"]
    feature = results["cascaded feature rules"]

    assert feature["precision"] >= 0.95
    assert feature["recall"] >= 0.95
    assert feature["f1"] >= 0.95
    assert feature["f1"] > baseline["f1"]
