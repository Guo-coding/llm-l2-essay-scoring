#!/usr/bin/env python3
"""Dependency-based syntactic measures used in the ELLIPSE experiments."""

import spacy


nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])

# spaCy has no direct clause count, so these labels serve as a proxy
SUB_CLAUSE_DEPS = {
    "advcl",
    "ccomp",
    "xcomp",
    "csubj",
    "csubjpass",
    "acl",
    "relcl",
}

NOMINAL_MOD_DEPS = {
    "amod",
    "nmod",
    "compound",
    "appos",
    "acl",
    "relcl",
    "poss",
    "nummod",
}

MODAL_LEMMAS = {
    "can", "could", "may", "might", "must",
    "shall", "should", "will", "would",
}


def is_content_token(tok) -> bool:
    return (not tok.is_space) and (not tok.is_punct)


def get_main_clause_roots(sent_tokens: list) -> list:
    roots = []
    token_set = set(sent_tokens)

    for t in sent_tokens:
        if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX"):
            roots.append(t)
        # coordinated verbs count as another main-clause head
        elif (
            t.dep_ == "conj"
            and t.pos_ in ("VERB", "AUX")
            and t.head in token_set
            and t.head.pos_ in ("VERB", "AUX")
        ):
            roots.append(t)

    if not roots:
        roots = [sent_tokens[0].sent.root]

    return roots


def count_subordinate_clauses(sent_tokens: list) -> int:
    return sum(1 for t in sent_tokens if t.dep_.lower() in SUB_CLAUSE_DEPS)


def count_total_clauses(sent_tokens: list) -> int:
    main_roots = get_main_clause_roots(sent_tokens)
    sub = count_subordinate_clauses(sent_tokens)
    return len(main_roots) + sub


def count_coordinate_phrases(sent_tokens: list) -> int:
    count = 0
    token_set = set(sent_tokens)

    for t in sent_tokens:
        if t.dep_ != "conj":
            continue
        if t.head not in token_set:
            continue

        has_cc = any(ch.dep_ == "cc" for ch in t.head.children) or any(ch.dep_ == "cc" for ch in t.children)
        if has_cc:
            count += 1

    return count


def sentence_mdd(sent_tokens: list) -> tuple[int, int]:
    # positions are measured within the current sentence
    index_map = {tok.i: idx for idx, tok in enumerate(sent_tokens)}
    total_dist = 0
    arc_count = 0

    for t in sent_tokens:
        if t.dep_ == "ROOT":
            continue
        if t.head.i not in index_map:
            continue
        dist = abs(index_map[t.i] - index_map[t.head.i])
        total_dist += dist
        arc_count += 1

    return total_dist, arc_count


def np_length_stats(doc) -> tuple[int, int]:
    total_len = 0
    np_count = 0

    for chunk in doc.noun_chunks:
        np_count += 1
        total_len += sum(1 for tok in chunk if is_content_token(tok))

    return total_len, np_count


def nominal_modifiers_per_np(doc) -> tuple[int, int]:
    total_mods = 0
    np_count = 0

    for chunk in doc.noun_chunks:
        np_count += 1
        head = chunk.root
        chunk_tokens = set(chunk)

        # only modifiers contained in the same noun chunk are counted
        for child in head.children:
            if child not in chunk_tokens:
                continue
            if child.dep_.lower() in NOMINAL_MOD_DEPS:
                total_mods += 1

    return total_mods, np_count


def count_pp(sent_tokens: list) -> int:
    return sum(1 for t in sent_tokens if t.pos_ == "ADP" and t.dep_ in ("prep", "case"))


def count_adverbial_modifiers(sent_tokens: list) -> int:
    return sum(1 for t in sent_tokens if t.dep_ in ("advmod", "npadvmod"))


def count_passive_nominal_subjects(sent_tokens: list) -> int:
    return sum(
        1
        for t in sent_tokens
        if t.dep_ == "nsubjpass" and t.pos_ in ("NOUN", "PROPN", "PRON")
    )


def count_modal_auxiliaries(sent_tokens: list) -> int:
    c = 0
    for t in sent_tokens:
        if t.tag_ == "MD":
            c += 1
        elif t.pos_ == "AUX" and t.lemma_.lower() in MODAL_LEMMAS:
            c += 1
    return c


def cal_syntax_metrics(text: str) -> dict[str, float]:
    doc = nlp(text)

    n_sent = 0

    total_sent_words = 0
    total_clause_words = 0
    total_clauses = 0
    total_sub_clauses = 0
    total_dep_clauses = 0
    total_coord_phrases = 0
    total_mdd_dist = 0
    total_mdd_arcs = 0
    total_pp = 0
    total_advmod = 0
    total_pass_subj = 0
    total_modals = 0

    # collect sentence counts before converting them to ratios
    for sent in doc.sents:
        sent_tokens = list(sent)
        n_sent += 1

        sent_words = sum(1 for t in sent_tokens if is_content_token(t))
        total_sent_words += sent_words

        sub_c = count_subordinate_clauses(sent_tokens)
        clause_c = count_total_clauses(sent_tokens)

        total_sub_clauses += sub_c
        total_dep_clauses += sub_c
        total_clauses += clause_c

        total_clause_words += sent_words
        total_coord_phrases += count_coordinate_phrases(sent_tokens)
        dist, arcs = sentence_mdd(sent_tokens)
        total_mdd_dist += dist
        total_mdd_arcs += arcs

        total_pp += count_pp(sent_tokens)
        total_advmod += count_adverbial_modifiers(sent_tokens)
        total_pass_subj += count_passive_nominal_subjects(sent_tokens)
        total_modals += count_modal_auxiliaries(sent_tokens)

    total_np_len, np_count = np_length_stats(doc)
    total_np_mods, _ = nominal_modifiers_per_np(doc)
    np_den = max(np_count, 1)

    # calculate the measures listed in the thesis
    MLS = total_sent_words / max(n_sent, 1)
    MLC = total_clause_words / max(total_clauses, 1)

    CP_S = total_coord_phrases / max(n_sent, 1)
    CP_C = total_coord_phrases / max(total_clauses, 1)

    SC_S = total_sub_clauses / max(n_sent, 1)
    DC_C = total_dep_clauses / max(total_clauses, 1)

    MDD = total_mdd_dist / max(total_mdd_arcs, 1)

    MLNP = total_np_len / np_den
    NomMod_NP = total_np_mods / np_den

    PP_C = total_pp / max(total_clauses, 1)
    AdvMod_C = total_advmod / max(total_clauses, 1)

    PassSubj_C = total_pass_subj / max(total_clauses, 1)
    Modal_C = total_modals / max(total_clauses, 1)

    return {
        "MLS": round(MLS, 3),
        "MLC": round(MLC, 3),
        "CP/S": round(CP_S, 3),
        "CP/C": round(CP_C, 3),
        "SC/S": round(SC_S, 3),
        "DC/C": round(DC_C, 3),
        "MDD": round(MDD, 3),
        "MLNP": round(MLNP, 3),
        "Nominal_Modifiers_per_NP": round(NomMod_NP, 3),
        "PP_per_Clause": round(PP_C, 3),
        "Adverbial_Modifiers_per_Clause": round(AdvMod_C, 3),
        "Passive_Nominal_Subjects_per_Clause": round(PassSubj_C, 3),
        "Modal_Auxiliaries_per_Clause": round(Modal_C, 3),
    }
