"""Детерминированная морфология норвежского (bokmål) — пакет по частям речи.
Без LLM, без зависимостей. Регулярные формы + детектор нерегулярности + дистракторы."""
from .noun import (regular_indef_pl, regular_def_sg, regular_def_pl, is_irregular_noun, distractors_noun,
                   noun_form_options, NOUN_FORM_CELLS)
from .verb import (predict_present, predict_weak_class, regular_past, regular_perfect,
                   all_weak_pasts, all_weak_perfects, strip_aux, is_irregular_verb, distractors_verb,
                   verb_form_options, VERB_FORM_CELLS)
from .adjective import (regular_neuter, regular_plural, regular_comparative, regular_superlative,
                        is_irregular_adj, distractors_adj, adj_form_options, ADJ_FORM_CELLS)
