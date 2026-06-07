"""Paper-level compound identity pool + deterministic SMILES backfill.

The only SMILES source that actually fires in the pipeline today is MolNexTR
reading the structure images drawn inside table cells — those SMILES are
written inline into each table's extracted CSV, and the per-source LLM copies
them into ``reactant*/product_smiles``.  That signal is LOCAL to one table, so
a structure drawn once (in a scheme or in Table 1) never fills a reference to
the same compound elsewhere.

This module MERGES every ``label/name -> SMILES`` association it can find
across the whole paper — table CSVs, the scheme ``compound_pool``, and records
that already resolved — into one paper-level lookup, then deterministically
backfills records whose SMILES the LLM left null.  No LLM, no guessing: every
SMILES is RDKit-validated and comes only from structure recognition or an
already-resolved record.
"""
from __future__ import annotations

import csv
import glob
import logging
import os
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# Compound label as it appears in a RECORD field (product_label, or a *_name
# field that actually holds a label): "2b", "S1", "4a'", "12c".  Must contain a
# digit, no spaces.  (Regex supplied by Zhao.)
LABEL_RE = re.compile(r"^[A-Za-z]{0,3}-?\d+[a-z]{0,3}('?){0,2}$")

# Stricter label for harvesting from CSV cells: must END with a letter, so pure
# integers (entry/row numbers like "1", "2") are NOT mistaken for compound
# labels and don't pollute the pool.  Precision over recall when harvesting.
_CSV_LABEL_RE = re.compile(r"^[A-Za-z]{0,3}-?\d+[a-z]{1,3}('?){0,2}$")

# Characters stripped when normalising a chemical name into an alias key.
_NAME_STRIP = " -,()'[]{}.\t\n/"

_RDKIT_SILENCED = False


def canonical_smiles(s) -> Optional[str]:
    """RDKit-canonical SMILES, or None if ``s`` is not a parseable molecule."""
    global _RDKIT_SILENCED
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        from rdkit import Chem
        if not _RDKIT_SILENCED:
            from rdkit import RDLogger
            RDLogger.DisableLog("rdApp.*")
            _RDKIT_SILENCED = True
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def normalize_name(s) -> str:
    """Alias key: lowercase + strip spacing/punctuation.

    Unifies SURFACE variants of the *same* name (case / punctuation / spacing).
    It does NOT resolve synonyms — "p-anisidine derivative" and
    "4-methoxy-N,N-diethylaniline" normalise to different keys (true synonym
    resolution needs an LLM / chemical DB, which is out of scope here).
    Example: "4-Methoxy-N,N-diethylaniline" -> "4methoxynndiethylaniline".
    """
    if not s or not isinstance(s, str):
        return ""
    s = s.lower()
    for ch in _NAME_STRIP:
        s = s.replace(ch, "")
    return s


def is_record_label(s) -> bool:
    return bool(s and isinstance(s, str) and LABEL_RE.match(s.strip()))


def _is_compound_label(s) -> bool:
    """A bare entry value counts as a compound identifier only if it carries a
    letter suffix (``2b``, ``4a'``).  Pure row indices (``1``, ``2``) are NOT
    compound labels — pooling them would collide across tables (every table has
    an entry ``1``)."""
    return bool(s and isinstance(s, str) and _CSV_LABEL_RE.match(s.strip()))


def _poolable_smiles(c) -> bool:
    """Gate for SMILES allowed INTO the pool (stricter than mere RDKit
    validity).  Rejects multi-fragment (``.``) and wildcard/attachment (``*``)
    strings — in single reactant/product cells these are almost always
    MolNexTR merge / incomplete-structure errors, and pooling them would
    propagate garbage to other records."""
    return bool(c) and "." not in c and "*" not in c


class EntityPool:
    """Paper-level ``label/name -> canonical SMILES`` lookup.

    Built incrementally from several sources; ``finalize()`` resolves
    conflicts (same key, different SMILES) by majority vote.
    """

    def __init__(self):
        self.label_to_smiles: dict[str, str] = {}
        self.name_to_smiles: dict[str, str] = {}
        self._label_votes: dict[str, Counter] = {}
        self._name_votes: dict[str, Counter] = {}

    def add_label(self, label, smiles) -> None:
        c = canonical_smiles(smiles)
        if not c or not _poolable_smiles(c) or not label:
            return
        k = str(label).strip().lower()
        if not k:
            return
        self._label_votes.setdefault(k, Counter())[c] += 1

    def add_name(self, name, smiles) -> None:
        c = canonical_smiles(smiles)
        if not c or not _poolable_smiles(c) or not name:
            return
        k = normalize_name(name)
        if len(k) < 3:  # too short to be a meaningful chemical-name key
            return
        self._name_votes.setdefault(k, Counter())[c] += 1

    def finalize(self) -> "EntityPool":
        for k, votes in self._label_votes.items():
            self.label_to_smiles[k] = votes.most_common(1)[0][0]
        for k, votes in self._name_votes.items():
            self.name_to_smiles[k] = votes.most_common(1)[0][0]
        return self

    def lookup(self, label=None, name=None) -> Optional[str]:
        """Resolve a SMILES from a label and/or a name field.

        Label is tried first (more specific).  A ``*_name`` field may itself
        hold a bare label, so a name that looks like a label is tried against
        the label table before the normalised-name table.
        """
        if label:
            hit = self.label_to_smiles.get(str(label).strip().lower())
            if hit:
                return hit
        if name and isinstance(name, str):
            if is_record_label(name):
                hit = self.label_to_smiles.get(name.strip().lower())
                if hit:
                    return hit
            hit = self.name_to_smiles.get(normalize_name(name))
            if hit:
                return hit
        return None

    @property
    def size(self) -> int:
        return len(self.label_to_smiles) + len(self.name_to_smiles)


def _harvest_csv(pool: EntityPool, csv_path: str) -> None:
    """Harvest (compound-label -> SMILES) from one extracted table CSV.

    MolNexTR writes recognised structures as inline SMILES cells.  Per data
    row we collect valid-SMILES cells and letter-bearing label cells, then
    associate each label with the SMILES cell nearest to it by column index
    (handles TATR column misalignment, where a product label sits next to its
    structure but not under the "Product" header).
    """
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
    except Exception as exc:
        logger.debug("entity_pool: cannot read %s: %s", csv_path, exc)
        return

    for row in rows:
        smiles_cells = []  # (col_idx, canonical_smiles)
        label_cells = []   # (col_idx, label_str)
        for idx, cell in enumerate(row):
            cell = (cell or "").strip()
            if not cell:
                continue
            c = canonical_smiles(cell)
            if c is not None:
                smiles_cells.append((idx, c))
            elif _CSV_LABEL_RE.match(cell):
                label_cells.append((idx, cell))
        if not smiles_cells:
            continue
        for lidx, label in label_cells:
            # Associate a label only with an ADJACENT SMILES cell (column
            # distance <= 1).  Clean tables put a compound label right next to
            # its structure; a far-away match means the columns are scrambled
            # (TATR/MolNexTR misalignment) and the association would be wrong.
            nidx, nsmiles = min(smiles_cells, key=lambda p: abs(p[0] - lidx))
            if abs(nidx - lidx) <= 1:
                pool.add_label(label, nsmiles)


def build_global_entity_pool(intermediate_dir: str, scheme_pools: dict,
                             records: list) -> EntityPool:
    """Merge every label/name -> SMILES association across one paper.

    Sources (all RDKit-validated, canonicalised):
      1. scheme compound_pool.json  — label -> SMILES (authoritative roles)
      2. every table's extracted CSV — label -> SMILES (the working signal)
      3. already-resolved records    — (label, name) -> SMILES (cross-record
         propagation: a structure drawn for one entry fills same-name entries
         elsewhere in the paper)
    """
    pool = EntityPool()

    # 1. scheme pools (reactant_pool / product_pool / compound_pool).
    if scheme_pools:
        for key in ("reactant_pool", "product_pool", "compound_pool"):
            for label, smiles in (scheme_pools.get(key) or {}).items():
                pool.add_label(label, smiles)

    # 2. table CSVs.
    tables_dir = os.path.join(intermediate_dir, "tables")
    if os.path.isdir(tables_dir):
        for csv_path in glob.glob(os.path.join(tables_dir, "**", "*_extracted.csv"),
                                  recursive=True):
            _harvest_csv(pool, csv_path)

    # 3. cross-record propagation (records that already carry a SMILES).
    for rec in records or []:
        ps = rec.get("product_smiles")
        if ps:
            pool.add_label(rec.get("product_label"), ps)
            ent = rec.get("entry_number")
            if _is_compound_label(ent):  # skip bare row indices (collide across tables)
                pool.add_label(ent, ps)
            pool.add_name(rec.get("product_name"), ps)
        for n in (1, 2):
            rs = rec.get(f"reactant{n}_smiles")
            if rs:
                pool.add_name(rec.get(f"reactant{n}_name"), rs)

    return pool.finalize()


def resolve_record_smiles(record: dict, pool: EntityPool) -> dict:
    """Backfill missing reactant/product SMILES from the paper-level pool.

    Only fills empty fields; never overwrites or guesses.  Filled SMILES are
    RDKit-canonical and a ``__smiles_source`` note records what was added.
    """
    if pool is None or pool.size == 0:
        return record

    filled = []

    if not record.get("product_smiles"):
        hit = pool.lookup(label=record.get("product_label"),
                          name=record.get("product_name"))
        if not hit and _is_compound_label(record.get("entry_number")):
            hit = pool.lookup(label=record.get("entry_number"))
        if hit:
            record["product_smiles"] = hit
            filled.append("product")

    for n in (1, 2):
        if not record.get(f"reactant{n}_smiles"):
            hit = pool.lookup(name=record.get(f"reactant{n}_name"))
            if hit:
                record[f"reactant{n}_smiles"] = hit
                filled.append(f"reactant{n}")

    if filled:
        note = record.get("__smiles_source")
        record["__smiles_source"] = (
            (note + "; " if note else "") + "entity_pool:" + ",".join(filled)
        )
    return record
