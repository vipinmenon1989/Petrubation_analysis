"""Barcode -> guide handling: parsing, assignment, and table generation.

The barcode table (``BARCODE_10x_Merged.txt``) lists **one row per detected
guide per cell**, so a cell carrying several guides appears several times. Three
things need care when collapsing that to one label per cell, and each is handled
here:

1. **Target parsing.** The ``gene`` column of the shipped table splits the guide
   name on ``_`` only, so guides whose names carry R's ``make.unique`` suffix
   instead (``CD81.2``) keep it and become separate targets. ``CD81`` ends up
   split into ``CD81.1`` and ``CD81.2``, each analysed with half its cells while
   the other half sits in its own control group.

2. **Multiplets.** ``set_index('cell')['gene'].to_dict()`` keeps whichever row
   came last, which assigns a guide at random for every multiplet. Here a cell is
   assigned only when its top guide is clearly dominant, and is otherwise
   labelled ``Ambiguous`` or ``Unassigned``.

3. **Library prefixes.** Stripping ``S1L1_`` makes barcodes from different lanes
   collide — the same 10x barcode occurs in every lane. Collisions are detected
   and reported rather than silently resolved.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Guides matching this are the non-targeting control population.
NTC_PATTERN = re.compile(r"^non[-_.]?targeting|^ntc|^non$", re.IGNORECASE)

NEG_CTRL_LABEL = "Non-Targeting"
AMBIGUOUS_LABEL = "Ambiguous"
UNASSIGNED_LABEL = "Unassigned"


def parse_target_gene(sgrna: str) -> str:
    """Return the target gene for a guide name.

    Splits on ``_``, ``-`` or ``.``, so both library conventions resolve to the
    same target::

        AFF4_P1P2_1  -> AFF4      (10x features file)
        AFF4-P1P2.2  -> AFF4      (Seurat genotype column)
        CD81.2       -> CD81      (R make.unique suffix)
        non_targeting.10 -> Non-Targeting

    Splitting on ``_`` alone leaves ``CD81.2`` unparsed, which silently splits
    one target into several.
    """
    name = str(sgrna)
    if NTC_PATTERN.match(name):
        return NEG_CTRL_LABEL
    return re.split(r"[_\-.]", name)[0]


def assign_guides(
    bc_frame: pd.DataFrame,
    min_umi: int = 3,
    dominance_ratio: float = 2.0,
    cell_col: str = "cell",
    gene_col: str = "gene",
    count_col: str = "umi_count",
) -> pd.Series:
    """Collapse a barcode table to one label per cell.

    A cell is assigned to its top guide only when that guide has at least
    ``min_umi`` counts **and** exceeds the runner-up by ``dominance_ratio``.
    Otherwise it is ``Ambiguous``; with nothing above ``min_umi`` it is
    ``Unassigned``. Both should be excluded from the target and the control
    group rather than folded into either.

    Returns a Series indexed by cell id.
    """
    if count_col not in bc_frame.columns:
        raise ValueError(
            f"assign_guides needs a {count_col!r} column to resolve multiplets; "
            f"got {list(bc_frame.columns)}"
        )

    frame = bc_frame.sort_values([cell_col, count_col], ascending=[True, False])
    # cumcount gives the within-cell rank directly. ``groupby.nth(1)`` looks
    # equivalent but returns a Series carrying the *original* row index, so
    # reindexing it by cell id yields all-NaN and every cell then looks
    # dominant — no cell is ever called ambiguous.
    rank = frame.groupby(cell_col, sort=False).cumcount()
    top = frame[rank == 0].set_index(cell_col)
    second_umi = (
        frame[rank == 1]
        .set_index(cell_col)[count_col]
        .reindex(top.index)
        .fillna(0)
        .astype(float)
    )

    top_umi = top[count_col].astype(float)
    assigned = (top_umi >= min_umi) & (top_umi > dominance_ratio * second_umi)

    labels = top[gene_col].astype(str).where(assigned, AMBIGUOUS_LABEL)
    labels[top_umi < min_umi] = UNASSIGNED_LABEL

    counts = labels.value_counts()
    n = len(labels)
    logger.info(
        "Guide assignment: %d cells | %.1f%% assigned, %.1f%% ambiguous, "
        "%.1f%% unassigned",
        n,
        100 * (~labels.isin([AMBIGUOUS_LABEL, UNASSIGNED_LABEL])).mean(),
        100 * counts.get(AMBIGUOUS_LABEL, 0) / max(n, 1),
        100 * counts.get(UNASSIGNED_LABEL, 0) / max(n, 1),
    )
    return labels.rename("gene")


def load_barcode_table(
    path: str,
    strip_prefix: bool = True,
    min_umi: int = 3,
    dominance_ratio: float = 2.0,
    reparse_gene: bool = True,
) -> pd.Series:
    """Read a barcode table and return one guide label per cell.

    ``reparse_gene`` re-derives the target from the ``sgrna`` column rather than
    trusting the file's ``gene`` column, which fixes the ``CD81.2`` case without
    needing the file to be regenerated.

    When ``strip_prefix`` is on, barcodes that collide across libraries are
    reported: the same 10x barcode legitimately occurs in every lane, so
    stripping ``S1L1_`` merges genuinely different cells.
    """
    frame = pd.read_csv(path, sep="\t")
    frame.columns = frame.columns.str.lower()

    if reparse_gene and "sgrna" in frame.columns:
        frame["gene"] = [parse_target_gene(s) for s in frame["sgrna"]]

    if strip_prefix:
        stripped = frame["cell"].astype(str).str.rsplit("_", n=1).str[-1]
        collisions = (
            frame.assign(_s=stripped).groupby("_s")["cell"].nunique().gt(1).sum()
        )
        if collisions:
            logger.warning(
                "%d barcode(s) occur in more than one library and collide once "
                "the prefix is stripped. The same 10x barcode is reused across "
                "lanes, so these are different cells; keep the library prefix on "
                "both sides where possible.",
                collisions,
            )
        frame["cell"] = stripped

    return assign_guides(
        frame, min_umi=min_umi, dominance_ratio=dominance_ratio
    )


def write_barcode_table(
    adata_guides,
    path: str,
    min_umi: int = 3,
    lane: str | None = None,
) -> pd.DataFrame:
    """Generate a barcode table from a guide count matrix.

    The shipped ``BARCODE_10x_Merged.txt`` is exactly the guide count matrix in
    long form keeping entries of at least 3 UMIs — verified against
    ``filtered_feature_bc_matrix_S1lane1``, where that threshold reproduces the
    file's per-cell totals and guides-per-cell for 100% of cells. Generating it
    here makes the table reproducible instead of a hand-maintained side file.

    Rows are written with the **highest-count guide last** within each cell, so
    that even a naive "last row wins" reader lands on the dominant guide.
    """
    from scipy import sparse

    X = sparse.csr_matrix(adata_guides.X)
    X.data[X.data < max(int(min_umi), 1)] = 0
    X.eliminate_zeros()
    coo = X.tocoo()

    guides = adata_guides.var_names.to_numpy().astype(str)
    barcodes = adata_guides.obs_names.to_numpy().astype(str)
    cells = barcodes[coo.row]
    if lane:
        cells = np.array([f"{lane}_{b}" for b in cells])

    table = pd.DataFrame(
        {
            "cell": cells,
            "barcode": guides[coo.col],
            "sgrna": guides[coo.col],
            "gene": [parse_target_gene(g) for g in guides[coo.col]],
            "umi_count": coo.data.astype(int),
        }
    ).sort_values(["cell", "umi_count"], ascending=[True, True])

    table.to_csv(path, sep="\t", index=False)
    logger.info(
        "Wrote %s: %d rows for %d cells (guides with >= %d UMIs)",
        path,
        len(table),
        table["cell"].nunique(),
        min_umi,
    )
    return table
