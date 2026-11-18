Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
#!/usr/bin/env python3

"""
peptide_fragment_matcher_residue_limited.py
Targeted peptide MS/MS fragment matcher supporting terminal (a, b, c, d, x, y, z),
precursor, and internal y-type fragments with residue-limited neutral losses.

Note: d-ions in this implementation are explicitly defined as (b + H2O) in mass.
"""

import argparse
import itertools
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


# ============================================================
# Fundamental monoisotopic masses
# ============================================================

PROTON = 1.007276466621

H = 1.00782503223
C = 12.00000000000
N = 14.00307400443
O = 15.99491461957
P = 30.97376199842
S = 31.9720711744


# ============================================================
# Common small-molecule masses
# ============================================================

H2O = 2 * H + O
NH3 = N + 3 * H
NH2 = N + 2 * H
CO = C + O
CO2 = C + 2 * O
H2 = 2 * H


# ============================================================
# Neutral-loss masses
# ============================================================

LOSS_MASSES = {
    "H2O": H2O,
    "NH3": NH3,
    "CO2": CO2,
    "CO": CO,
    "CH2N2": C + 2 * H + 2 * N,
    "H3PO4": 3 * H + P + 4 * O,
    "HPO3": H + P + 3 * O,
    "C2H4O": 2 * C + 4 * H + O,
    "CH2O": C + 2 * H + O,
    "C7H6O": 7 * C + 6 * H + O,
    "C4H9N": 4 * C + 9 * H + N,
    "C3H9N": 3 * C + 9 * H + N,
    "C4H9N3": 4 * C + 9 * H + 3 * N,
    "C3H9N3": 3 * C + 9 * H + 3 * N,
    "C3H6": 3 * C + 6 * H,
    "C2H4": 2 * C + 4 * H,
    "C4H8": 4 * C + 8 * H,
    "C7H8": 7 * C + 8 * H,
    "CH4": C + 4 * H,
    "C9H7N": 9 * C + 7 * H + N,
    "CH2NO": C + 2 * H + N + O,
    "CHNO": C + H + N + O,
    "C3H5NO": 3 * C + 5 * H + N + O,
    "C2H4NO": 2 * C + 4 * H + N + O,
    "C4H4N2": 4 * C + 4 * H + 2 * N,
    "C2H5S": 2 * C + 5 * H + S,
    "CH4S": C + 4 * H + S,
    "C3H6S": 3 * C + 6 * H + S,
    "C2H6S": 2 * C + 6 * H + S,
    "H2S": 2 * H + S,
    "HS": H + S,
    "CH2S": C + 2 * H + S,
    "C3H4O2": 3 * C + 4 * H + 2 * O,
    "CO2H": C + 2 * O + H,
    "C2H2O2": 2 * C + 2 * H + 2 * O,
    "C2H3O2": 2 * C + 3 * H + 2 * O,
}


# ============================================================
# Amino-acid residue masses
# ============================================================

AA_MASS = {
    "A": 71.037113805,
    "R": 156.101111050,
    "N": 114.042927470,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.058577540,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.094963050,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.079312980,
    "Y": 163.063328575,
    "V": 99.068413945,
}


# ============================================================
# Common modification mass shifts
# ============================================================

MOD_MASS = {
    "Oxidation": 15.994914620,
    "Carbamidomethyl": 57.021463735,
    "Phospho": 79.966330410,
    "Acetyl": 42.010564684,
    "Formyl": 27.994914620,
    "Methyl": 14.015650065,
    "Dimethyl": 28.031300130,
    "Trimethyl": 42.046950195,
    "Amidated": -0.984015585,
    "Deamidated": 0.984015585,
}


# ============================================================
# Residue-specific neutral-loss rules
# ============================================================

RESIDUE_SPECIFIC_LOSSES = {
    "R": ["CH2N2"],
    "A": ["CH4"],
    "S": ["CH2O"],
    "T": ["C2H4O"],
    "Y": ["C7H6O"],
    "F": ["C7H8"],
    "L": ["C2H4", "C3H6", "C4H8"],
    "P": ["C2H4", "C3H6"],
    "K": ["C4H9N"],
    "V": ["C2H4", "C3H6"],
    "I": ["C2H4", "C3H6", "C4H8"],
    "W": ["C9H7N"],
    #"N": ["CH2NO"],
    #"Q": ["C3H5NO", "C2H4NO", "CHNO"],
    "H": ["C4H4N2"],
    "M": ["CH4S", "C3H6S", "C2H6S"],
    "C": ["H2S", "CH2S"],
    #"D": ["CO2H"],
    #"E": ["C3H4O2", "C2H2O2", "C2H3O2"],
}


# ============================================================
# Data classes
# ============================================================

@dataclass
class Residue:
    aa: str
    mods: List[str]

    @property
    def mod_mass(self) -> float:
        return sum(parse_mod_mass(m) for m in self.mods)

    @property
    def mass(self) -> float:
        return AA_MASS[self.aa] + self.mod_mass


@dataclass
class Fragment:
    ion: str
    number: int
    charge: int
    fragment_seq: str
    residues: str
    loss: str
    neutral_mass: float
    mz: float
    cleavage_index: int
    direction: str
    ion_mode: str
    internal_start: Optional[int] = None
    internal_end: Optional[int] = None


@dataclass
class Match:
    observed_mz: float
    intensity: Optional[float]
    theoretical_mz: float
    ppm_error: float
    da_error: float
    ion: str
    number: int
    charge: int
    fragment_seq: str
    loss: str
    neutral_mass: float
    cleavage_index: int
    direction: str
    ion_mode: str
    internal_start: Optional[int] = None
    internal_end: Optional[int] = None


# ============================================================
# Modification and peptide parsing
# ============================================================

def parse_mod_mass(mod_text: str) -> float:
    """Convert modification text to mass shift."""
    mod_text = mod_text.strip()
    if mod_text in MOD_MASS:
        return MOD_MASS[mod_text]
    if re.match(r"^[+-]?\d+(\.\d+)?$", mod_text):
        return float(mod_text)
    raise ValueError(
        f"Unknown modification '{mod_text}'. "
        f"Add it to MOD_MASS or use numeric notation like [+15.9949]."
    )


def parse_peptide(seq: str) -> Tuple[List[Residue], float, float, List[str], List[str]]:
    """Parse a modified peptide sequence."""
    seq = seq.strip().replace(" ", "")

    nterm_mods = []
    cterm_mods = []

    # N-terminal modification, e.g. [Acetyl]-PEPTIDE
    nterm_match = re.match(r"^(\[[^\]]+\]-)+", seq)
    if nterm_match:
        block = nterm_match.group(0)
        nterm_mods = re.findall(r"\[([^\]]+)\]", block)
        seq = seq[len(block):]

    # C-terminal modification, e.g. PEPTIDE-[Amidated]
    cterm_match = re.search(r"(-\[[^\]]+\])+$", seq)
    if cterm_match:
        block = cterm_match.group(0)
        cterm_mods = re.findall(r"\[([^\]]+)\]", block)
        seq = seq[: -len(block)]

    residues = []
    i = 0
    while i < len(seq):
        aa = seq[i]
        if aa not in AA_MASS:
            raise ValueError(f"Unsupported residue '{aa}' in sequence.")

        i += 1
        mods = []
        while i < len(seq) and seq[i] == "[":
            j = seq.find("]", i)
            if j == -1:
                raise ValueError("Unclosed modification bracket.")
            mods.append(seq[i + 1:j])
            i = j + 1

        residues.append(Residue(aa=aa, mods=mods))

    nterm_mass = sum(parse_mod_mass(m) for m in nterm_mods)
    cterm_mass = sum(parse_mod_mass(m) for m in cterm_mods)

    return residues, nterm_mass, cterm_mass, nterm_mods, cterm_mods


# ============================================================
# Neutral-loss handling
# ============================================================

def fragment_loss_options(fragment_residues: List[Residue]) -> List[str]:
    """Decide which neutral losses should be generated for a fragment/precursor."""
    aas = {r.aa for r in fragment_residues}
    mods = {m for r in fragment_residues for m in r.mods}

    losses = ["H2O", "NH3", "CO2"]

    for aa, aa_losses in RESIDUE_SPECIFIC_LOSSES.items():
        if aa in aas:
            losses.extend(aa_losses)

    if "Phospho" in mods:
        losses.extend(["H3PO4", "HPO3"])

    return list(dict.fromkeys(losses))


def internal_fragment_losses(fragment_residues: List[Residue]) -> List[Tuple[str, float]]:
    """Determine single neutral losses allowed for internal y fragments."""
    losses = [("", 0.0)]
    losses.append(("H2O", LOSS_MASSES["H2O"]))
    losses.append(("CO2", LOSS_MASSES["CO2"]))

    aas = {r.aa for r in fragment_residues}
    for aa, aa_losses in RESIDUE_SPECIFIC_LOSSES.items():
        if aa in aas:
            for loss in aa_losses:
                losses.append((loss, LOSS_MASSES[loss]))

    unique = {}
    for name, mass in losses:
        unique[name] = mass

    return list(unique.items())


def neutral_loss_group_limits(fragment_residues: List[Residue], ion_type: str = "") -> Tuple[Dict[str, str], Dict[str, int]]:
    """Define residue/modification-aware repetition limits for neutral losses."""
    aa_counts = Counter(r.aa for r in fragment_residues)
    phospho_count = sum(
        1 for r in fragment_residues for m in r.mods if m == "Phospho"
    )

    loss_group_map: Dict[str, str] = {}
    max_group_counts: Dict[str, int] = {}

    for aa, aa_losses in RESIDUE_SPECIFIC_LOSSES.items():
        count = aa_counts.get(aa, 0)
        if count <= 0:
            continue
        group_name = f"residue_{aa}"  # Shared group for all losses from this amino acid
        for loss_name in aa_losses:
            loss_group_map[loss_name] = group_name
        max_group_counts[group_name] = count

    if phospho_count > 0:
        loss_group_map["H3PO4"] = "phospho"
        loss_group_map["HPO3"] = "phospho"
        max_group_counts["phospho"] = phospho_count

    # Global neutral losses limits (H2O, CO2, NH3)
    de_count = aa_counts.get("D", 0) + aa_counts.get("E", 0)
    nq_count = aa_counts.get("N", 0) + aa_counts.get("Q", 0)

    # (1) H2O Limit
    loss_group_map["H2O"] = "global_H2O"
    if ion_type in ["x", "y", "z", "precursor"]:
        max_group_counts["global_H2O"] = de_count + 1
    else:
        max_group_counts["global_H2O"] = de_count

    # (2) NH3 Limit
    loss_group_map["NH3"] = "global_NH3"
    if ion_type in ["a", "b", "c", "d", "precursor"]:
        max_group_counts["global_NH3"] = nq_count + 1
    else:
        max_group_counts["global_NH3"] = nq_count

    # (3) CO2 Limit
    loss_group_map["CO2"] = "global_CO2"
    if ion_type in ["x", "y", "z", "precursor"]:
        max_group_counts["global_CO2"] = de_count + 1
    else:
        max_group_counts["global_CO2"] = de_count

    return loss_group_map, max_group_counts


def format_loss_combo(combo: Tuple[str, ...]) -> str:
    """Convert repeated neutral losses to readable labels."""
    counts = Counter(combo)
    parts = []
    for loss_name in sorted(counts.keys()):
        n = counts[loss_name]
        if n == 1:
            parts.append(loss_name)
        else:
            parts.append(f"{n}*{loss_name}")
    return "+".join(parts)


def loss_combinations(
    losses: List[str],
    max_losses: int,
    include_no_loss: bool = True,
    loss_group_map: Optional[Dict[str, str]] = None,
    max_group_counts: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, float]]:
    """Generate neutral-loss combinations respecting residue-specific limits."""
    output = [("", 0.0)] if include_no_loss else []
    unique_losses = list(dict.fromkeys(losses))

    if loss_group_map is None:
        loss_group_map = {}
    if max_group_counts is None:
        max_group_counts = {}

    for r in range(1, max_losses + 1):
        for combo in itertools.combinations_with_replacement(unique_losses, r):
            combo_counts = Counter(combo)
            grouped_counts = Counter()
            for loss_name, count in combo_counts.items():
                if loss_name in loss_group_map:
                    group_name = loss_group_map[loss_name]
                    grouped_counts[group_name] += count

            violates_group_limit = False
            for group_name, group_count in grouped_counts.items():
                allowed_count = max_group_counts.get(group_name, max_losses)
                if group_count > allowed_count:
                    violates_group_limit = True
                    break

            if violates_group_limit:
                continue

            label = format_loss_combo(combo)
            mass = sum(LOSS_MASSES[x] for x in combo)
            output.append((label, mass))

    return output


# ============================================================
# Charge and m/z handling
# ============================================================

def mz_from_neutral(neutral_mass: float, charge: int) -> float:
    """Calculate m/z from neutral mass and signed charge."""
    if charge == 0:
        raise ValueError("Charge cannot be zero.")
    z_abs = abs(charge)
    if charge > 0:
        return (neutral_mass + z_abs * PROTON) / z_abs
    return (neutral_mass - z_abs * PROTON) / z_abs


def parse_charges(charges_text: str, ion_mode: str) -> List[int]:
    """Parse charge states from command-line input."""
    charges = [int(x.strip()) for x in charges_text.split(",") if x.strip()]
    if any(c == 0 for c in charges):
        raise ValueError("Charge state cannot be zero.")
    if ion_mode == "positive":
        return [abs(c) for c in charges]
    if ion_mode == "negative":
        return [-abs(c) for c in charges]
    raise ValueError("ion_mode must be 'positive' or 'negative'.")


def format_charge_label(charge: int) -> str:
    if charge > 0:
        return f"{charge}+"
    return f"{abs(charge)}-"


def make_fragment_label(row):
    if row["direction"] == "internal":
        base = (
            f"{row['ion']}("
            f"{row['internal_start']}_"
            f"{row['internal_end']})"
            f"[{format_charge_label(int(row['charge']))}]"
        )
    elif row["ion"] == "precursor":
        base = f"precursor[{format_charge_label(int(row['charge']))}]"
    else:
        base = f"{row['ion']}{row['number']}[{format_charge_label(int(row['charge']))}]"

    if row["loss"] == "none":
        return base
    return base + "-" + "(" + row["loss"] + ")"


# ============================================================
# Theoretical fragment generation
# ============================================================

def generate_fragments(
    peptide: str,
    ion_types: List[str],
    charges: List[int],
    max_losses: int = 1,
    ion_mode: str = "positive",
    include_precursor_losses: bool = True,
    include_internal: bool = True,
) -> pd.DataFrame:
    """
    Generate theoretical terminal, precursor, and internal y-type fragments.
    
    Backbone formulas (neutral masses):
      a_k = prefix - CO
      b_k = prefix
      c_k = prefix + NH3
      d_k = b_k + H2O (custom user definition)
      x_k = suffix + CO
      y_k = suffix + H2O
      z_k = suffix + H2O - NH3
    """
    residues, nterm_mass, cterm_mass, _, _ = parse_peptide(peptide)
    n = len(residues)
    masses = np.array([r.mass for r in residues])
    prefix = np.concatenate(([0.0], np.cumsum(masses)))
    total_mass = float(prefix[-1])

    fragments = []

    c_mass = {}
    y_mass = {}

    for i in range(1, n):
        prefix_residues = residues[:i]
        suffix_residues = residues[i:]

        prefix_res_mass = float(prefix[i]) + nterm_mass
        suffix_res_mass = total_mass - float(prefix[i]) + cterm_mass

        c_mass[i] = prefix_res_mass + NH3
        y_mass[i] = suffix_res_mass + H2O

        # Terminal N-terminal series (a, b, c, d)
        for ion in ["a", "b", "c", "d"]:
            if ion not in ion_types:
                continue

            if ion == "a":
                base_neutral = prefix_res_mass - CO
            elif ion == "b":
                base_neutral = prefix_res_mass
            elif ion == "c":
                base_neutral = prefix_res_mass + NH3
            elif ion == "d":
                base_neutral = prefix_res_mass + H2O  # Custom d = b + H2O

            loss_opts = fragment_loss_options(prefix_residues)
            group_map, group_limits = neutral_loss_group_limits(prefix_residues, ion_type=ion)
            combos = loss_combinations(
                loss_opts, max_losses, True, group_map, group_limits
            )

            frag_seq = "".join(r.aa for r in prefix_residues)

            for loss_label, loss_m in combos:
                mz_neutral = base_neutral - loss_m
                if mz_neutral <= 0:
                    continue

                for charge in charges:
                    mz = mz_from_neutral(mz_neutral, charge)
                    fragments.append(
                        Fragment(
                            ion=ion,
                            number=i,
                            charge=charge,
                            fragment_seq=frag_seq,
                            residues=frag_seq,
                            loss=loss_label if loss_label else "none",
                            neutral_mass=mz_neutral,
                            mz=mz,
                            cleavage_index=i,
                            direction="n_term",
                            ion_mode=ion_mode,
                        )
                    )

        # Terminal C-terminal series (x, y, z)
        for ion in ["x", "y", "z"]:
            if ion not in ion_types:
                continue

            y_num = n - i
            if ion == "x":
                base_neutral = suffix_res_mass + CO2 
            elif ion == "y":
                base_neutral = suffix_res_mass + H2O
            elif ion == "z":
                base_neutral = suffix_res_mass + H2O - NH3

            loss_opts = fragment_loss_options(suffix_residues)
            group_map, group_limits = neutral_loss_group_limits(suffix_residues, ion_type=ion)
            combos = loss_combinations(
                loss_opts, max_losses, True, group_map, group_limits
            )

            frag_seq = "".join(r.aa for r in suffix_residues)

            for loss_label, loss_m in combos:
                mz_neutral = base_neutral - loss_m
                if mz_neutral <= 0:
                    continue

                for charge in charges:
                    mz = mz_from_neutral(mz_neutral, charge)
                    fragments.append(
                        Fragment(
                            ion=ion,
                            number=y_num,
                            charge=charge,
                            fragment_seq=frag_seq,
                            residues=frag_seq,
                            loss=loss_label if loss_label else "none",
                            neutral_mass=mz_neutral,
                            mz=mz,
                            cleavage_index=i,
                            direction="c_term",
                            ion_mode=ion_mode,
                        )
                    )

    # Precursor ions
    if include_precursor_losses:
        prec_neutral = total_mass + nterm_mass + cterm_mass + H2O
        loss_opts = fragment_loss_options(residues)
        group_map, group_limits = neutral_loss_group_limits(residues, ion_type="precursor")
        combos = loss_combinations(
            loss_opts, max_losses, True, group_map, group_limits
        )
        full_seq = "".join(r.aa for r in residues)

        for loss_label, loss_m in combos:
            mz_neutral = prec_neutral - loss_m
            if mz_neutral <= 0:
                continue

            for charge in charges:
                mz = mz_from_neutral(mz_neutral, charge)
                fragments.append(
                    Fragment(
                        ion="precursor",
                        number=0,
                        charge=charge,
                        fragment_seq=full_seq,
                        residues=full_seq,
                        loss=loss_label if loss_label else "none",
                        neutral_mass=mz_neutral,
                        mz=mz,
                        cleavage_index=0,
                        direction="precursor",
                        ion_mode=ion_mode,
                    )
                )

    # Internal y-type fragments
    # Formula: y(left_right) = y_mass[left] - y_mass[right]
    if include_internal and "y" in ion_types:
        for left in range(1, n - 1):
            for right in range(left + 1, n):
                frag_residues = residues[left:right]
                frag_seq = "".join(r.aa for r in frag_residues)

                neutral = y_mass[left] - y_mass[right]
                losses = internal_fragment_losses(frag_residues)

                for loss_label, loss_mass in losses:
                    mz_neutral = neutral - loss_mass
                    if mz_neutral <= 0:
                        continue

                    for charge in charges:
                        mz = mz_from_neutral(mz_neutral, charge)
                        fragments.append(
                            Fragment(
                                ion="b",
                                number=0,
                                charge=charge,
                                fragment_seq=frag_seq,
                                residues=frag_seq,
                                loss=loss_label if loss_label else "none",
                                neutral_mass=mz_neutral,
                                mz=mz,
                                cleavage_index=0,
                                direction="internal",
                                ion_mode=ion_mode,
                                internal_start=left,
                                internal_end=right,
                            )
                        )

    df = pd.DataFrame([asdict(x) for x in fragments])
    if len(df):
        df["label"] = df.apply(make_fragment_label, axis=1)
        return df.sort_values("mz").reset_index(drop=True)

    return pd.DataFrame()


# ============================================================
# Spectrum CSV handling and matching
# ============================================================

def infer_columns(
    df: pd.DataFrame,
    mz_col: Optional[str],
    intensity_col: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Infer m/z and intensity columns from a spectrum CSV."""
    cols_lower = {c.lower(): c for c in df.columns}

    if mz_col is None:
        for candidate in ["mz", "m/z", "mass", "mass_to_charge"]:
            if candidate in cols_lower:
                mz_col = cols_lower[candidate]
                break

    if mz_col is None:
        raise ValueError("Could not infer m/z column. Use --mz-col to specify it.")

    if intensity_col is None:
        for candidate in ["intensity", "i", "abundance", "relative_intensity", "height"]:
            if candidate in cols_lower:
                intensity_col = cols_lower[candidate]
                break

    return mz_col, intensity_col


def match_spectrum(
    theoretical: pd.DataFrame,
    spectrum_csv: str,
    tolerance: float,
    tolerance_unit: str = "ppm",
    mz_col: Optional[str] = None,
    intensity_col: Optional[str] = None,
) -> pd.DataFrame:
    """Match theoretical m/z values to experimental m/z peaks."""
    tolerance_unit = tolerance_unit.lower()

    if tolerance < 0:
        raise ValueError("Tolerance must be >= 0.")
    if tolerance_unit not in {"ppm", "da"}:
        raise ValueError("tolerance_unit must be 'ppm' or 'da'.")

    spec = pd.read_csv(spectrum_csv)
    mz_col, intensity_col = infer_columns(spec, mz_col, intensity_col)

    spec = spec.copy()
    spec[mz_col] = pd.to_numeric(spec[mz_col], errors="coerce")
    spec = spec.dropna(subset=[mz_col])
    spec = spec.sort_values(mz_col).reset_index(drop=True)

    exp_mz = spec[mz_col].to_numpy(dtype=float)

    if intensity_col is not None and intensity_col in spec.columns:
        exp_intensity = pd.to_numeric(spec[intensity_col], errors="coerce").to_numpy(dtype=float)
    else:
        exp_intensity = np.array([np.nan] * len(spec))

    matches = []
    if len(theoretical) == 0:
        return pd.DataFrame()

    for _, frag in theoretical.iterrows():
        target = frag["mz"]
        if tolerance_unit == "ppm":
            tol_da = target * tolerance / 1e6
        else:
            tol_da = tolerance

        left = np.searchsorted(exp_mz, target - tol_da, side="left")
        right = np.searchsorted(exp_mz, target + tol_da, side="right")

        for idx in range(left, right):
            obs = exp_mz[idx]
            intensity_value = exp_intensity[idx]

            da_error = obs - target
            ppm_error = da_error / target * 1e6

            intensity_out = None if (isinstance(intensity_value, float) and math.isnan(intensity_value)) else float(intensity_value)

            matches.append(
                Match(
                    observed_mz=obs,
                    intensity=intensity_out,
                    theoretical_mz=target,
                    ppm_error=ppm_error,
                    da_error=da_error,
                    ion=frag["ion"],
                    number=int(frag["number"]),
                    charge=int(frag["charge"]),
                    fragment_seq=frag["fragment_seq"],
                    loss=frag["loss"],
                    neutral_mass=frag["neutral_mass"],
                    cleavage_index=int(frag["cleavage_index"]),
                    direction=frag["direction"],
                    ion_mode=frag["ion_mode"],
                    internal_start=frag["internal_start"] if pd.notna(frag["internal_start"]) else None,
                    internal_end=frag["internal_end"] if pd.notna(frag["internal_end"]) else None,
                )
            )

    out = pd.DataFrame([asdict(m) for m in matches])
    if len(out) == 0:
        return out

    out["abs_ppm_error"] = out["ppm_error"].abs()
    out["abs_da_error"] = out["da_error"].abs()
    out["label"] = out.apply(make_fragment_label, axis=1)

    return out.sort_values(
        ["abs_ppm_error", "observed_mz", "theoretical_mz"]
    ).reset_index(drop=True)


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate theoretical peptide fragments (with d = b + H2O) and match them to spectrum CSV."
    )

    parser.add_argument(
        "--peptide",
        required=True,
        help="Peptide sequence. Examples: PEPTIDER, M[Oxidation]PEPTIDER",
    )

    parser.add_argument(
        "--spectrum",
        required=True,
        help="CSV file containing observed m/z values.",
    )

    parser.add_argument(
        "--out",
        default="matched_fragments.csv",
        help="Output CSV file for matched peaks.",
    )

    parser.add_argument(
        "--theoretical-out",
        default="theoretical_fragments.csv",
        help="Output CSV file for all theoretical fragments.",
    )

    parser.add_argument(
        "--mz-col",
        default=None,
        help="Name of m/z column in spectrum CSV.",
    )

    parser.add_argument(
        "--intensity-col",
        default=None,
        help="Name of intensity column in spectrum CSV.",
    )

    parser.add_argument(
        "--ions",
        default="a,b,c,d,x,y,z",
        help="Fragment types to generate. Allowed: a,b,c,d,x,y,z",
    )

    parser.add_argument(
        "--charges",
        default="1,2",
        help="Comma-separated charge states for backbone/internal fragments.",
    )

    parser.add_argument(
        "--ion-mode",
        choices=["positive", "negative"],
        default="negative",
        help="Ion mode used for fragment m/z calculation.",
    )

    parser.add_argument(
        "--max-losses",
        type=int,
        default=2,
        help="Maximum neutral-loss events per fragment/precursor.",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=10.0,
        help="Mass tolerance value.",
    )

    parser.add_argument(
        "--tol-unit",
        choices=["ppm", "da"],
        default="ppm",
        help="Mass tolerance unit.",
    )

    parser.add_argument(
        "--no-precursor-losses",
        action="store_true",
        help="Disable precursor and precursor neutral-loss generation.",
    )

    parser.add_argument(
        "--no-internal",
        action="store_true",
        help="Disable generation of internal y-type fragments.",
    )

    args = parser.parse_args()

    if args.max_losses < 0:
        raise ValueError("--max-losses must be >= 0.")
    if args.tol < 0:
        raise ValueError("--tol must be >= 0.")

    ion_types = [x.strip() for x in args.ions.split(",") if x.strip()]
    charges = parse_charges(args.charges, args.ion_mode)

    theoretical = generate_fragments(
        peptide=args.peptide,
        ion_types=ion_types,
        charges=charges,
        max_losses=args.max_losses,
        ion_mode=args.ion_mode,
        include_precursor_losses=not args.no_precursor_losses,
        include_internal=not args.no_internal,
    )

    theoretical.to_csv(args.theoretical_out, index=False)

    matches = match_spectrum(
        theoretical=theoretical,
        spectrum_csv=args.spectrum,
        tolerance=args.tol,
        tolerance_unit=args.tol_unit,
        mz_col=args.mz_col,
        intensity_col=args.intensity_col,
    )

    matches.to_csv(args.out, index=False)

    internal_count = int((theoretical["direction"] == "internal").sum()) if len(theoretical) else 0

    print(f"Peptide: {args.peptide}")
    print(f"Ion mode: {args.ion_mode}")
    print(f"Ion types: {','.join(ion_types)}")
    print(f"Charge states: {charges}")
    print(f"Tolerance: {args.tol} {args.tol_unit}")
    print(f"Maximum neutral-loss events: {args.max_losses}")
    print(f"Internal y-type fragments included: {not args.no_internal} (Count: {internal_count})")
    print(f"Theoretical fragments/ions total: {len(theoretical)}")
    print(f"Matches found: {len(matches)}")
    print(f"Wrote theoretical fragments to: {args.theoretical_out}")
    print(f"Wrote matches to: {args.out}")

    if len(matches):
        print("\nTop matches:")
        print(matches.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
