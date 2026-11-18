-----------------------------
Peptide Fragment Mass Matcher
-----------------------------

Run on Windows by double-clicking Run_GUI.bat, or from a terminal:
  py -3 peptide_fragment_matcher_gui.py

Requirements: Python 3, tkinter, numpy, pandas.
Install numerical dependencies if needed:
  py -3 -m pip install numpy pandas

Enter an unmodified or modified peptide sequence, positive/negative ion mode,
precursor charge state (the GUI searches fragment charge states from 1 up to that value), maximum neutral-loss events, observed fragment m/z,
and tolerance in ppm. Results list every generated terminal, precursor, and
internal y-type fragment within the tolerance, ordered by absolute ppm error.
The matcher uses the residue-limited neutral-loss rules and ion mass formulas
from peptide_fragment_matcher_residue_limited.py in this folder.

Modification syntax follows the matcher, for example M[Oxidation],
[Acetyl]-PEPTIDE, or PEPTIDE-[Amidated].

