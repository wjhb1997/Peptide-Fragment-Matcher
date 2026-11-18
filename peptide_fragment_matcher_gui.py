Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

from peptide_fragment_matcher_residue_limited import generate_fragments

APP_DIR = Path(__file__).resolve().parent

class MatcherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Peptide Fragment Mass Matcher")
        self.geometry("1050x680")
        self.minsize(780, 500)
        self.matches = []
        self._build()

    def _build(self):
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Peptide Fragment Mass Matcher", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Find theoretical peptide fragments that fall within the observed m/z tolerance.").pack(anchor="w", pady=(2, 12))

        form = ttk.LabelFrame(outer, text="Input", padding=10)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Peptide sequence").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=5)
        self.sequence = ttk.Entry(form)
        self.sequence.grid(row=0, column=1, sticky="ew", pady=5)
        self.sequence.insert(0, "PEPTIDE")
        ttk.Label(form, text="Supports residue modifications like M[Oxidation], [Acetyl]-PEPTIDE, and PEPTIDE-[Amidated].").grid(row=1, column=1, sticky="w")

        row = ttk.Frame(form)
        row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for i in range(8): row.columnconfigure(i, weight=1)
        ttk.Label(row, text="Ion mode").grid(row=0, column=0, sticky="w")
        self.mode = tk.StringVar(value="positive")
        ttk.Combobox(row, textvariable=self.mode, values=("positive", "negative"), state="readonly", width=12).grid(row=1, column=0, sticky="ew", padx=(0, 10))
        ttk.Label(row, text="Precursor charge").grid(row=0, column=1, sticky="w")
        self.charge = ttk.Entry(row, width=10)
        self.charge.insert(0, "1")
        self.charge.grid(row=1, column=1, sticky="ew", padx=(0, 10))
        ttk.Label(row, text="Max losses").grid(row=0, column=2, sticky="w")
        self.losses = ttk.Entry(row, width=10)
        self.losses.insert(0, "2")
        self.losses.grid(row=1, column=2, sticky="ew", padx=(0, 10))
        ttk.Label(row, text="Fragment m/z").grid(row=0, column=3, sticky="w")
        self.mass = ttk.Entry(row, width=14)
        self.mass.grid(row=1, column=3, sticky="ew", padx=(0, 10))
        ttk.Label(row, text="Tolerance (ppm)").grid(row=0, column=4, sticky="w")
        self.ppm = ttk.Entry(row, width=12)
        self.ppm.insert(0, "10")
        self.ppm.grid(row=1, column=4, sticky="ew", padx=(0, 10))
        self.run_button = ttk.Button(row, text="Find matches", command=self.find_matches)
        self.run_button.grid(row=1, column=5, sticky="ew")
        self.mass.bind("<Return>", lambda _e: self.find_matches())
        self.ppm.bind("<Return>", lambda _e: self.find_matches())

        tools = ttk.Frame(outer)
        tools.pack(fill="x", pady=(10, 5))
        self.status = ttk.Label(tools, text="Enter values and select Find matches.")
        self.status.pack(side="left")
        ttk.Button(tools, text="Export results.", command=self.export).pack(side="right")

        table_frame = ttk.Frame(outer)
        table_frame.pack(fill="both", expand=True)
        cols = ("label", "theoretical", "observed", "ppm", "charge", "sequence", "loss", "direction")
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings")
        headings = {"label":"Ion", "theoretical":"Theoretical m/z", "observed":"Input m/z", "ppm":"Error (ppm)", "charge":"Charge", "sequence":"Fragment sequence", "loss":"Neutral loss", "direction":"Fragment type"}
        widths = {"label":130,"theoretical":115,"observed":105,"ppm":95,"charge":65,"sequence":160,"loss":120,"direction":105}
        for c in cols:
            self.table.heading(c, text=headings[c])
            self.table.column(c, width=widths[c], anchor="w" if c in ("label","sequence","loss","direction") else "e")
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        xscroll = ttk.Scrollbar(outer, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        table_frame.rowconfigure(0, weight=1); table_frame.columnconfigure(0, weight=1)
        xscroll.pack(fill="x")
        self.sequence.focus_set()

    def find_matches(self):
        try:
            sequence = self.sequence.get().strip()
            mode = self.mode.get()
            charge = int(self.charge.get().strip())
            max_losses = int(self.losses.get().strip())
            observed = float(self.mass.get().strip())
            tolerance = float(self.ppm.get().strip())
            if not sequence: raise ValueError("Enter a peptide sequence.")
            if charge < 1: raise ValueError("Precursor charge must be a positive integer.")
            if max_losses < 0: raise ValueError("Maximum losses must be zero or greater.")
            if observed <= 0: raise ValueError("Fragment m/z must be greater than zero.")
            if tolerance < 0: raise ValueError("Tolerance must be zero or greater.")
            charge_sign = 1 if mode == "positive" else -1
            fragment_charges = [charge_sign * z for z in range(1, charge + 1)]
            theoretical = generate_fragments(sequence, ["a","b","c","d","x","y","z"], fragment_charges, max_losses=max_losses, ion_mode=mode, include_precursor_losses=True, include_internal=True)
            found = []
            if not theoretical.empty:
                for fragment in theoretical.to_dict("records"):
                    theo = float(fragment["mz"])
                    err = (observed - theo) / theo * 1_000_000
                    if abs(err) <= tolerance:
                        found.append({"label": fragment["label"], "theoretical": theo, "observed": observed, "ppm": err, "charge": f"{abs(int(fragment['charge']))}{'+' if int(fragment['charge']) > 0 else '-'}", "sequence": fragment["fragment_seq"], "loss": fragment["loss"], "direction": fragment["direction"]})
            self.matches = sorted(found, key=lambda x: (abs(x["ppm"]), x["label"]))
            self.table.delete(*self.table.get_children())
            for m in self.matches:
                self.table.insert("", "end", values=(m["label"], f"{m['theoretical']:.6f}", f"{m['observed']:.6f}", f"{m['ppm']:+.2f}", m["charge"], m["sequence"], m["loss"], m["direction"]))
            self.status.config(text=f"{len(self.matches)} match(es) within �{tolerance:g} ppm; sorted by absolute ppm error.")
        except Exception as exc:
            messagebox.showerror("Could not calculate matches", str(exc), parent=self)

    def export(self):
        if not self.matches:
            messagebox.showinfo("No results", "There are no matches to export.", parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, title="Export fragment matches", initialdir=APP_DIR, initialfile="fragment_matches.csv", defaultextension=".csv", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not path: return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.matches[0].keys()))
            writer.writeheader(); writer.writerows(self.matches)
        self.status.config(text=f"Exported {len(self.matches)} match(es) to {path}")

if __name__ == "__main__":
    MatcherApp().mainloop()




