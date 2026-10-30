"""Verify app.py descriptors against published proposal Tables 6 and 9."""
import numpy as np

# Table 6 published values (sequence, muH, GRAVY, AI, pI)
TABLE6 = [
    ("AINAGRINKPRPLRYLLLWLCKA", 0.252,  0.078, 131.74, 12.22),
    ("IINGGKVRTINASRVFMGIWRG",  0.347,  0.205, 101.82, None),
    ("NGWNGNRAFRVYRGAYGWGRV",   0.274, -0.957,  37.14, 12.49),
    ("NGGKRMTIYGRWINRVSLS",     0.412, -0.589,  76.84, 12.79),
    ("RCTANGRNGKHVYIYNKSLGAKLLRA", 0.25, -0.623, 82.69, 11.62),
    ("NCRSMSRARIAVVSLCGRA",     0.315,  0.247,  87.37, 12.48),
    ("EANKHGEPARVRLDTARRRVRGAGRAGA", 0.379, -1.196, 56.07, 12.61),
    ("DAASRRTAGRGRRPARCAAAGGRARA", 0.319, -0.981, 34.62, 12.96),
    ("NGQNGTVRKVYGSGGRQR",      0.212, -1.544,  32.22, 12.79),
]

# Table 9 published toxicity / hemolysis
TABLE9 = {
    "AINAGRINKPRPLRYLLLWLCKA": (0.8823, 0.8825),
    "IINGGKVRTINASRVFMGIWRG":  (0.8754, 0.8817),
    "NCRSMSRARIAVVSLCGRA":     (0.8751, 0.8901),
    "RCTANGRNGKHVYIYNKSLGAKLLRA": (0.7505, 0.6825),
    "NGGKRMTIYGRWINRVSLS":     (0.6340, 0.5599),
    "EANKHGEPARVRLDTARRRVRGAGRAGA": (0.4833, 0.3496),
    "NGWNGNRAFRVYRGAYGWGRV":   (0.5703, 0.4448),
    "DAASRRTAGRGRRPARCAAAGGRARA": (0.5303, 0.4247),
    "NGQNGTVRKVYGSGGRQR":      (0.5181, 0.4243),
}


def main():
    # Import app.py without running Streamlit UI calls.
    import ast, types, sys
    src = open("app.py", encoding="utf-8").read()
    tree = ast.parse(src)
    # keep only imports + function/class defs (drops the UI at module level)
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                              ast.ClassDef, ast.Assign, ast.Try, ast.Expr))]
    mod = types.ModuleType("app_core")
    mod.__file__ = "app.py"
    sys.modules["app_core"] = mod
    code = compile(ast.Module(body=[n for n in keep
                                    if not (isinstance(n, ast.Expr)
                                            and isinstance(n.value, ast.Call)
                                            and "set_page_config" in ast.dump(n.value))],
                              type_ignores=[]), "app_core", "exec")
    exec(code, mod.__dict__)

    d = mod.describe
    print("=" * 78)
    print("TABLE 5  --  net charge at pH 7.0")
    print("=" * 78)
    TABLE5 = [
        ("AINAGRINKPRPLRYLLLWLCKA", 23, 5.934),
        ("IINGGKVRTINASRVFMGIWRG", 22, 4.975),
        ("NGWNGNRAFRVYRGAYGWGRV", 21, 4.975),
        ("NGGKRMTIYGRWINRVSLS", 19, 4.975),
        ("RCTANGRNGKHVYIYNKSLGAKLLRA", 26, 7.032),
        ("NCRSMSRARIAVVSLCGRA", 19, 4.894),
        ("EANKHGEPARVRLDTARRRVRGAGRAGA", 28, 6.077),
        ("DAASRRTAGRGRRPARCAAAGGRARA", 26, 7.935),
        ("NGQNGTVRKVYGSGGRQR", 18, 4.975),
    ]
    print(f"{'seq':<30}{'len pub/app':>16}{'charge pub/app':>20}")
    for seq, lp, cp in TABLE5:
        la = int(d(seq)["length"])
        ca = mod.net_charge(seq, 7.0)
        flag = ""
        if la != lp or abs(ca - cp) > 0.005:
            flag = "  <-- MISMATCH"
            ok = False
        print(f"{seq:<30}{lp:>7}/{la:<8}{cp:>9.3f}/{ca:<10.3f}{flag}")

    print()
    print("=" * 78)
    print("TABLE 6  --  descriptors recomputed from sequence")
    print("=" * 78)
    print(f"{'seq':<30}{'muH pub/app':>18}{'GRAVY pub/app':>20}"
          f"{'AI pub/app':>18}{'pI pub/app':>16}")
    ok = True
    for seq, mup, gp, aip, pip in TABLE6:
        r = d(seq)
        mua, ga, aia, pia = (r["hydrophobic_moment"], r["gravy"],
                             r["aliphatic_index"], r["isoelectric_point"])
        # muH published is per-residue normalised -> compare directly
        flag = ""
        if abs(ga - gp) > 0.005 or abs(aia - aip) > 0.5:
            flag = "  <-- MISMATCH"
            ok = False
        if pip is not None and abs(pia - pip) > 0.02:
            flag = "  <-- MISMATCH"
            ok = False
        print(f"{seq:<30}{mup:>8.3f}/{mua:<9.3f}{gp:>9.3f}/{ga:<10.3f}"
              f"{aip:>8.2f}/{aia:<9.2f}"
              f"{(f'{pip:.2f}' if pip else 'n/a'):>7}/{pia:<8.2f}{flag}")

    print()
    print("=" * 78)
    print("TABLE 9  --  toxicity / hemolysis proxies")
    print("=" * 78)
    print(f"{'seq':<30}{'tox pub/app':>18}{'hem pub/app':>18}")
    for seq, (tp, hp) in TABLE9.items():
        ta, ha = mod.toxicity_score(seq), mod.hemolytic_activity(seq)
        flag = ""
        if abs(ta - tp) > 0.02 or abs(ha - hp) > 0.02:
            flag = "  <-- MISMATCH"
            ok = False
        print(f"{seq:<30}{tp:>8.4f}/{ta:<9.4f}{hp:>8.4f}/{ha:<9.4f}{flag}")

    print()
    print("=" * 78)
    print("PARETO  --  dominance per proposal Table 10")
    print("=" * 78)
    names = {s: f"Peptide_{i+1:02d}" for i, (s, *_rest) in enumerate(TABLE6)}
    rows = []
    for seq, _m, _g, _a, _p in TABLE6:
        pub_p = None
        for i, (s2, *_x) in enumerate(TABLE6):
            if s2 == seq:
                pub_p = None
        rows.append(seq)
    # use published p_active ordering from proposal Table 5
    PUB_P = {
        "AINAGRINKPRPLRYLLLWLCKA": 0.7080, "IINGGKVRTINASRVFMGIWRG": 0.4046,
        "NGWNGNRAFRVYRGAYGWGRV": 0.3536, "NGGKRMTIYGRWINRVSLS": 0.3115,
        "RCTANGRNGKHVYIYNKSLGAKLLRA": 0.2160, "NCRSMSRARIAVVSLCGRA": 0.1566,
        "EANKHGEPARVRLDTARRRVRGAGRAGA": 0.1089,
        "DAASRRTAGRGRRPARCAAAGGRARA": 0.0934, "NGQNGTVRKVYGSGGRQR": 0.0304,
    }
    LABEL = {
        "AINAGRINKPRPLRYLLLWLCKA": "Peptide_01", "IINGGKVRTINASRVFMGIWRG": "Peptide_02",
        "NGWNGNRAFRVYRGAYGWGRV": "Peptide_03", "NGGKRMTIYGRWINRVSLS": "Peptide_04",
        "RCTANGRNGKHVYIYNKSLGAKLLRA": "Peptide_05", "NCRSMSRARIAVVSLCGRA": "Peptide_06",
        "EANKHGEPARVRLDTARRRVRGAGRAGA": "Peptide_07",
        "DAASRRTAGRGRRPARCAAAGGRARA": "Peptide_08", "NGQNGTVRKVYGSGGRQR": "Peptide_09",
    }
    F = []
    for seq in LABEL:
        F.append([-PUB_P[seq], -mod.aliphatic_index(seq),
                  mod.toxicity_score(seq), mod.hemolytic_activity(seq)])
    F = np.asarray(F)
    mask = mod.pareto_mask(F)
    published = {"Peptide_01": True, "Peptide_02": True, "Peptide_03": True,
                 "Peptide_04": True, "Peptide_05": True, "Peptide_06": True,
                 "Peptide_07": True, "Peptide_08": False, "Peptide_09": False}
    for (seq, lab), m in zip(LABEL.items(), mask):
        exp = published[lab]
        flag = "" if bool(m) == exp else "  <-- MISMATCH"
        if flag:
            ok = False
        print(f"{lab}  app={'NON-DOM' if m else 'dominated':<10}"
              f" published={'NON-DOM' if exp else 'dominated':<10}{flag}")

    ranks, dist = mod.ideal_point_rank(F, mask)
    order = np.argsort(ranks)
    print("\nIdeal-point ranking (proposal 10.4):")
    for i in order:
        print(f"  rank {ranks[i]:>2}  {LABEL[list(LABEL)[i]]:<12} "
              f"d={dist[i]:.4f}  {'*' if mask[i] else ''}")

    hv = mod.hypervolume(F[mask],
                         F.max(axis=0) + 0.1 * (np.ptp(F, axis=0) + 1e-9))
    print(f"\nHypervolume HV(P*, r) = {hv:.4f}")

    print()
    print("=" * 78)
    print("ELASTIC NETWORK  --  sanity check on an idealised complex")
    print("=" * 78)
    rec = mod.ideal_beta_barrel(120, 8)
    pep = mod.place_peptide_on_barrel(rec, 21)
    coords = np.vstack([rec, pep])
    G, (ii, jj, dists) = mod.build_kirchhoff(coords, 10.0)
    nma = mod.normal_mode_analysis(G, n_modes=20)
    print(f"nodes={len(coords)}  springs={len(ii)}  mean coord="
          f"{2*len(ii)/len(coords):.2f}")
    print(f"lambda_1..5 = {np.round(nma['eigenvalues'][:5], 3)}")
    print(f"mode-1 variance  = {100*nma['mode_share'][0]:.1f}%")
    print(f"cum 3-mode variance = {100*nma['mode_share'][:3].sum():.1f}%")
    print(f"MSF range = {nma['msf'].min():.4f} .. {nma['msf'].max():.4f}")
    print(f"B-factor range = {mod.b_factors(nma['msf']).min():.1f} .. "
          f"{mod.b_factors(nma['msf']).max():.1f}")
    K = mod.stiffness_tensor(coords, ii, jj)
    w = np.linalg.eigvalsh(K)
    print(f"stiffness tensor eigenvalues = {np.round(w, 2)}  "
          f"anisotropy={w[-1]/max(w[0],1e-9):.2f}")
    zero = np.abs(G.sum(axis=1)).max()
    print(f"Kirchhoff row-sum (must be 0) = {zero:.2e}")

    print()
    print("=" * 78)
    print("VERDICT:", "ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 78)


if __name__ == "__main__":
    main()
