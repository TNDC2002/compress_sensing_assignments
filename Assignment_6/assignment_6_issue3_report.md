# Assignment 6 — Issue 3: Selfies, Gallery Extension, and Outlier Rejection

**Data:** my selfies `selfy/` (8), friend's selfies `friend_selfies/` (3).  
**My gallery ID after enrollment:** class **39** (new subject, not in original Yale B).

This report follows the full pipeline: test without me → outlier gate → add me and retrain → test me and friend again.

---

## Phase 1 — Train on Yale only (me NOT in database)

**Setup:** Dictionary built from **1209** Yale training faces (38 subjects). My selfies are **not** in the gallery.

**Question: Can SRC recognize me?**

| Result | Detail |
|--------|--------|
| **No** | Every selfie gets a forced Yale label (e.g. 38, 11, 28), never class 39 |
| Fit quality | \(r_1 \approx 0.92\)–\(0.95\), \(r_2/r_1 \approx 1\) — poor, ambiguous |

SRC always picks the nearest Yale subject; that is **not** recognition of me.

---

## Phase 2 — Outlier-rejection mechanism

**Goal:** Declare *person not in database* when sparse coding fit is poor or ambiguous.

**Calibration:** Thresholds from **200** in-gallery Yale test faces (95th-percentile):

- \(\tau_{\mathrm{res}} \approx 0.927\), \(\tau_{\mathrm{ratio}} \approx 1.003\), \(\tau_{\mathrm{total}} \approx 0.409\), \(\tau_{\mathrm{conc}} \approx 0.054\)

**Reject** if any: \(r_{(1)} > \tau_{\mathrm{res}}\), \(r_{(2)}/r_{(1)} < \tau_{\mathrm{ratio}}\), \(\|y-A\hat\alpha\|_2 > \tau_{\mathrm{total}}\), or concentration \(< \tau_{\mathrm{conc}}\).

**Demo (Yale-only gallery, extended probes = my selfies):**

| Probe | Rejected as unknown |
|-------|--------------------:|
| My selfies (8) | **8/8** |
| Friend's selfies (3) | **3/3** |

The gate correctly flags everyone who is **not** in the 38-subject Yale gallery.

---

## Phase 3 — Add me to the database and retrain

**Setup:** Append **5** of my selfies to the training dictionary as **class 39**; **3** held out for testing. Rebuild \(A\) with **1214** atoms (1209 Yale + 5 mine).

Friend's images are **never** added to training.

---

## Phase 4 — Test after enrollment

**Calibration:** Thresholds from Yale test faces + my **test** selfies (all in-gallery).

### My selfies (now in gallery)

| Split | Count | Predicted class 39 | Accepted (not rejected) |
|-------|------:|-------------------:|------------------------:|
| Test (held out) | 3 | **3/3** | **2/3** |
| Train (in dictionary) | 5 | **5/5** | **5/5** |

After enrollment, SRC **recognizes me** as class **39** with much lower residuals (\(r_1 \approx 0.81\)–\(0.86\)) than in Phase 1. One held-out test image is still rejected by the conservative gate (\(r_1\) borderline); label is still correct.

### Friend's selfies (still NOT in gallery)

| Result |
|--------|
| **3/3 rejected** as unknown |

Even when raw SRC occasionally picks class 39 for a friend image, the **outlier gate still rejects** — friend is not treated as enrolled.

---

## Pipeline summary

| Phase | Gallery | Probe | Outcome |
|-------|---------|-------|---------|
| 1 | 38 Yale only | My selfies (8) | **0/8** recognized as class 39; forced Yale labels (38, 11, 28, …) |
| 2 | 38 Yale only | My selfies (8) | **8/8** rejected as unknown |
| 2 | 38 Yale only | Friend's selfies (3) | **3/3** rejected as unknown |
| 3 | + class 39 (5 train, 3 test) | — | Dictionary rebuilt: **1214** atoms |
| 4 | Extended | My test (3 held out) | **3/3** pred. class 39; **2/3** gate accepted |
| 4 | Extended | My train (5 in dict.) | **5/5** pred. class 39; **5/5** accepted |
| 4 | Extended | Friend's selfies (3) | **3/3** rejected (one raw label 39, still rejected) |

## Observations

1. **Without enrollment**, SRC cannot recognize me; it only forces a Yale neighbor.
2. **Outlier rejection** handles unknown identities before enrollment.
3. **After adding my training selfies**, I am recognized as class 39 with clearly better residuals.
4. **Friend remains unknown** in Phase 4 — reject mechanism still works.

## Reproduce

```bash
cd Assignment_6
python assignment_6_issue3_selfies.py --calibration-max 200
```

Outputs: `issue3_pipeline_results.csv`, `issue3_pipeline_summary.csv`, preprocessed image grids.
