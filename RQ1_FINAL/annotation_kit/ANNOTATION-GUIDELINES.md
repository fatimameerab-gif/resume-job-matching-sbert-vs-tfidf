# Annotation guidelines (draft v0.1 — revise after a 20-item pilot)

Task: for each row in `annotation_sheet.csv`, read the resume and the job text and give ONE score.

| Score | Meaning |
|---|---|
| 0 | No fit. Different occupation or the core required skills are absent. |
| 1 | Partial fit. Related field or overlapping skills, but a clear gap in core requirements or seniority. |
| 2 | Good fit. Core required skills/experience are present; you would shortlist this candidate. |

Rules
1. Judge only the text shown. Do not use the category name, the source, or any outside knowledge of where the text came from (the key file is hidden from you on purpose).
2. Score skill/experience fit, not writing quality, name, gender, location or nationality.
3. Job texts may be short taxonomy (ESCO-style) definitions. Score against what the text actually requires.
4. Resume text is lower-cased and pre-processed; do not penalise that.
5. Work independently. Do NOT discuss items with the other annotator until both have finished.
6. Use `annotatorN_notes` for anything ambiguous (e.g. "job text is generic").

Process
- Pilot: both annotators do items A001–A020, run `compute_agreement.py`, discuss, refine the rules above, then annotate the rest.
- Then run `python compute_agreement.py annotation_sheet.csv`. Report kappa (weighted and binary) in the paper. Adjudicate `disagreements_to_adjudicate.csv` and record the adjudicated label in a NEW column; never overwrite the raw annotator columns.
- Suggested annotators: two people with HR/technical-hiring or domain knowledge (not the paper's author alone). Record who they are and their expertise for the methods section.

Limitations to state in the paper
- This sample (240 pairs) reuses the existing random-negative pairs, so it measures label validity of the current weak labels, not hard-negative difficulty. Hard-negative tiers must be mined by the pipeline first, then a second batch annotated.
- 240 pairs is enough to estimate agreement and audit weak-label noise, but not enough for a stand-alone benchmark (the reviewer suggests >= 500 pairs and kappa >= 0.6).
