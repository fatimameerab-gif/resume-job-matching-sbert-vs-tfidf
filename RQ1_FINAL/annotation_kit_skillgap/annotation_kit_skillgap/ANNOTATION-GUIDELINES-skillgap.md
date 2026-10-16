# Skill-gap annotation guidelines
Task: for each row you see a RESUME (look it up by resume_id in gap_resumes.csv), a TARGET OCCUPATION and one ESSENTIAL SKILL of that occupation (ESCO). Decide whether the resume shows that the person HAS the skill.
Fill the column `present` with exactly one of:
- yes    : the resume names the skill or a clear synonym/abbreviation, or describes work that obviously requires it (e.g. 'built regression models in scikit-learn' for 'apply statistical analysis techniques').
- partly : the resume shows a related or lower-level version of the skill, or only weak indirect evidence.
- no     : no evidence in the resume. (Absence of evidence is 'no', even if the person might have the skill.)
Rules: judge ONLY from the resume text; do not look at system scores; do not talk to the other annotator until both sheets are finished; use `evidence` for a short quote (optional).
ESCO essential skills are often generic (e.g. 'comply with legal regulations'); label them the same way - 'no' if the resume gives no evidence.
Resume texts are lower-cased and cleaned; ignore that. Time: about 20-30 seconds per row.
