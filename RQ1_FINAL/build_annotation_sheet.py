import pandas as pd, os

key = pd.read_csv('annotation_kit/annotation-KEY-do-not-share-with-annotators.csv')
pairs = pd.read_csv('data/resume_job_pairs_v3.csv')

# Build a lookup by (resume_id, job_id)
pair_lookup = pairs.set_index(['resume_id', 'job_id'])[['resume_text', 'job_text']]

rows = []
missing = []
for _, kr in key.iterrows():
    rid, jid = int(kr['resume_id']), int(kr['job_id'])
    if (rid, jid) in pair_lookup.index:
        rt = pair_lookup.loc[(rid, jid), 'resume_text']
        jt = pair_lookup.loc[(rid, jid), 'job_text']
        # If there are multiple rows with the same (resume_id, job_id),
        # take the first one.
        if isinstance(rt, pd.Series):
            rt = rt.iloc[0]
            jt = jt.iloc[0]
    else:
        rt = 'LOOKUP_FAILED'
        jt = 'LOOKUP_FAILED'
        missing.append((rid, jid))
    rows.append({
        'item_id': kr['item_id'],
        'resume_text': rt,
        'job_text': jt,
        'annotator1_score_0_1_2': '',
        'annotator1_notes': '',
        'annotator2_score_0_1_2': '',
        'annotator2_notes': '',
    })

sheet = pd.DataFrame(rows)
sheet.to_csv('annotation_kit/annotation-sheet.csv', index=False)
print(f"Written {len(sheet)} rows to annotation_kit/annotation-sheet.csv")
if missing:
    print(f"WARNING: {len(missing)} (resume_id, job_id) pairs not found in resume_job_pairs_v3.csv: {missing[:5]}")
else:
    print("All pairs resolved successfully.")
