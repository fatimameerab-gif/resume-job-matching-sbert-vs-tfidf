import skill_gap_v2 as sg
from colab_common import load_encoders
ENCODERS = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2"
if __name__ == "__main__":
    for enc in load_encoders(ENCODERS, chunk_long=False):
        sg.main(out="results/skill_gap_dense", scorer=sg.DenseScorer(enc))
