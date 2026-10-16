# Colab GPU run: steps

1. In Google Drive create `My Drive/RQ1_GPU/` and put `RQ1_FINAL.zip` inside it (do not unzip it; the notebook does that). It contains the code and the identifier-scrubbed data copy.
2. Open `notebooks/RQ1_GPU_colab.ipynb` in Colab (upload it, or open it from Drive with "Open with Colab").
3. Runtime -> Change runtime type -> **T4 GPU**.
4. Run cells 1 to 4. Cell 4 is a self-test (about 5 minutes) and must end with `FAILED STAGES: none`. If it prints anything else, stop and send the output.
5. Run cell 5 (the main run, about 9-10 hours in total). If Colab disconnects: reopen the notebook, run cells 1-4, then run cell 5 again; finished work is skipped. Cell 5 must end with `FAILED STAGES: none`; if not, run it once more, and if a stage still fails send the output.
6. Cell 7 writes `results_for_paper.zip` to your Drive. Upload it to the chat so the CPU analyses and the paper placeholders can be updated.

Notes
- Results are copied to `My Drive/RQ1_GPU/results_pinned` after every stage and every 10 minutes.
- If a stage fails while reading a `.npz` or `.csv` (a file left incomplete by a disconnect), delete that one file from `results_pinned` on Drive and run cell 5 again.
- All encoders are fine-tuned with one explicit training loop (`FT_MODE=manual`): MSE on the cosine similarity, AdamW, learning rate 2e-5, 3 epochs, batch size 8, 10% linear warm-up, mixed precision, gradient clipping 1.0, training texts cut at 256 tokens.
