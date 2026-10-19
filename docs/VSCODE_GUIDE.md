# Running the prototype in VS Code

## One-time setup
1. Install VS Code (https://code.visualstudio.com) if you do not have it.
2. VS Code -> **File -> Open Folder...** -> choose the repository folder.
   Click "Yes, I trust the authors".
3. When VS Code offers to install the recommended extensions (Python, Pylance), click **Install**.
   (Or open Extensions with Ctrl+Shift+X and install "Python" by Microsoft.)
4. Check the interpreter: bottom-right corner should say the `.venv` Python (3.14).
   If not: Ctrl+Shift+P -> "Python: Select Interpreter" -> choose
   `.venv\Scripts\python.exe (created by setup_env.bat)`.

## Run the demo
- Press **F5** (Run and Debug panel -> "Run demo (press F5)"). Output appears in the terminal at the bottom.
- Or: Ctrl+Shift+P -> "Tasks: Run Task" -> **Run demo**, then **Open demo report** and **Open risk map**.
- Or from the VS Code terminal (Ctrl+`):

      cd code
      python run_demo.py --data ..\data\real_dataset_matched.npz --model ..\results\proposed_model.pt --out ..\demo_output

Then open `demo_output\demo_report.html` (right-click -> "Reveal in File Explorer", or run the task "Open demo report").

## Running train.py
`train.py` now works on its own: it uses `data\real_dataset_matched.npz` (the dataset the model was trained on) with the
pooled, location-grouped split, and writes every run to its own new folder `train_output\run_<date_time>\` (never to `results\`, so your final results stay safe).
**How to read the results of a train.py run:** when it finishes, the terminal prints a boxed summary and the folder
path. In that folder open **SUMMARY.txt** (one page: proposed-model scores + all models ranked). The full table is
`comparison.csv`; graphs are the `.png` files. `train_output\runs_log.csv` has one line per run so you can compare runs.

- F5 -> **Train (quick test, 2 epochs)**: a few minutes on CPU, numbers will be lower than final.
- F5 -> **Train (full, long)**: 30 epochs for all 10 models; can take hours on CPU. Not needed for the demo.
- For the exact reported tables/figures use `paper_results.py` (3 seeds, writes `paper_results\`).

## What to show in the viva
- `code\train.py`: model (`ImageBranch`, `SeqBranch`, `Net`), training, `load_and_split`.
- `code\run_demo.py`: the demo.
- `demo_output\demo_report.html` and `risk_map.html`.
- `docs\CODE_GUIDE.md` for answers to likely questions.

## If something fails
Copy the red error text from the terminal (or screenshot it) for troubleshooting.

## Using the GPU (NVIDIA only)
`train.py` already picks the GPU automatically (`cuda` if PyTorch can see one, otherwise CPU) and prints `device: cuda`
or `device: cpu` at the start. So the only question is whether your PyTorch can see the GPU.

1. Run `python code\check_gpu.py` in the VS Code terminal.
2. If it says `GPU available : True`, just run train.py; nothing else to change.
3. If it says `Built with CUDA : None`, your PyTorch is the CPU-only build. Install the CUDA build into the same `.venv`:
   - Open https://pytorch.org/get-started/locally/ , choose Stable / Windows / Pip / Python / a CUDA version,
     and copy the `pip install ...` command it shows.
   - Run it in the VS Code terminal (with the `.venv` active), then run `check_gpu.py` again.
4. No NVIDIA GPU (Intel/AMD graphics only)? Training stays on the CPU. For GPU training without one, use Google Colab.
Your NVIDIA driver must be recent; check with `nvidia-smi` in a terminal.
