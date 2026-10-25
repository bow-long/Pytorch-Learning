# PyTorch Learning

A collection of small Python and PyTorch experiments. The repository currently
contains standalone scripts rather than one installable Python package. The
instructions below focus on the circular key-lock game in
`programs/keyLockGame copy`.

## Key-lock game

The game uses Pygame to draw a moving player and target around a circular track.
A PyTorch DQN learns when to trigger the player's action. The initial weights
are stored in [`tensor.pt`](./tensor.pt); training saves resumable progress to
`training_checkpoint.pt` and the highest-scoring model to
`best_generation.pt`.

### Run from source (Windows)

Use 64-bit Python 3.14. The dependencies below are pinned to the versions used
to prepare this project.

1. Open PowerShell in the repository folder.
2. Create and activate a virtual environment:

   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run the interpreter directly in the
   commands below instead.
3. Install the game dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements-keyLockGame.txt
   ```
4. Start training from the repository folder:

   ```powershell
   python ".\programs\train_key_lock_game.py"
   ```

Training starts from `tensor.pt` the first time and resumes from
`training_checkpoint.pt` on subsequent runs. The agent updates its DQN from
replayed game transitions, periodically syncs a target network, and writes a
new `best_generation.pt` whenever it beats its previous score. Press Esc to
stop. To play the best saved model, run:

```powershell
python ".\programs\play_best_generation.py"
```

If training has not yet saved a best model, play mode uses the bundled
`tensor.pt` weights. Both scripts look for checkpoints in the repository root.

## Build a portable Windows version

The portable build bundles Python, PyTorch, Pygame, NumPy, and the model
checkpoint into a Windows folder. The recipient does not need Python or these
packages installed. Build it on Windows for the same architecture as the
target computer; an internet connection is needed to install the build
dependencies.

1. Follow the virtual-environment and dependency-installation steps above.
2. In PowerShell, run:

   ```powershell
   .\build-portable.ps1
   ```

   If your execution policy blocks local scripts, run it for this invocation
   with:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build-portable.ps1
   ```

3. Share the complete `dist\portable\` folder or
   `dist\KeyLockGame-portable.zip`.

In the portable folder, double-click `Train-KeyLockGame.bat` to train or
`Play-Best-Generation.bat` to play the best saved model. The folder includes
both Python scripts, the DQN game code, the bundled initial checkpoint, and
the packaged Python/runtime dependencies. Training stores its checkpoints
beside `KeyLockGame.exe`, so play mode uses the best generation saved by
training. Share the entire folder (or ZIP), not only the executable.
`build\` and `dist\` outputs are not committed to Git.

## Other files

- `programs\` contains the individual experiments and game scripts.
- `Libs\` contains supporting project files.
- `tensor.pt` and `tensor copy.pt` are model checkpoints used by the
  experiments.

Each experiment may need its own dependencies and run instructions.
