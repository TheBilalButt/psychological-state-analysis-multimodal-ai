# DAIC-WOZ Depressed Audio Dataset (`daicwoz_audio`)

Contains all **191 files** (188 `.wav` audio files + 3 `.csv` label files), packaged into **55 standard `.zip` archives** (each under 95 MB to comply with GitHub's 100 MB per-file limit).

## How to Extract

### Windows (PowerShell)
```powershell
Get-ChildItem audio-files-depressed\*.zip | ForEach-Object { Expand-Archive -Path $_.FullName -DestinationPath . -Force }
```

### Linux / macOS / Git Bash
```bash
for z in audio-files-depressed/*.zip; do unzip -o "$z" -d .; done
```

This recreates the complete `daicwoz_audio/audio/` and `daicwoz_audio/labels/` directory structure.
