# Seyd-Yaar lean mode

## Default run
```powershell
cd "C:\Users\MorBit\Documents\GitHub\seyed-yaar\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
$env:PYTHONPATH=(Resolve-Path .\backend).Path
python -m seydyaar run-daily --out docs/latest
```

## What the default optimized run does
- species: skipjack only
- past-days: 1
- future-days: 5
- step-hours: 12
- grid: 160x160
- no heavy diagnostic exports
- no extra online wind fetch

## Useful toggles
Turn on 7-day front persistence:
```powershell
$env:SEYDYAAR_ENABLE_FRONT_PERSIST_7D="1"
```

Turn on NPP anomaly:
```powershell
$env:SEYDYAAR_ENABLE_NPP_ANOM="1"
```

Write extended layers:
```powershell
$env:SEYDYAAR_WRITE_EXTENDED_LAYERS="1"
```

Write heavy diagnostics too:
```powershell
$env:SEYDYAAR_WRITE_DIAGNOSTICS="1"
```
