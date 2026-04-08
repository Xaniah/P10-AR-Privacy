# P10-AR-Privacy

## Server commands
The ucloud GPU server is where we perform training

### Copy a model to your local machine
```
scp -P <PORT> ucloud@ssh.cloud.sdu.dk:/work/P10-AR-Privacy/runs/detect/<TRAIN_NO>/weights/best.pt <LOCAL_PATH>
```

**Parameters:**
- `<PORT>` — SSH port (e.g. `2699`)
- `<TRAIN_NO>` — training run folder (e.g. `train5`)
- `<LOCAL_PATH>` — local destination path (e.g. `C:\Users\Frederik\Downloads\`)