# P10-AR-Privacy

## Dataset utilities
### Count annotations for every subfolder
Used for the paper when we display number of annotations in the dataset table.
```
total=0
while IFS= read -r -d '' f; do
n=$(awk 'END{print NR}' "$f")
printf '%s %s\n' "$n" "$f"
total=$((total + n))
done < <(find . -type f -print0)
printf 'TOTAL %s\n' "$total"
```

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