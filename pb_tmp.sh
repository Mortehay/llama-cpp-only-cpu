for m in none group model; do
  echo "--- mode: $m ---"
  python3 /app/scripts/probe-gguf-offload.py --mode $m --blocks 2 2>&1 | tail -2
done
echo "=== PROBE DONE ==="
