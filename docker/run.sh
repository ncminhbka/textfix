#!/usr/bin/env bash
# Lệnh chính của container: in thông tin GPU, rồi chạy máy chủ ở tiền cảnh (log ra `docker logs`).
set -e
cd "$(dirname "$0")/.."
PY="${TEXTFIX_PY:-python}"
# image notebook đặt libcuda lệch driver (thư mục .../compat) lên đầu LD_LIBRARY_PATH (lỗi 803 trên máy notebook): mặc định bỏ,
# như scripts/start_server.sh. Máy driver cũ cần compat để chạy CUDA mới hơn: TEXTFIX_KEEP_COMPAT=1 giữ nguyên.
if [ "${TEXTFIX_KEEP_COMPAT:-0}" != "1" ]; then
    export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd: -)"
fi
if [ -z "${TEXTFIX_FLUX_URL:-}" ]; then
    echo "LỖI: chưa đặt TEXTFIX_FLUX_URL (image không chứa FLUX; xem docker/textfix.env.example)" >&2
    exit 1
fi
# máy lạ: in ngay torch / CUDA / driver / từng card để biết máy có hợp không (GPU không dùng được thì Hi-SAM + LaMa chạy CPU)
"$PY" - <<'EOF' || true
import subprocess, torch
print(f"[textfix] torch {torch.__version__}, build CUDA {torch.version.cuda}, GPU dùng được: {torch.cuda.is_available()}")
try:
    print("[textfix] " + subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
                                         "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
          .replace("\n", "\n[textfix] "))
except Exception as e:
    print(f"[textfix] không chạy được nvidia-smi ({e}) -- container không thấy GPU? (docker run --gpus all)")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, tot = torch.cuda.mem_get_info(i)
        warn = "  << dưới 8 GB trống: Hi-SAM + LaMa có thể hết bộ nhớ (TEXTFIX_GPUS để bỏ card này)" if free < 8 * 2 ** 30 else ""
        print(f"[textfix] cuda:{i} {torch.cuda.get_device_name(i)}: trống {free / 2 ** 30:.1f} / {tot / 2 ** 30:.1f} GB{warn}")
else:
    print("[textfix] CHÚ Ý: không dùng được GPU -> Hi-SAM + LaMa chạy CPU (chậm). Driver quá cũ cho torch này? thử TEXTFIX_KEEP_COMPAT=1")
EOF
mkdir -p "${TEXTFIX_RUNS:-/data/runs}"
exec "$PY" server/app.py --port "${TEXTFIX_PORT:-8004}" "$@"
