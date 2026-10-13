#!/usr/bin/env bash
# SO HAI HƯỚNG trên cùng bộ đề, cùng seed (cùng bản nháp), MỘT lệnh:
#   bash scripts/ab_lonly.sh bench/icons1.json
# Lượt 1: hướng cũ (ô vỏ S + ô chi tiết I, TEXTFIX_LONLY=off). Lượt 2: chỉ dòng chữ + LaMa (TEXTFIX_LONLY=on), --fresh.
# Báo cáo lượt 2 tự so từng poster với lượt 1 (tờ ảnh ghép  nháp | hướng cũ | chỉ dòng chữ) và gói MỘT zip.
set -e
cd "$(dirname "$0")/.."
SET="${1:-bench/icons1.json}"
PORT="${TEXTFIX_PORT:-8088}"
export TEXTFIX_GEO=on

wait_ready() {
  for _ in $(seq 1 180); do
    if curl -s "localhost:${PORT}/api/health" | grep -q '"ready": *true'; then return 0; fi
    sleep 5
  done
  echo "máy chủ không sẵn sàng sau 15 phút -- xem server.log"; exit 1
}

echo "== LƯỢT 1 / 2: hướng cũ (TEXTFIX_LONLY=off) =="
TEXTFIX_LONLY=off bash scripts/start_server.sh
wait_ready
python scripts/dev_batch.py "$SET" --fresh --no-ship

echo "== LƯỢT 2 / 2: chỉ dòng chữ (TEXTFIX_LONLY=on) =="
TEXTFIX_LONLY=on bash scripts/start_server.sh
wait_ready
python scripts/dev_batch.py "$SET" --fresh
echo "XONG. Máy chủ đang chạy ở chế độ chỉ dòng chữ."
