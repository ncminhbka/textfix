#!/usr/bin/env bash
# Xuất image ra MỘT tệp tar.gz để tải về / mang sang máy khác (sau docker/build.sh). Chạy từ gốc repo:
#   bash docker/save.sh                      # textfix:demo -> ~/work/textfix-demo.tar.gz
#   TAG=textfix:demo2 bash docker/save.sh
# Máy nhận: docker load -i textfix-demo.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${TAG:-textfix:demo}"
OUT="${OUT:-$HOME/work/$(echo "$TAG" | tr ':/' '--').tar.gz}"

docker image inspect "$TAG" >/dev/null 2>&1 || { echo "LỖI: không có image $TAG (chạy docker/build.sh trước)" >&2; exit 1; }
SIZE="$(docker image inspect -f '{{.Size}}' "$TAG")"
FREE="$(df -B1 --output=avail "$(dirname "$OUT")" | tail -1)"
[ "$FREE" -gt "$SIZE" ] || { echo "LỖI: $(dirname "$OUT") không đủ chỗ trống" >&2; exit 1; }
# pigz nén song song (nhanh hơn nhiều); không có thì gzip mức 1 (nhanh, tệp to hơn chút)
if command -v pigz >/dev/null; then GZ=(pigz -6); else GZ=(gzip -1); fi
echo "$TAG ($((SIZE / 1024 / 1024 / 1024)) GB chưa nén) -> $OUT (vài phút đến ~20 phút)"
rm -f "$OUT"
docker save "$TAG" | "${GZ[@]}" > "$OUT"
ls -lh "$OUT"
echo "Xong. Tải: cây thư mục JupyterLab -> work/$(basename "$OUT") -> chuột phải -> Download"
echo "Máy nhận: docker load -i $(basename "$OUT")"
