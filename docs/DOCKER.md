# Docker demo trên notebook

Image chứa code, thư viện, Chromium, Hi-SAM, LaMa (OCR nằm trong gói pip). **FLUX.2 [klein] 9B, LLM, VLM ở máy khác**, cấu
hình bằng tệp env lúc chạy. Bản demo: một container, chưa có xác thực / giới hạn lượt gọi.

## Notebook này chạy docker thế nào

Notebook là một container; lệnh `docker` trong Terminal gọi docker của **máy thật** (qua `/run/docker.sock`). Vì vậy:
- đường dẫn trong `-v` là đường dẫn **máy thật**: `/u02/jupyter-notebook/work` chính là `~/work` của notebook;
- `-p 8004:8004` mở cổng 8004 **trên máy thật**: mở giao diện / API ở `http://<IP máy chủ>:8004`;
- `--env-file`, thư mục build thì đọc từ notebook (lệnh docker gửi đi).

## 1. Build (một lần, và mỗi khi đổi code)

```bash
cd ~/work/<thư mục repo>
bash docker/build.sh          # kiểm trọng số + Chromium trong ~/persistent-data, gom ~/work/textfix-docker-build, build textfix:demo
```

Image gốc là image của chính notebook (đã có torch CUDA và thư viện Chromium cần); pip qua Nexus nội bộ. Trọng số được CHÉP
vào thư mục build (không link), kiểm đúng kích thước trước khi build.

## 2. Cấu hình

```bash
cp docker/textfix.env.example ~/work/textfix.env    # ngoài repo, KHÔNG commit
nano ~/work/textfix.env                              # điền FLUX 9B, LLM, VLM
```

## 3. Chạy

```bash
mkdir -p ~/work/textfix-data                          # = /u02/jupyter-notebook/work/textfix-data trên máy thật
docker run -d --name textfix-demo --restart unless-stopped \
  --gpus all --shm-size 1g \
  -p 8004:8004 \
  --env-file ~/work/textfix.env \
  -v /u02/jupyter-notebook/work/textfix-data:/data \
  textfix:demo

docker logs -f textfix-demo                           # đợi dòng "Sẵn sàng" (~1 phút: nạp Hi-SAM mỗi card, OCR)
curl -s localhost:8004/api/health                     # từ notebook có thể không tới được cổng máy thật: dùng http://<IP máy chủ>:8004
```

`docker logs` in ngay lúc khởi động địa chỉ FLUX / LLM / VLM phân giải ra IP nào. Báo "không phân giải được": mạng docker không
có DNS nội bộ, thêm vào `docker run` một `--add-host <tên máy>:<IP>` cho mỗi máy (IP lấy trong notebook: `getent hosts <tên máy>`),
hoặc `--dns <IP máy DNS>` (xem `cat /etc/resolv.conf` trong notebook).

Kết quả từng lượt chạy (ảnh debug, plan.json) nằm ở `~/work/textfix-data/runs/`, mở được từ JupyterLab.

Tự kiểm trong container (thư mục, thư viện, Hi-SAM / LaMa, Chromium, OCR, gọi thử LLM / VLM / FLUX):
`docker exec textfix-demo bash -c '"$TEXTFIX_PY" scripts/check_server.py'`.

## Chạy thử trên notebook khi chưa nối được FLUX 9B

`scripts/flux_gateway.py` là máy FLUX giả nói đúng khuôn API của BFL nhưng vẽ bằng FLUX 4B có sẵn trên notebook. Container
dùng chung mạng với notebook (`--network container:...`): tới máy FLUX giả qua `localhost`, gọi LLM / VLM bằng DNS của notebook.
GPU: FLUX 4B ở card 0, Hi-SAM + LaMa của container ở card 1.

```bash
pkill -f server/app.py                                   # máy chủ textfix cũ giữ FLUX trên cả hai card
cd ~/work/<thư mục repo>
nohup python scripts/flux_gateway.py --port 8010 --device cuda:0 > flux_gateway.log 2>&1 &
tail -f flux_gateway.log                                 # đợi "sẵn sàng" (~1 phút), Ctrl+C

docker run -d --name textfix-test --gpus all --shm-size 1g   --network container:$(hostname)   --env-file ~/work/textfix.env   -e TEXTFIX_FLUX_URL=http://localhost:8010/v1/flux-2-klein-4b -e TEXTFIX_FLUX_CONCURRENCY=1 -e TEXTFIX_GPUS=1   -v /u02/jupyter-notebook/work/textfix-data:/data textfix:demo
docker logs -f textfix-test                              # đợi "Sẵn sàng"
```

Mở giao diện qua proxy của JupyterLab: `https://<địa-chỉ-jupyter>/user/<tên>/proxy/8004/` (nhớ dấu `/` cuối). Không có `-p`:
container dùng mạng của notebook nên cổng 8004 nằm ngay trong notebook. Xong thì `docker rm -f textfix-test` và
`pkill -f flux_gateway.py`.

## Cập nhật, dừng

```bash
bash docker/build.sh && docker rm -f textfix-demo && docker run ...   # (lệnh run như trên)
docker stop textfix-demo ; docker start textfix-demo
docker rm -f textfix-demo                                             # xoá container (kết quả ở ~/work/textfix-data còn nguyên)
```

## Chạy trên máy khác (máy của sếp)

Image không phụ thuộc máy build. Cần ở máy chạy:
- **GPU NVIDIA ≥ 8 GB trống mỗi card** (số card tự đếm), **driver đủ mới cho bản torch trong image**, và **NVIDIA Container
  Toolkit** (`docker info | grep -i runtimes` có `nvidia`). Thiếu một trong ba: container vẫn chạy, Hi-SAM + LaMa trên CPU (chậm).
  Lúc khởi động `docker logs` in sẵn: bản torch / CUDA, driver, từng card và VRAM trống, có dùng được GPU không.
  Máy driver cũ: thử thêm `-e TEXTFIX_KEEP_COMPAT=1` (giữ thư viện CUDA compat của image).
- Gọi tới được FLUX 9B, LLM, VLM (các URL trong tệp env).

Chuyển image: qua kho nội bộ (cần quyền đẩy vào namespace đích)
```bash
docker tag textfix:demo registry.kcntt.net/<namespace>/textfix:demo
docker push registry.kcntt.net/<namespace>/textfix:demo        # máy chạy: docker pull ... cùng tên
```
hoặc tệp: `docker save textfix:demo | gzip > textfix-demo.tar.gz` (~8-10 GB), máy chạy: `gunzip -c textfix-demo.tar.gz | docker load`.

Lệnh `docker run` như mục 3, đổi thư mục mount (`-v <thư mục bất kỳ>:/data`) và tên image nếu kéo từ kho.
Tự kiểm trong container: `docker exec textfix-demo bash -c '"$TEXTFIX_PY" scripts/check_server.py'`.

## GPU

Mỗi card một luồng Hi-SAM (~4,4 GB đỉnh) + LaMa; FLUX không chạy trong container. Máy chủ textfix cũ trong notebook giữ FLUX
4B ~17 GB mỗi card: tắt nó khi chuyển sang docker (`pkill -f server/app.py`). Chỉ dùng một card: thêm `-e TEXTFIX_GPUS=0`.
