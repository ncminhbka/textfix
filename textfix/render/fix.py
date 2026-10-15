"""Vòng sửa bằng code: lỗi nặng đo được -> nén sâu hơn / cắt khung chồng, dựng lại, giữ khi bớt lỗi.
"""

from __future__ import annotations

from .geometry import _ov

FIX_ROUNDS = 2   # sửa bằng code: sửa -> dựng -> kiểm, tối đa 2 lượt (lượt 2 sửa lỗi lượt 1 làm lộ ra)


def _trim(A: list, B: list) -> list | None:
    """Khung A cắt bớt MỘT phía để không còn giao B -- phía giữ được nhiều diện tích A nhất; None khi cắt mất quá nửa."""
    w, h = A[2] - A[0], A[3] - A[1]
    c = [[A[0], A[1], min(A[2], B[0]), A[3]], [max(A[0], B[2]), A[1], A[2], A[3]],
         [A[0], A[1], A[2], min(A[3], B[1])], [A[0], max(A[1], B[3]), A[2], A[3]]]
    c = [x for x in c if x[2] - x[0] > 2 and x[3] - x[1] > 2]
    if not c or w <= 0 or h <= 0:
        return None
    best = max(c, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    return best if (best[2] - best[0]) * (best[3] - best[1]) >= 0.5 * w * h else None


def autofix_units(U: list[dict], errs: list[dict], res: list[dict]) -> tuple[list[dict], list[str]]:
    """SỬA LỖI NẶNG BẰNG CODE: chữ và kiểu giữ nguyên, chỉ sửa phần đo được.
      low_contrast    -- đúng màu nháp: giữ màu + quầng; khác màu nháp: màu tương phản nhất (bảng màu poster, màu nháp đậm / nhạt dần)
      overflow        -- nén ngang sâu hơn (0.75)
      content_overlap -- phần tử thò khỏi khung nhiều hơn (hoặc khung to hơn) cắt khung một phía cho hết giao, co trong khung mới;
                         phần tử xoay / theo đường cong không cắt.
    -> (đơn vị mới, log). Không lỗi nào sửa được: y nguyên."""
    import copy
    U2 = copy.deepcopy(U)
    by = {u["id"]: u for u in U2}
    R = {r["id"]: r for r in res}
    el = {e["id"]: (u, e) for u in U2 if u["kind"] == "text" for e in u.get("els") or []}
    log = []
    fx = lambda i: by[i.split("#")[0]].setdefault("fix", {})   # noqa: E731
    for e in errs:
        t, ids = e["type"], [i for i in e.get("ops") or [] if i in el]
        if t == "low_contrast" and ids and (R.get(ids[0]) or {}).get("recolor"):
            rc = R[ids[0]]["recolor"]
            fx(ids[0]).update(color=rc["color"], halo=rc["halo"])
            log.append(f"{ids[0]}: chìm nền {e['frac']} -> màu {rc['color']}" + (" + quầng" if rc["halo"] else ""))
        elif t == "overflow" and ids and not el[ids[0]][0].get("set"):
            fx(ids[0])["condense"] = 0.75
            log.append(f"{ids[0]}: tràn {e['px']}px -> nén ngang tới 0.75")
        elif t == "content_overlap" and len(ids) == 2:
            cand = [i for i in ids if not el[i][1]["path"] and not el[i][1]["angle"]]
            if not cand:
                continue

            def spill(i):   # phần nét chữ thò ra ngoài khung của chính phần tử
                r, b = R.get(i), el[i][1]["box"]
                if not r:
                    return 0.0
                a = r["ink"]
                area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
                return 1 - _ov(a, b) * min(area, (b[2] - b[0]) * (b[3] - b[1])) / area
            area = lambda i: (el[i][1]["box"][2] - el[i][1]["box"][0]) * (el[i][1]["box"][3] - el[i][1]["box"][1])   # noqa: E731
            i = max(cand, key=lambda i: (round(spill(i), 2), area(i)))
            j = ids[1] if i == ids[0] else ids[0]
            nb = _trim(el[i][1]["box"], R[j]["ink"] if R.get(j) else el[j][1]["box"])
            if nb is None:
                log.append(f"{i}: chồng {j} nhưng cắt khung mất quá nửa -> để nguyên")
                continue
            f = fx(i)
            f.setdefault("boxes", {})[i] = [round(v, 1) for v in nb]
            f["fit"] = "shrink"
            log.append(f"{i}: chồng {j} ({e.get('frac')}) -> cắt khung còn {[round(v) for v in nb]}")
    return U2, log
