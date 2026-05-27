import json
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
CL_DIR = BASE_DIR / "data" / "cl_requests"
CL_DIR.mkdir(parents=True, exist_ok=True)


def _req_path(request_id: str) -> Path:
    return CL_DIR / f"{request_id}.json"


def list_requests(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return list of continual-learning requests stored as json files under app/data/cl_requests.
    If status is provided, filter by req['status'].
    """
    out: List[Dict[str, Any]] = []
    for p in CL_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if status is None or str(obj.get("status", "")).upper() == str(status).upper():
                out.append(obj)
        except Exception:
            continue

    # sort newest first if created_at exists, else by filename
    def key_fn(x: Dict[str, Any]):
        return x.get("created_at", "")

    out.sort(key=key_fn, reverse=True)
    return out


def update_request(request_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Patch a request json and save.
    If file does not exist, create it with minimal fields.
    """
    p = _req_path(request_id)

    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
    else:
        obj = {"request_id": request_id}

    # apply patch
    for k, v in patch.items():
        obj[k] = v

    # ensure status exists
    if "status" not in obj:
        obj["status"] = "PENDING"

    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return obj