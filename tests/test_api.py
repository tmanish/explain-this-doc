from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

LEASE_SNIPPET = (
    "RESIDENTIAL LEASE AGREEMENT between Landlord and Tenant. Monthly rent "
    "of $1,850.00 due on the first. Security deposit of $2,775.00. This "
    "lease will automatically renew. Late fee of $75.00 applies."
)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_text():
    r = client.post("/api/analyze/text", json={"text": LEASE_SNIPPET})
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["classification"]["document_type"] == "lease"
    assert body["document_id"] is None  # not stored by default
    assert body["result"]["disclaimer"]


def test_analyze_empty_text_rejected():
    r = client.post("/api/analyze/text", json={"text": "   "})
    assert r.status_code == 422


def test_redact_before_analysis():
    text = LEASE_SNIPPET + " Tenant SSN 123-45-6789."
    r = client.post("/api/analyze/text", json={"text": text, "redact_first": True})
    assert r.status_code == 200
    assert "123-45-6789" not in r.text


def test_store_and_delete_roundtrip():
    r = client.post("/api/analyze/text", json={"text": LEASE_SNIPPET, "store": True})
    doc_id = r.json()["document_id"]
    assert doc_id

    r = client.get(f"/api/documents/{doc_id}")
    assert r.status_code == 200

    r = client.get(f"/api/documents/{doc_id}/report.md")
    assert r.status_code == 200 and "## Disclaimer" in r.text

    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert client.get(f"/api/documents/{doc_id}").status_code == 404


def test_demo_endpoints():
    r = client.get("/api/demo")
    ids = {d["id"] for d in r.json()}
    assert {"lease", "medical_bill", "credit_card_notice",
            "insurance_policy", "employment_agreement"} <= ids
    r = client.post("/api/demo/lease")
    assert r.status_code == 200
    assert r.json()["result"]["classification"]["document_type"] == "lease"


def test_demo_path_traversal_blocked():
    assert client.post("/api/demo/..%2F..%2Fetc%2Fpasswd").status_code in (404, 405, 422)


def test_explain_endpoint():
    r = client.post("/api/explain", json={
        "passage": "This agreement is subject to binding arbitration.",
        "mode": "risks",
    })
    assert r.status_code == 200
    assert "rbitrat" in r.json()["explanation"]


def test_upload_txt():
    r = client.post(
        "/api/analyze/upload",
        files={"file": ("lease.txt", LEASE_SNIPPET.encode(), "text/plain")},
    )
    assert r.status_code == 200


def test_redact_endpoint():
    r = client.post("/api/redact", json={"text": "Call (704) 555-0182"})
    assert r.status_code == 200
    assert r.json()["counts"]["phone"] == 1
