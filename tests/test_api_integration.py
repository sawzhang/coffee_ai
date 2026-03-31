import pytest
from fastapi.testclient import TestClient

from api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_status_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "feature_dim" in data
        assert data["beans_loaded"] > 0

    def test_health_has_feature_dim(self, client):
        data = client.get("/api/health").json()
        assert isinstance(data["feature_dim"], int)
        assert data["feature_dim"] > 0


class TestVersion:
    def test_version_returns_expected_fields(self, client):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "api_version" in data
        assert "model_loaded" in data
        assert "feature_dim" in data
        assert isinstance(data["feature_dim"], int)


class TestPredict:
    def test_predict_default_body(self, client):
        resp = client.post("/api/predict", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert 60 <= data["score"] <= 100
        assert "grade" in data
        assert "attribution" in data
        attr = data["attribution"]
        assert "G" in attr
        assert "P" in attr
        assert "R" in attr
        assert "B" in attr

    def test_predict_custom_gesha_ethiopia(self, client):
        body = {
            "G": {
                "variety": "Gesha",
                "altitude_m": 2100,
                "country": "Ethiopia",
                "region": "Yirgacheffe",
                "soil_type": "volcanic",
                "shade_pct": 50,
                "latitude": 6.5,
                "delta_t_c": 14,
            },
            "P": {
                "method": "natural",
                "anaerobic": True,
                "fermentation_hours": 72,
                "drying_method": "raised_bed",
                "drying_days": 18,
            },
        }
        resp = client.post("/api/predict", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert 60 <= data["score"] <= 100
        assert data["grade"] in (
            "Outstanding", "Excellent", "Very Good", "Good", "Fair", "Below Specialty"
        )

    def test_predict_attribution_sums_to_one(self, client):
        resp = client.post("/api/predict", json={})
        attr = resp.json()["attribution"]
        total = attr["G"] + attr["P"] + attr["R"] + attr["B"]
        assert abs(total - 1.0) < 0.05

    def test_predict_has_feature_dim(self, client):
        resp = client.post("/api/predict", json={})
        data = resp.json()
        assert "feature_dim" in data


class TestRecommend:
    def test_recommend_defaults_returns_beans(self, client):
        resp = client.post("/api/recommend", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "beans" in data
        assert len(data["beans"]) > 0
        assert len(data["beans"]) <= 10

    def test_recommend_custom_top_k(self, client):
        resp = client.post("/api/recommend", json={"top_k": 3})
        data = resp.json()
        assert len(data["beans"]) == 3

    def test_recommend_sorted_by_combined_score(self, client):
        resp = client.post("/api/recommend", json={"top_k": 10})
        beans = resp.json()["beans"]
        scores = [b["combined_score"] for b in beans]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_bean_has_required_fields(self, client):
        resp = client.post("/api/recommend", json={"top_k": 1})
        bean = resp.json()["beans"][0]
        required = {
            "name", "country", "variety", "process", "altitude",
            "predicted_score", "pref_match", "combined_score", "actual_score",
        }
        assert required.issubset(bean.keys())


class TestExplore:
    def test_explore_defaults(self, client):
        resp = client.post("/api/explore", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["combinations"] > 0
        assert data["best"] is not None
        assert len(data["results"]) > 0

    def test_explore_results_sorted_descending(self, client):
        resp = client.post("/api/explore", json={})
        results = resp.json()["results"]
        scores = [r["predicted_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestModelInfo:
    def test_model_info_returns_metadata(self, client):
        resp = client.get("/api/model-info")
        assert resp.status_code == 200
        data = resp.json()
        if "error" not in data:
            assert "val_mae" in data
            assert "feature_names" in data
            assert data.get("version") == "v2"
