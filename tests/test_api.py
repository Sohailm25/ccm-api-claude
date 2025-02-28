from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
import pytest

from database import Base, get_db
from main import app

# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the get_db dependency
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Mock authentication for tests
API_KEY = "test_api_key"
os.environ["API_KEY"] = API_KEY

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    # Create tables for each test
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables after each test
    Base.metadata.drop_all(bind=engine)

def test_create_entry():
    """Test creating a new entry"""
    response = client.post(
        "/entries",
        json={
            "url": "https://example.com",
            "content": "Test content",
            "thoughts": "Test thoughts"
        },
        headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == "https://example.com"
    assert data["content"] == "Test content"
    assert data["thoughts"] == "Test thoughts"
    assert "id" in data
    assert "created_at" in data

def test_unauthorized_access():
    """Test that API key is required"""
    response = client.post(
        "/entries",
        json={
            "url": "https://example.com",
            "content": "Test content",
            "thoughts": "Test thoughts"
        }
    )
    assert response.status_code == 403 