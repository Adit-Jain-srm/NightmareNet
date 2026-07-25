import pytest

pytest.importorskip("sqlalchemy")

from nightmarenet_server.models import tables


def test_user_instantiation():
    user = tables.User(email="test@example.com", name="Test User")
    assert user.email == "test@example.com"
    assert user.name == "Test User"
    assert user.__tablename__ == "users"

def test_org_instantiation():
    org = tables.Org(name="Test Org", plan_tier="pro")
    assert org.name == "Test Org"
    assert org.plan_tier == "pro"

def test_org_member_instantiation():
    member = tables.OrgMember(org_id="org_1", user_id="user_1", role="admin")
    assert member.org_id == "org_1"
    assert member.user_id == "user_1"
    assert member.role == "admin"

def test_project_instantiation():
    project = tables.Project(name="Proj", org_id="org_1")
    assert project.name == "Proj"
    assert project.org_id == "org_1"

def test_experiment_instantiation():
    exp = tables.Experiment(name="Exp", project_id="proj_1", status="idle")
    assert exp.name == "Exp"
    assert exp.status == "idle"

def test_run_instantiation():
    run = tables.Run(experiment_id="exp_1", status="pending")
    assert run.experiment_id == "exp_1"
    assert run.status == "pending"

def test_run_event_instantiation():
    event = tables.RunEvent(run_id="run_1", event_type="start")
    assert event.run_id == "run_1"
    assert event.event_type == "start"

def test_api_key_instantiation():
    key = tables.ApiKey(org_id="org_1", user_id="user_1", key_hash="hash", name="default")
    assert key.key_hash == "hash"
    assert key.name == "default"
