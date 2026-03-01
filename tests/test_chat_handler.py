"""Tests for the chat handler and session manager."""

import pytest

from src.chat.intent import ChatIntent
from src.chat.session import Session, SessionManager


class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager(ttl_seconds=60)
        session = mgr.create("/tmp/test.jpg")
        assert session.id
        assert session.image_path == "/tmp/test.jpg"

    def test_get_session(self):
        mgr = SessionManager()
        session = mgr.create("/tmp/test.jpg")
        retrieved = mgr.get(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    def test_get_nonexistent(self):
        mgr = SessionManager()
        assert mgr.get("nonexistent") is None

    def test_update_session(self):
        mgr = SessionManager()
        session = mgr.create("/tmp/test.jpg")
        mgr.update(session.id, {"prediction": {"name": "Paris"}})
        assert mgr.get(session.id).pipeline_state["prediction"]["name"] == "Paris"

    def test_delete_session(self):
        mgr = SessionManager()
        session = mgr.create("/tmp/test.jpg")
        assert mgr.delete(session.id) is True
        assert mgr.get(session.id) is None

    def test_add_message(self):
        session = Session(id="test", image_path="/tmp/test.jpg")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")
        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "user"


class TestChatIntent:
    def test_intent_enum_values(self):
        assert ChatIntent.ASK_WHY_NOT == "ask_why_not"
        assert ChatIntent.TRY_SEARCH == "try_search"
        assert ChatIntent.GENERAL == "general"
