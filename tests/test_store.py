"""Unit tests for the app store (conversations + sanctioned paths)."""
from groundwire.store import Store


def test_conversation_roundtrip_with_provenance(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    cid = s.new_conversation("First chat", "qwen2.5:7b")
    s.add_message(cid, "user", "how does it start?")
    s.add_message(cid, "assistant", "It opens…", model="qwen2.5:7b",
                  sources=[{"file": "drop/wp.txt", "cids": [24, 5]}])
    conv = s.get_conversation(cid)
    assert conv["title"] == "First chat"
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    # provenance (which chunks the answer used) survives the round-trip
    assert conv["messages"][1]["sources"][0]["file"] == "drop/wp.txt"
    assert s.list_conversations()[0]["id"] == cid


def test_rename_and_delete(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    cid = s.new_conversation("tmp", None)
    s.rename_conversation(cid, "renamed")
    assert s.list_conversations()[0]["title"] == "renamed"
    s.delete_conversation(cid)
    assert s.list_conversations() == []


def test_sanctioned_paths_crud(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    pid = s.add_path("/docs", "docs", local_only=True)
    p = s.list_paths()[0]
    assert p["path"] == "/docs" and p["local_only"] and p["enabled"]
    s.set_path_enabled(pid, False)
    assert not s.list_paths()[0]["enabled"]
    s.remove_path(pid)
    assert s.list_paths() == []


def test_local_only_excluded_for_cloud(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    s.add_path("/public", "public", local_only=False)
    s.add_path("/secret", "secret", local_only=True)
    s.add_path("/off", "off", local_only=False)
    s.set_path_enabled(s.list_paths()[-1]["id"], False)     # disabled -> never used
    # cloud backend: local-only paths are withheld
    assert {p["scope"] for p in s.paths_for(cloud=True)} == {"public"}
    # local backend: all enabled paths, including local-only
    assert {p["scope"] for p in s.paths_for(cloud=False)} == {"public", "secret"}
