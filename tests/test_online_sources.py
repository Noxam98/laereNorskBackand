"""Источники слов онлайн-комнаты: ручной список использует точные pool_id и не обходит видимость."""
import pytest

from db import get_online_words_by_ids
from db.core import _conn, _release


async def test_selected_words_preserve_order_and_visibility(fresh_db):
    owner_id, owner_dict = await pytest.seed_user("owner")
    other_id, other_dict = await pytest.seed_user("other")
    public_id, _ = await pytest.seed_word(owner_dict, "arbeid", "работа")
    own_id, _ = await pytest.seed_word(owner_dict, "avtale", "договорённость")
    foreign_id, _ = await pytest.seed_word(other_dict, "møte", "встреча")

    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET approved = 0, created_by = ? WHERE id = ?", (owner_id, own_id))
        await db.execute("UPDATE word_pool SET approved = 0, created_by = ? WHERE id = ?", (other_id, foreign_id))
        await db.commit()
    finally:
        await _release(db)

    words = await get_online_words_by_ids([own_id, foreign_id, public_id, own_id], owner_id)
    assert [word["norwegian"] for word in words] == ["avtale", "arbeid"]
    assert words[0]["translate"]["ru"] == ["договорённость"]
