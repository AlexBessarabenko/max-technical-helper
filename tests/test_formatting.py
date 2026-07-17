"""Санитизация LLM-markdown под подмножество Markdown, поддерживаемое MAX."""

from src.bot.formatting import to_max_markdown


def test_plain_text_unchanged():
    assert to_max_markdown("Просто текст без разметки.") == "Просто текст без разметки."


def test_headings_become_bold():
    assert to_max_markdown("# Как настроить VPN") == "**Как настроить VPN**"
    assert to_max_markdown("## Шаг 1") == "**Шаг 1**"
    assert to_max_markdown("### Подраздел") == "**Подраздел**"
    assert to_max_markdown("#### Детали") == "**Детали**"


def test_heading_inside_text():
    text = "Вот инструкция:\n\n## VPN на Windows\n\nДальше шаги."
    assert to_max_markdown(text) == "Вот инструкция:\n\n**VPN на Windows**\n\nДальше шаги."


def test_heading_with_closing_hashes():
    # ATX-заголовок может закрываться решётками.
    assert to_max_markdown("## Заголовок ##") == "**Заголовок**"


def test_not_a_heading():
    # Решётка без пробела — не заголовок.
    assert to_max_markdown("#хэштег") == "#хэштег"
    # Пять и более решёток — вне диапазона h1-h4, оставляем как есть.
    assert to_max_markdown("##### слишком глубоко") == "##### слишком глубоко"


def test_horizontal_rules_removed():
    assert to_max_markdown("Раздел 1\n---\nРаздел 2") == "Раздел 1\n\nРаздел 2"
    assert to_max_markdown("a\n***\nb") == "a\n\nb"
    assert to_max_markdown("a\n___\nb") == "a\n\nb"


def test_supported_markup_kept():
    text = "**жирный**, *курсив*, `код`, ~~зачёркнутый~~, [ссылка](https://example.com)"
    assert to_max_markdown(text) == text


def test_lists_kept():
    text = "Шаги:\n- установить клиент\n- ввести логин\n\n1. раз\n2. два"
    assert to_max_markdown(text) == text


def test_list_item_is_not_horizontal_rule():
    assert to_max_markdown("- пункт списка") == "- пункт списка"


def test_empty_and_multiline():
    assert to_max_markdown("") == ""
    assert to_max_markdown("## Один\n\n---\n\nТекст") == "**Один**\n\n\n\nТекст"
