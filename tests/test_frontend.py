"""Static checks on the single-file frontend.

These exist because the JS harness executes the script but does not apply CSS,
and pytest never opens a browser - so a whole class of bug (an element that
will not hide, a call that forgets its auth header) is invisible to both.
"""

import re

import pytest

from app import config

HTML = (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")
SCRIPT = re.search(r"<script[^>]*>(.*?)</script>", HTML, re.DOTALL).group(1)
STYLE = re.search(r"<style[^>]*>(.*?)</style>", HTML, re.DOTALL).group(1)


def test_hidden_attribute_is_forced_to_win():
    """The bug that left the sign-in panel on screen after a valid login.

    `[hidden]` only hides an element while nothing else sets `display`.
    #gate, main and .who-picker all set it, so the override must be present.
    """
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important",
                     STYLE), "missing the [hidden] display override"


def test_every_element_toggled_by_hidden_is_covered():
    """Any element the script hides must be one the override applies to."""
    toggled = set(re.findall(r"(\w+)(?:El)?\.hidden\s*=", SCRIPT))
    assert toggled, "no element is toggled via .hidden - has the UI changed?"
    # The override is global, so simply assert it exists ahead of any rule
    # that sets display.
    override_at = STYLE.find("[hidden]")
    first_display = STYLE.find("display:")
    assert override_at != -1
    assert override_at < first_display or first_display == -1, (
        "[hidden] override must come before rules that set display"
    )


def test_no_client_supplied_session_identifier_remains():
    """Identity comes from the token; a client-chosen id would be spoofable."""
    assert "session_id" not in SCRIPT
    assert "va_session" not in SCRIPT


def test_token_is_kept_in_session_storage_not_local_storage():
    assert "sessionStorage.setItem" in SCRIPT
    assert not re.search(r"localStorage\.setItem\(\s*TOKEN_KEY", SCRIPT)


@pytest.mark.parametrize("endpoint", ["/ask-text", "/ask-audio", "/clear"])
def test_authenticated_endpoints_go_through_the_api_wrapper(endpoint):
    """api() attaches the bearer token and handles 401 centrally."""
    assert f'api("{endpoint}"' in SCRIPT, f"{endpoint} bypasses api()"
    assert f'fetch("{endpoint}"' not in SCRIPT, f"{endpoint} uses raw fetch()"


def test_api_wrapper_sends_the_bearer_token_and_handles_401():
    assert 'Authorization' in SCRIPT and 'Bearer ' in SCRIPT
    assert "r.status === 401" in SCRIPT


def test_both_endpoints_share_one_stream_consumer():
    """/ask-text and /ask-audio emit identical events; one reader handles both."""
    assert SCRIPT.count("function consumeStream") == 1
    assert SCRIPT.count("return consumeStream(r);") == 2


@pytest.mark.parametrize("event", ["stage", "delta", "final", "audio"])
def test_stream_consumer_handles_every_server_event(event):
    assert f'evt.event === "{event}"' in SCRIPT, f"UI ignores {event!r} events"


def test_pipeline_covers_every_stage_the_server_emits():
    """A stage the server sends but the strip lacks would silently never light."""
    from app import main  # noqa: PLC0415 - imported here to read the source

    import inspect

    server = set(re.findall(r'"stage":\s*"(\w+)"', inspect.getsource(main)))
    for name in server:
        assert f'data-stage="{name}"' in HTML, f"pipeline strip missing '{name}'"


def test_orb_is_fed_from_microphone_and_playback():
    """Both amplitude sources must drive the orb, or it looks dead while speaking."""
    assert SCRIPT.count("feedOrb(") >= 3
    assert "createMediaElementSource" in SCRIPT, "playback never reaches the orb"


def test_innerhtml_is_only_used_with_escaped_or_constant_content():
    """The SQL panel is the one place raw HTML is injected.

    LLM-generated SQL is untrusted text. `highlightSql` escapes before adding
    markup (tests/highlight_check.js proves it), so every other `innerHTML`
    write must be a constant or a clearing assignment.
    """
    assert "function escapeHtml" in SCRIPT
    assert "escapeHtml(sql)" in SCRIPT, "highlightSql no longer escapes first"

    for match in re.findall(r"\.innerHTML\s*=\s*([^;\n]+)", SCRIPT):
        value = match.strip()
        allowed = (
            value in ('""', "''")                 # clearing
            or value.startswith('"') or value.startswith("'")  # literal
            or value == "TALK_LABEL"              # module constant
            or "highlightSql" in value            # escaped above
        )
        assert allowed, f"unreviewed innerHTML assignment: {value}"


def test_every_getelementbyid_target_exists_in_the_markup():
    for ident in sorted(set(re.findall(r'getElementById\("([^"]+)"\)', SCRIPT))):
        assert f'id="{ident}"' in HTML, f"getElementById('{ident}') has no element"
