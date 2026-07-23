from __future__ import annotations

from hengshui_insar.audit import release_acceptance


def test_full_release_audit() -> None:
    result = release_acceptance({"tests_status": "passed", "wheel_build_status": "passed", "clean_venv_install_status": "passed", "cli_smoke_test_status": "passed"})
    assert result["manifest_hash_match"] is True
    assert result["cache_hash_match"] is True
    assert result["common_mask_hash_match"] is True
