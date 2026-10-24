"""CLI wiring: argument parsing and the error path, without the real dataset."""

from __future__ import annotations

import pytest

from finetune import cli


def test_every_command_parses():
    parser = cli.build_parser()
    assert parser.parse_args(["data-download"]).command == "data-download"
    args = parser.parse_args(
        ["sweep", "--arms", "tfidf", "--ks", "5,all", "--keep-checkpoint", "all:1"]
    )
    assert (args.arms, args.ks, args.keep_checkpoint) == ("tfidf", "5,all", "all:1")
    assert parser.parse_args(["predict", "--model", "m", "hello"]).text == ["hello"]


def test_missing_data_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["data-check", "--data-dir", str(tmp_path)]) == 2
    assert "finetune data-download" in capsys.readouterr().err


def test_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
