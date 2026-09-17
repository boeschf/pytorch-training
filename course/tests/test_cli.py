from __future__ import annotations

import subprocess
from pathlib import Path

from pytorch_course.cli import main

BASH_PROBE = r"""
source "$1"
shift
COMP_WORDS=("$@")
COMP_CWORD=$((${#COMP_WORDS[@]} - 1))
_shtab_course
printf '%s\n' "${COMPREPLY[@]}"
"""


def complete(script: Path, words: list[str], *, cwd: Path) -> list[str]:
    """Run the packaged completion function for one simulated command line."""
    completed = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            BASH_PROBE,
            "bash",
            str(script),
            *words,
        ],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.splitlines()


def test_bash_completion_covers_commands_choices_and_paths(
    tmp_path: Path,
    capsys,
) -> None:
    script = tmp_path / "course.bash"
    assert main(["completion", "bash"]) == 0
    script.write_text(capsys.readouterr().out, encoding="utf-8")

    assert complete(script, ["course", "tr"], cwd=tmp_path) == ["train"]
    assert complete(
        script,
        ["course", "train", "mlp", "--device", "c"],
        cwd=tmp_path,
    ) == ["cpu", "cuda"]

    output = tmp_path / "result.json"
    output.touch()
    assert complete(
        script,
        ["course", "train", "mlp", "--output", "res"],
        cwd=tmp_path,
    ) == ["result.json"]
