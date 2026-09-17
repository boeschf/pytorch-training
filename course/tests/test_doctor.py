from pytorch_course.doctor import (
    collect_report,
    read_image_metadata,
    validate_profile,
    validate_report,
)


def test_image_reference_is_immutable() -> None:
    image = read_image_metadata()

    assert image["COURSE_IMAGE_DIGEST"].startswith("sha256:")
    assert image["COURSE_IMAGE"].endswith("@" + image["COURSE_IMAGE_DIGEST"])


def test_host_report_has_required_structure() -> None:
    report = collect_report()

    assert report["python"]["version"]
    assert report["host"]["machine"]
    assert report["course"]["image"]["COURSE_IMAGE_TAG"]
    assert (
        validate_report(
            report,
            require_torch=False,
            require_cuda=False,
            minimum_gpus=0,
            require_profile=None,
        )
        == []
    )


def torch_report(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "available": True,
        "version": "2.11.0",
        "inside_environment": True,
        "compiled_cuda": None,
        "distributed_available": True,
        "nccl_available": True,
    }
    return report | overrides


def test_local_profiles_enforce_their_locked_torch_builds() -> None:
    cpu_report = {"torch": torch_report(compiled_cuda="12.8"), "course": {"image": {}}}
    cuda_report = {"torch": torch_report(compiled_cuda=None), "course": {"image": {}}}
    outdated_report = {
        "torch": torch_report(version="2.10.0"),
        "course": {"image": {}},
    }

    assert validate_profile(cpu_report, "cpu") == [
        "cpu profile requires a CPU-only PyTorch build, found CUDA 12.8"
    ]
    assert validate_profile(cuda_report, "cuda") == [
        "cuda profile requires the CUDA 12.8 PyTorch build, found None"
    ]
    assert validate_profile(outdated_report, "cpu") == [
        "cpu profile requires PyTorch 2.11.0, found 2.10.0"
    ]


def test_alps_profile_rejects_environment_owned_torch() -> None:
    report = {
        "torch": torch_report(
            version="2.13.0a0+9186a08",
            inside_environment=True,
            compiled_cuda="13.0",
        ),
        "course": {"image": {"COURSE_IMAGE_PYTORCH_VERSION": "2.13.0a0+9186a08"}},
    }

    assert validate_profile(report, "alps-gh200") == [
        "alps-gh200 profile must use image-provided PyTorch"
    ]
