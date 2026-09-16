from pytorch_course.doctor import collect_report, read_image_metadata, validate_report


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
            require_image_torch=False,
        )
        == []
    )
