"""Start the installed course kernel and execute one request."""

from __future__ import annotations

import argparse
import time

from jupyter_client import KernelManager

from pytorch_course.profiles import PROFILES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="cscs-pytorch-course-cpu")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    prefix = "cscs-pytorch-course-"
    expected_profile = args.kernel.removeprefix(prefix)
    if expected_profile not in PROFILES:
        parser.error(f"kernel must identify one of: {', '.join(PROFILES)}")

    manager = KernelManager(kernel_name=args.kernel)
    manager.start_kernel()
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=args.timeout)
        request_id = client.execute(
            "import json, platform; "
            "from pytorch_course.profiles import read_profile; "
            "print(json.dumps({'profile': read_profile(), 'python': platform.python_version()}))"
        )
        deadline = time.monotonic() + args.timeout
        stream_output = ""
        while time.monotonic() < deadline:
            message = client.get_iopub_msg(timeout=max(0.1, deadline - time.monotonic()))
            if message.get("parent_header", {}).get("msg_id") != request_id:
                continue
            message_type = message["header"]["msg_type"]
            if message_type == "stream":
                stream_output += message["content"]["text"]
            elif message_type == "error":
                raise RuntimeError("\n".join(message["content"]["traceback"]))
            elif message_type == "status" and message["content"]["execution_state"] == "idle":
                break
        else:
            raise TimeoutError("kernel did not become idle")

        if f'"profile": "{expected_profile}"' not in stream_output:
            raise RuntimeError(f"unexpected kernel profile: {stream_output!r}")
        if '"python": "3.12.' not in stream_output:
            raise RuntimeError(f"unexpected kernel Python: {stream_output!r}")
        print(f"Kernel {args.kernel}: PASS ({stream_output.strip()})")
        return 0
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    raise SystemExit(main())
