"""Binwalk Analyzer for Image Submissions."""

import os
import shutil
import subprocess
from pathlib import Path

from .utils import MAX_PENDING_TIME, update_data


def analyze_binwalk(input_img: Path, output_dir: Path, extract: bool = False) -> None:
    """Analyze an image submission using binwalk. Scan-only by default."""

    if not input_img.exists():
        update_data(
            output_dir,
            {"binwalk": {"status": "error", "error": f"Input image not found: {input_img}"}}
        )
        return

    if not output_dir.exists():
        update_data(
            output_dir,
            {"binwalk": {"status": "error", "error": f"Output directory not found: {output_dir}"}}
        )
        return

    image_name = input_img.name
    extracted_dir = output_dir / f"_{image_name}.extracted"

    try:
        stderr = ""
        cmd = ["binwalk", "../" + str(image_name)]
        if extract:
            run_as = os.getenv("USER", "app")
            cmd = ["binwalk", "-e", f"--run-as={run_as}", "../" + str(image_name)]

        try:
            data = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=MAX_PENDING_TIME,
            )
            stderr += data.stderr
        except subprocess.TimeoutExpired:
            update_data(
                output_dir,
                {
                    "binwalk": {
                        "status": "error",
                        "error": f"Binwalk timed out after {MAX_PENDING_TIME} seconds",
                    }
                },
            )
            return
        except FileNotFoundError:
            update_data(
                output_dir,
                {
                    "binwalk": {
                        "status": "error",
                        "error": "binwalk command not found. Install binwalk to use this analyzer.",
                    }
                },
            )
            return
        except Exception as e:
            update_data(
                output_dir,
                {"binwalk": {"status": "error", "error": f"Failed to run binwalk: {str(e)}"}},
            )
            return

        zip_exist = False
        if extract and extracted_dir.exists():
            try:
                zip_data = subprocess.run(
                    ["7z", "a", "../binwalk.7z", "*"],
                    cwd=extracted_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=MAX_PENDING_TIME,
                )
                zip_exist = True
                stderr += zip_data.stderr
            except subprocess.TimeoutExpired:
                print(f"Warning: 7z archive creation timed out after {MAX_PENDING_TIME} seconds")
            except FileNotFoundError:
                print("Warning: 7z command not found, cannot create archive of extracted files")
            except Exception as e:
                print(f"Warning: Failed to create 7z archive: {str(e)}")

        if "root privileges" in stderr.lower():
            update_data(
                output_dir,
                {
                    "binwalk": {
                        "status": "skipped",
                        "error": "Binwalk extraction requires elevated helpers; ran scan-only.",
                        "output": data.stdout.split("\n") if data else [],
                    }
                },
            )
            return

        if len(stderr) > 0 and extract:
            err = {
                "binwalk": {
                    "status": "error",
                    "error": stderr,
                }
            }
            update_data(output_dir, err)
            return

        # Remove the extracted directory
        if extracted_dir.exists():
            try:
                shutil.rmtree(extracted_dir)
            except Exception as e:
                print(f"Warning: Failed to remove extracted directory: {str(e)}")

        status = "ok"
        output_data = {
            "binwalk": {
                "status": status,
                "output": data.stdout.split("\n") if data else [],
            }
        }
        if zip_exist:
            output_data["binwalk"]["download"] = f"/download/{output_dir.name}/binwalk"

        update_data(output_dir, output_data)

    except Exception as e:
        update_data(
            output_dir,
            {"binwalk": {"status": "error", "error": f"Binwalk analysis failed: {str(e)}"}}
        )
