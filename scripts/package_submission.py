"""Build a validated Agenthon submission ZIP without ever storing the Team Key.

Requires qfbench2-common 2.4.4. The Team Key is read with getpass, used only to derive the team
alias and HMAC proof, then discarded when the process exits. It is never printed or written.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
import pathlib
import re
import zipfile


HOUSE_MODEL = {
    "name": "nvidia/nemotron-3-super-120b-a12b",
    "version": "rl-030326-fp8",
    "revision": "rl-030326-fp8",
    "training_cutoff": "unpublished",
    "access": "api",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create submission.zip with a sealed descriptor and v2 team claim")
    parser.add_argument("--team-number", required=True, type=int, help="numeric team number from agenthon.net")
    parser.add_argument("--registry", required=True, help="registry hostname, for example ghcr.io")
    parser.add_argument("--repository", required=True, help="lowercase image path, for example owner/t2-solution")
    parser.add_argument("--digest", required=True, help="immutable sha256:<64 lowercase hex> image digest")
    parser.add_argument("--phase", choices=("dev", "final"), default="dev")
    parser.add_argument("--image-access", choices=("public", "organizer_mirror"), default="public")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("submissions/submission.zip"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.team_number <= 0:
        raise SystemExit("--team-number must be positive")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.digest):
        raise SystemExit("--digest must be sha256: followed by 64 lowercase hexadecimal characters")
    if args.repository != args.repository.lower():
        raise SystemExit("--repository must be lowercase")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing archive: {args.out}")

    try:
        import qfbench2_common
        from qfbench2_common.contracts.descriptor import SubmissionDescriptor, seal_descriptor_digest
    except ImportError as exc:
        raise SystemExit(
            "qfbench2-common 2.4.4 is required; install the project's [submission] extra"
        ) from exc
    if qfbench2_common.__version__ != "2.4.4":
        raise SystemExit(f"qfbench2-common 2.4.4 required, found {qfbench2_common.__version__}")

    team_key = getpass.getpass("Team Key (hidden; never stored): ")
    if not team_key:
        raise SystemExit("empty Team Key refused")
    team_id = "team-" + hashlib.sha256(
        b"agenthon2026-team-alias:" + str(args.team_number).encode() + b":" + team_key.encode()
    ).hexdigest()[:32]
    competition_id = "agenthon2026-forecasting-dev" if args.phase == "dev" else "agenthon2026-forecasting-final"
    descriptor = {
        "schema_version": "1.1.0",
        "interface_version": "2.0",
        "competition_id": competition_id,
        "team_id": team_id,
        "track": "forecasting",
        "phase": args.phase,
        "category": "api",
        "image": {"registry": args.registry, "repository": args.repository, "digest": args.digest},
        "image_access": args.image_access,
        "models": [HOUSE_MODEL],
        "license": "MIT",
    }
    sealed = seal_descriptor_digest(descriptor)
    parsed = SubmissionDescriptor.from_mapping(sealed)
    parsed.matches_plan(competition_id=competition_id, track="forecasting", phase=args.phase)

    descriptor_bytes = (json.dumps(sealed, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor_sha256 = hashlib.sha256(descriptor_bytes).hexdigest()
    proof_key = hashlib.sha256(b"agenthon2026-team-proof-key:v2:" + team_key.encode()).digest()
    message = (b"agenthon2026-team-claim:v2:" + str(args.team_number).encode()
               + b":" + descriptor_sha256.encode())
    proof = hmac.new(proof_key, message, hashlib.sha256).hexdigest()
    claim = {"schema_version": "2.0", "site_team_id": args.team_number,
             "descriptor_sha256": descriptor_sha256, "proof": proof}
    claim_bytes = json.dumps(claim, separators=(",", ":"), sort_keys=True).encode("utf-8")

    # This assertion catches accidental secret interpolation before anything reaches disk.
    if team_key.encode() in descriptor_bytes or team_key.encode() in claim_bytes:
        raise SystemExit("internal safety check failed: Team Key would be written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(args.out, flags, 0o600)
    try:
        with zipfile.ZipFile(os.fdopen(fd, "wb"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("submission.json", descriptor_bytes)
            archive.writestr("team-claim.json", claim_bytes)
    except Exception:
        args.out.unlink(missing_ok=True)
        raise

    validate_archive(args.out, descriptor_sha256)
    print(f"created {args.out} for {competition_id}")
    print(f"team alias: {team_id}")
    print(f"image: {parsed.image_reference()}")
    print("validated: descriptor digest, phase binding, v2 claim hash, and two-file ZIP layout")
    return 0


def validate_archive(path: pathlib.Path, expected_descriptor_hash: str) -> None:
    from qfbench2_common.contracts.descriptor import SubmissionDescriptor
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["submission.json", "team-claim.json"]:
            raise ValueError("submission ZIP must contain exactly two root files")
        descriptor_bytes = archive.read("submission.json")
        claim = json.loads(archive.read("team-claim.json"))
        if hashlib.sha256(descriptor_bytes).hexdigest() != expected_descriptor_hash:
            raise ValueError("submission.json byte hash changed while packing")
        if claim.get("descriptor_sha256") != expected_descriptor_hash or claim.get("schema_version") != "2.0":
            raise ValueError("team claim is not bound to the packed descriptor")
        SubmissionDescriptor.from_mapping(json.loads(descriptor_bytes))


if __name__ == "__main__":
    raise SystemExit(main())
