"""Authenticated, streaming private build transfer. Never prints the key.

Install cryptography separately on the build host; not an application dependency.
Only encrypted artifacts leave CI. The key is supplied as a GitHub secret or
a local 0600 file outside the input folder, never a command-line value.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

MAGIC = b"BVPRIVATE1"


def crypt(source: Path, target: Path, key: bytes, *, decrypt: bool = False) -> None:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    if len(key) != 32:
        raise ValueError("Expected a 256-bit transfer key")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(prefix=".transfer-", dir=target.parent)
    try:
        with source.open("rb") as reader, os.fdopen(fd, "wb") as writer:
            if decrypt:
                if reader.read(len(MAGIC)) != MAGIC:
                    raise ValueError("Unknown encrypted artifact format")
                nonce = reader.read(12)
                remaining = source.stat().st_size - len(MAGIC) - 12 - 16
                if len(nonce) != 12 or remaining < 0:
                    raise ValueError("Truncated encrypted artifact")
                reader.seek(-16, 2)
                tag = reader.read(16)
                reader.seek(len(MAGIC) + 12)
                operation = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
            else:
                nonce = os.urandom(12)
                operation = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
                writer.write(MAGIC + nonce)
                remaining = source.stat().st_size
            operation.authenticate_additional_data(MAGIC)
            while remaining:
                chunk = reader.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Source changed during transfer")
                writer.write(operation.update(chunk))
                remaining -= len(chunk)
            writer.write(operation.finalize())
            if not decrypt:
                writer.write(operation.tag)
        # Decrypted output is exposed only after GCM authentication succeeds.
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def archive(folder: Path, target: Path) -> None:
    if target.exists() or target.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Archive must be new and outside the input directory")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as output:
        for source in sorted(folder.rglob("*")):
            if source.is_symlink():
                raise ValueError("Symlinks are not allowed in private build transfers")
            if source.is_file():
                output.write(source, source.relative_to(folder).as_posix())


def extract(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(source) as incoming:
        members = incoming.infolist()
        for member in members:
            name = member.filename
            posix, windows = PurePosixPath(name), PureWindowsPath(name)
            if (posix.is_absolute() or windows.drive or ".." in posix.parts or ".." in windows.parts
                    or "\\" in name or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe archive member")
        if sum(member.file_size for member in members) > 12 * 1024**3:
            raise ValueError("Archive exceeds the build input budget")
        target.mkdir(parents=True)
        for member in members:
            destination = target / member.filename
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with incoming.open(member) as reader, destination.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["encrypt", "decrypt", "pack", "unpack"])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.action == "pack":
        archive(args.source, args.target)
    elif args.action == "unpack":
        extract(args.source, args.target)
    else:
        value = args.key_file.read_text().strip() if args.key_file else os.environ["BOXING_WINDOWS_TRANSFER_KEY"]
        crypt(args.source, args.target, bytes.fromhex(value), decrypt=args.action == "decrypt")


if __name__ == "__main__":
    main()
