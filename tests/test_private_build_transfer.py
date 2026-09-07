import os
import zipfile

import pytest

from tools.private_build_transfer import archive, crypt, extract


def test_authenticated_private_transfer_roundtrip_and_tamper(tmp_path):
    pytest.importorskip("cryptography")
    source = tmp_path / "source"
    source.mkdir()
    (source / "private-video.mp4").write_bytes(b"Private footage fixture")
    archive(source, tmp_path / "plain.zip")
    key = os.urandom(32)
    crypt(tmp_path / "plain.zip", tmp_path / "secret.bvenc", key)
    assert b"Private footage" not in (tmp_path / "secret.bvenc").read_bytes()
    crypt(tmp_path / "secret.bvenc", tmp_path / "received.zip", key, decrypt=True)
    extract(tmp_path / "received.zip", tmp_path / "received")
    assert (tmp_path / "received/private-video.mp4").read_bytes() == b"Private footage fixture"
    encrypted = bytearray((tmp_path / "secret.bvenc").read_bytes())
    encrypted[-1] ^= 1
    (tmp_path / "corrupted.bvenc").write_bytes(encrypted)
    from cryptography.exceptions import InvalidTag
    with pytest.raises(InvalidTag):
        crypt(tmp_path / "corrupted.bvenc", tmp_path / "rejected.zip", key, decrypt=True)
    assert not (tmp_path / "rejected.zip").exists()
    assert not list(tmp_path.glob(".transfer-*"))


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "/escape", "nested\\..\\escape"])
def test_private_zip_rejects_traversal_before_creating_destination(tmp_path, name):
    with zipfile.ZipFile(tmp_path / "unsafe.zip", "w") as output:
        output.writestr(name, b"no")
    with pytest.raises(ValueError):
        extract(tmp_path / "unsafe.zip", tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
