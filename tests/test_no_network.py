"""The offline guard in ``tests/conftest.py``: the internet is refused, loopback is not."""

import socket

import pytest


def test_dns_lookup_of_a_real_host_is_refused():
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.getaddrinfo("pypi.org", 443)


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_connection_to_a_real_address_is_refused(method):
    with socket.socket() as sock, pytest.raises(RuntimeError, match="network access is disabled"):
        getattr(sock, method)(("151.101.0.223", 443))  # a pypi.org address


def test_loopback_is_allowed():
    """Streamlit's event loop needs 127.0.0.1; the OS just refuses since nothing listens."""
    with socket.socket() as sock:
        sock.settimeout(5)
        assert sock.connect_ex(("127.0.0.1", 9)) != 0  # an OS error code, not our RuntimeError
    assert socket.getaddrinfo("localhost", 80)
