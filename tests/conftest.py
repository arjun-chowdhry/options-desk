"""Project-wide test setup: no test may reach the network.

The tests must run offline (on a laptop with no Wi-Fi, and in CI), and yfinance is the only
network code, mocked in ``tests/data``. As a guard, every outbound connection and every DNS
lookup of a real host fails loudly. Loopback (127.0.0.1, ::1, localhost) and local Unix
sockets stay open: on Windows, asyncio inside Streamlit wires its event loop with a socket
connected to 127.0.0.1, and that never leaves the machine.

The guard is a session-scoped autouse fixture, so pytest sets it up before any other fixture,
including the session-scoped ones that build the synthetic market.
"""

from __future__ import annotations

import socket

import pytest

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", None}


def _refuse(target: object) -> None:
    raise RuntimeError(f"network access is disabled in tests (tried to reach {target!r})")


@pytest.fixture(scope="session", autouse=True)
def no_network():
    connect, connect_ex, getaddrinfo = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
    )

    def local(address: object) -> bool:
        # Unix sockets have string (or bytes) addresses; IP sockets have (host, port, ...).
        return not isinstance(address, tuple) or address[0] in LOCAL_HOSTS

    def guarded_connect(sock, address, *args, **kwargs):
        if not local(address):
            _refuse(address)
        return connect(sock, address, *args, **kwargs)

    def guarded_connect_ex(sock, address, *args, **kwargs):
        if not local(address):
            _refuse(address)
        return connect_ex(sock, address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in LOCAL_HOSTS:
            _refuse(host)
        return getaddrinfo(host, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", guarded_connect)
        mp.setattr(socket.socket, "connect_ex", guarded_connect_ex)
        mp.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
        yield
