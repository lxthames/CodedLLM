#!/usr/bin/env python3
"""Deterministic, CPU-only same-host TCP transport smoke test.

This is deliberately not a datacenter-network emulator or a CUDA benchmark.
It exercises framed socket payload transfer, out-of-order arrival, XOR parity
recovery, a coordinator-to-consumer handoff, and an auditable event trace.
"""

import argparse
import csv
import json
import platform
import random
import socket
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HEADER = struct.Struct("!IIBI")  # request, shard, flags, byte count
RECOVERED = 1


def recv_exact(conn, count):
    data = bytearray()
    while len(data) < count:
        piece = conn.recv(count - len(data))
        if not piece:
            raise ConnectionError("peer closed framed transfer")
        data.extend(piece)
    return bytes(data)


def xor_bytes(left, right):
    if len(left) != len(right):
        raise ValueError("XOR inputs must have equal lengths")
    return bytes(a ^ b for a, b in zip(left, right))


@dataclass
class Event:
    request_id: int
    kind: str
    timestamp_ns: int
    shard_id: int = -1
    byte_count: int = 0


class Consumer:
    def __init__(self):
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen()
        self.address = self._listener.getsockname()
        self.payloads = {}
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            conn, _ = self._listener.accept()
            with conn:
                request, shard, flags, size = HEADER.unpack(recv_exact(conn, HEADER.size))
                if flags == 255:
                    return
                self.payloads[(request, shard)] = recv_exact(conn, size)

    def close(self):
        with socket.create_connection(self.address) as conn:
            conn.sendall(HEADER.pack(0, 0, 255, 0))
        self._thread.join(timeout=2)
        self._listener.close()


class Coordinator:
    def __init__(self, consumer, events):
        self.consumer = consumer
        self.events = events
        self.shards = {}
        self._lock = threading.Lock()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen()
        self.address = self._listener.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _event(self, request, kind, shard=-1, byte_count=0):
        self.events.append(Event(request, kind, time.monotonic_ns(), shard, byte_count))

    def _serve(self):
        while True:
            conn, _ = self._listener.accept()
            with conn:
                request, shard, flags, size = HEADER.unpack(recv_exact(conn, HEADER.size))
                if flags == 255:
                    return
                payload = recv_exact(conn, size)
                self._event(request, "shard_arrival", shard, size)
                self._accept(request, shard, payload)

    def _accept(self, request, shard, payload):
        with self._lock:
            slots = self.shards.setdefault(request, {})
            slots.setdefault(shard, payload)
            # k=2: 0=A, 1=B, 2=A xor B.  Recover B when A and parity arrive.
            if 1 not in slots and 0 in slots and 2 in slots:
                self._event(request, "planning_start")
                slots[1] = xor_bytes(slots[0], slots[2])
                self._event(request, "recovery_complete", 1, len(slots[1]))
            if 0 in slots and 1 in slots:
                for shard_id in (0, 1):
                    with socket.create_connection(self.consumer.address) as downstream:
                        downstream.sendall(HEADER.pack(request, shard_id, RECOVERED,
                                                       len(slots[shard_id])))
                        downstream.sendall(slots[shard_id])
                self._event(request, "consumer_ready")

    def close(self):
        with socket.create_connection(self.address) as conn:
            conn.sendall(HEADER.pack(0, 0, 255, 0))
        self._thread.join(timeout=2)
        self._listener.close()


def send_shard(address, request, shard, payload, delay_s, events):
    time.sleep(delay_s)
    events.append(Event(request, "producer_send", time.monotonic_ns(), shard, len(payload)))
    with socket.create_connection(address) as conn:
        conn.sendall(HEADER.pack(request, shard, 0, len(payload)))
        conn.sendall(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--shard-bytes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--straggler-delay-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("transport_smoke_events.csv"))
    args = parser.parse_args()
    if args.requests <= 0 or args.shard_bytes <= 0:
        raise SystemExit("requests and shard-bytes must be positive")

    events = []
    consumer = Consumer()
    coordinator = Coordinator(consumer, events)
    rng = random.Random(args.seed)
    expected = {}
    try:
        with ThreadPoolExecutor(max_workers=3 * args.requests) as pool:
            futures = []
            for request in range(args.requests):
                left = rng.randbytes(args.shard_bytes)
                right = rng.randbytes(args.shard_bytes)
                parity = xor_bytes(left, right)
                expected[(request, 0)] = left
                expected[(request, 1)] = right
                # A and parity arrive first; B is independently delayed.
                futures.extend((
                    pool.submit(send_shard, coordinator.address, request, 0, left,
                                0.0005 * request, events),
                    pool.submit(send_shard, coordinator.address, request, 2, parity,
                                0.0005 * request + 0.0002, events),
                    pool.submit(send_shard, coordinator.address, request, 1, right,
                                0.0005 * request + args.straggler_delay_ms / 1000.0,
                                events),
                ))
            for future in futures:
                future.result()

        deadline = time.monotonic() + 5
        while len(consumer.payloads) < 2 * args.requests and time.monotonic() < deadline:
            time.sleep(0.001)
        if consumer.payloads != expected:
            raise RuntimeError("consumer payload differs from original systematic bytes")
    finally:
        coordinator.close()
        consumer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("request_id", "event", "timestamp_ns", "shard_id", "byte_count"))
        for event in sorted(events, key=lambda item: item.timestamp_ns):
            writer.writerow((event.request_id, event.kind, event.timestamp_ns,
                             event.shard_id, event.byte_count))
    try:
        revision = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    with args.output.with_name("manifest.json").open("w", encoding="utf-8") as destination:
        json.dump(
            {
                "harness": "same-host TCP CPU-only smoke test",
                "seed": args.seed,
                "requests": args.requests,
                "shard_bytes": args.shard_bytes,
                "straggler_delay_ms": args.straggler_delay_ms,
                "source_revision": revision,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "completion_boundary": "consumer received host-resident systematic bytes",
            },
            destination,
            indent=2,
            sort_keys=True,
        )
    print(f"PASS: {args.requests} requests, {args.shard_bytes} B shards, events={len(events)}")


if __name__ == "__main__":
    main()
