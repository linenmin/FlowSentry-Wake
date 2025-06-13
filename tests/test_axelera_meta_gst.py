# Copyright Axelera AI, 2025
import numpy as np
import pytest

from axelera.app.meta import gst

NULL_TS = b'\x00' * 8
FORTY_TWO_TS = np.uint64(40_000_000_002).tobytes()


@pytest.mark.parametrize(
    'sid, ts, exp',
    [
        (b'\x00\x00\x00\x00', NULL_TS, (0, 0)),
        (b'\xff\x00\x00\x00', NULL_TS, (255, 0)),
        (b'\xff\x00\x00\x00', FORTY_TWO_TS, (255, 40.000000002)),
        (b'\xff', NULL_TS, 'Expecting stream_id to be a byte stream 4 bytes long'),
        (b'\xff' * 5, NULL_TS, 'Expecting stream_id to be a byte stream 4 bytes long'),
        (b'\x00' * 4, None, "Expecting timestamp meta element"),
        (b'\x00' * 4, b'\x00' * 7, "Expecting timestamp to be a byte stream 8 bytes long"),
        (b'\x00' * 4, b'\x00' * 9, "Expecting timestamp to be a byte stream 8 bytes long"),
        (None, NULL_TS, "Expecting stream_id meta element"),
    ],
)
def test_decode_stream_meta(sid, ts, exp):
    data = {} if sid is None else {'stream_id': sid}
    if ts is not None:
        data['timestamp'] = ts
    if isinstance(exp, str):
        with pytest.raises(RuntimeError, match=exp):
            gst.decode_stream_meta(data)
    else:
        assert exp == gst.decode_stream_meta(data)
