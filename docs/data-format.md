# Data Format

The libaom instrumentation writes fixed-size records in little-endian order.

Header format: `<HHHHHHBBBB`

Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `frame_width` | `uint16` | Frame width in pixels. |
| `frame_height` | `uint16` | Frame height in pixels. |
| `blk_x` | `uint16` | Block origin X in pixels. |
| `blk_y` | `uint16` | Block origin Y in pixels. |
| `blk_w` | `uint16` | Actual block width. |
| `blk_h` | `uint16` | Actual block height. |
| `partition_mode` | `uint8` | AV1 partition class, 0 through 9. |
| `qindex` | `uint8` | Raw AV1 base qindex, preserved exactly. |
| `level` | `uint8` | `0=64x64`, `1=32x32`, `2=16x16`. |
| `reserved` | `uint8` | Reserved for schema-compatible future use. |
| `y_data` | `int16[4096]` | 64x64 Y-plane block, padded with zeros when smaller. |

Record size is 8208 bytes: 16 bytes of header plus 8192 bytes of Y data.

Converted datasets are compressed NPZ files with arrays:

- `block_y`: `(N, 1, 64, 64)`, normalized to `[0,1]`.
- `partition_mode`: `(N,)`, int64.
- `qindex`: `(N,)`, uint8.
- `qindex_norm`: `(N,)`, float32 in `[0,1]`.
- `level`: `(N,)`, int64.
- `coords`: `(N,2)`, int64.
- `frame_size`: `(N,2)`, int64.
- `block_size`: `(N,2)`, int64.
- `sequence_id`: `(N,)`, string. Converted logs default to `unknown`; production datasets must populate real sequence IDs before splitting.
