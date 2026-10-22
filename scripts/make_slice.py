"""Cut the first N requests out of a (possibly .zst-compressed) oracleGeneral trace and save
them as a small uncompressed .bin you can zip and upload.

  pip install zstandard numpy
  python scripts/make_slice.py web_0.oracleGeneral.zst web_0_slice.bin --n 3000000

3,000,000 requests = 72 MB uncompressed (about 25 MB zipped). Use --n 1000000 if the upload is too big.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import read_records  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--n", type=int, default=3_000_000)
a = ap.parse_args()
rec = read_records(a.src, a.n)
rec.tofile(a.dst)
print(f"wrote {len(rec):,} requests to {a.dst} ({os.path.getsize(a.dst) / 1e6:.1f} MB), "
      f"{len(set(rec['id'].tolist())):,} unique objects")
