"""Download the small real CloudPhysics sample shipped in the libCacheSim repo (2.7 MB,
113,872 requests) to data/. This is a real trace but far too short for final results;
use it for sanity checks. MSR-Cambridge traces must be downloaded from SNIA IOTTA."""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import SAMPLE_URL  # noqa: E402

os.makedirs("data", exist_ok=True)
dst = "data/cloudphysics_sample.oracleGeneral.bin"
urllib.request.urlretrieve(SAMPLE_URL, dst)
print("saved", dst, os.path.getsize(dst), "bytes")
