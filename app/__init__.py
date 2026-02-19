import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402  # must import before numpy (Windows MKL DLL conflict)
