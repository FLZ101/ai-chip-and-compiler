import os
import sys

import lit.formats

config.name = "mlir"
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root
config.test_format = lit.formats.ShTest(False)

python = sys.executable
filecheck = os.path.join(os.path.dirname(sys.executable), "filecheck")
if not os.path.exists(filecheck):
    raise SystemExit("filecheck is required to run the MLIR lit tests")

config.substitutions.append(("%python", python))
config.substitutions.append(("%filecheck", filecheck))
