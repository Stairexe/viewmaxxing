# Fake conda for tests: activation is a no-op, the test's python is already on PATH.
conda() { :; }
