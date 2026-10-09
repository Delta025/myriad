import pytest

from myriad.testing import VARIANTS, make_tiny_checkpoint


@pytest.fixture(scope="session", params=sorted(VARIANTS))
def tiny_checkpoint(request, tmp_path_factory):
    """Path to a tiny random-weight Gemma 4 checkpoint, once per variant."""
    variant = request.param
    return make_tiny_checkpoint(tmp_path_factory.mktemp(f"tiny-{variant}"), variant)
