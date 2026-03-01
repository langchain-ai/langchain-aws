import pytest
import sys

from tests.integration_tests.utils import (
    add,
    multiply,
    get_weather,
    generate_large_data,
)


@pytest.mark.parametrize("size_kb", [10, 100, 500], ids=["10KB", "100KB", "500KB"])
def test_generate_large_data(size_kb: int):
    data: str = generate_large_data(size_kb)
    memsize_b: int = sys.getsizeof(data)
    memsize_kb: float = memsize_b // 1024
    assert memsize_kb >= size_kb


class TestAgentTools:
    
    def test_add():
        assert add.invoke({"a": 5, "b": 3}) == 8

    def test_multiply():
        assert multiply.invoke({"a": 4, "b": 6}) == 24

    def test_get_weather():
        assert get_weather.invoke("sf") == "It's always sunny in sf"
        assert get_weather.invoke("nyc") == "It might be cloudy in nyc"