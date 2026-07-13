from __future__ import annotations

import re
import unittest

from service.schemas import utc_now


class TimestampTests(unittest.TestCase):
    def test_utc_now_keeps_millisecond_precision(self) -> None:
        self.assertRegex(
            utc_now(),
            re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00$"),
        )


if __name__ == "__main__":
    unittest.main()
