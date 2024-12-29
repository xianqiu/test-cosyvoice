# Copyright (c) 2024 Xingchen Song (sxc19@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from tn.english.rules.word import Word
from tn.english.test.utils import parse_test_case


class TestWord:

    word = Word()
    word_cases = parse_test_case('data/word.txt')

    @pytest.mark.parametrize("written, spoken", word_cases)
    def test_char(self, written, spoken):
        assert self.word.normalize(written) == spoken
