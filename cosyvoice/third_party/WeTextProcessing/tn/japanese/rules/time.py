# Copyright (c) 2024 Logan Liu (2319277867@qq.com)
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

from tn.processor import Processor
from tn.utils import get_abs_path

from pynini import string_file
from pynini.lib.pynutil import delete, insert


class Time(Processor):

    def __init__(self):
        super().__init__(name='time')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        h = string_file(get_abs_path('japanese/data/time/hour.tsv'))
        m = string_file(get_abs_path('japanese/data/time/minute.tsv'))
        s = string_file(get_abs_path('japanese/data/time/second.tsv'))
        noon = string_file(get_abs_path('japanese/data/time/noon.tsv'))

        colon = delete(':') | delete('：')
        h_noon = (insert('hour: "') + h + insert('" noon: "') + noon +
                  insert('"'))
        h = insert('hour: "') + h + insert('" ')
        m = insert('minute: "') + m + insert('"')
        s = insert(' second: "') + s + insert('"')
        noon = insert(' noon: "') + noon + insert('"')

        tagger = ((h + colon + m +
                   (colon + s).ques + delete(' ').ques + noon.ques)
                  | h_noon)
        tagger = self.add_tokens(tagger)

        to = (delete('-') | delete('~')) + insert(' char { value: "から" } ')
        self.tagger = tagger + (to + tagger).ques

    def build_verbalizer(self):
        noon = delete('noon: "') + self.SIGMA + delete('" ')
        hour = delete('hour: "') + self.SIGMA + delete('"')
        minute = delete(' minute: "') + self.SIGMA + delete('"')
        second = delete(' second: "') + self.SIGMA + delete('"')
        verbalizer = noon.ques + hour + minute + second.ques | noon + hour
        self.verbalizer = self.delete_tokens(verbalizer)
