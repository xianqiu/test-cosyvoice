from covo2 import Covo2Wrap


class RunTest:

    cw = Covo2Wrap()

    @classmethod
    def short_text(cls):
        text = "有一种撕心裂肺的感觉，是辣椒，我加了辣椒！"
        cls.cw.load_speaker("houyi").to_speech(text)

    @classmethod
    def long_text(cls):
        cls.cw.load_speaker("meizi").load_text("test_story.txt").to_speech()

    @classmethod
    def with_instruct(cls):
        text = "在这宁静的夜晚，我们可以沿着小路慢慢走，感受微风拂面的轻柔，与自然融为一体。"
        # 支持方言：粤语/四川话/郑州话/上海话/长沙话/天津话
        cls.cw.load_speaker("meizi").with_instruct("郑州话").to_speech(text)

    @classmethod
    def cross_lingual(cls):
        text = "你昨天的 presentation がよかったので, 오늘도 좋은 피드백을 받을 거예요。"
        cls.cw.load_speaker("dachui").to_speech(text)

    @classmethod
    def cross_lingual_1(cls):
        text = "你昨天的 presentation がよかったので, 오늘도 좋은 피드백을 받을 거예요。"
        cls.cw.load_speaker("dachui").with_cross_lingual().to_speech(text)


if __name__ == '__main__':
    RunTest.short_text()
    # RunTest.with_instruct()
    RunTest.long_text()
    # RunTest.cross_lingual()
    # RunTest.cross_lingual_1()
