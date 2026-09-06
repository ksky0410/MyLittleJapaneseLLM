from analyze_response_functions import classify_response_function


def test_classifies_response_functions_by_priority() -> None:
    assert classify_response_function("元気ですか？", "こんにちは", 3) == "greeting"
    assert classify_response_function("今日はどうでしたか？", "楽しかったです", 5) == "question_answer"
    assert classify_response_function("今日は雨でした", "なるほど", 3) == "backchannel"
    assert classify_response_function("今日は雨でした", "そうですね、いいですね", 8) == "backchannel"
    assert classify_response_function("今日は雨でした", "いや、それは違うと思う", 8) == "agreement_disagreement"
    assert classify_response_function("今日は雨でした", "またね", 3) == "closing"
    assert classify_response_function("今日は雨でした", "死んでしまたね", 16) == "topic_continuation"
    assert classify_response_function("今日は雨でした", "昨日の話だけど、駅の近くで友達に会って、いろいろ話しました", 20) == "topic_continuation"
    assert classify_response_function("今日は雨でした", "いい天気", 4) == "other"
