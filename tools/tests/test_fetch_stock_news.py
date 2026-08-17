"""Tests for the keyless Eastmoney news fetcher."""

from tools.fetch_stock_news import clean_markup, normalize_news, search_name_for


def test_clean_markup_removes_highlight_tags() -> None:
    assert clean_markup("<em>北方</em>华创 发布公告") == "北方华创 发布公告"


def test_normalize_news_filters_and_shapes_articles() -> None:
    articles = [
        {
            "title": "<em>北方华创</em>订单创新高",
            "content": "公司在手订单超过 820 亿元",
            "date": "2026-08-17 08:53:00",
            "mediaName": "证券时报",
            "url": "https://example.com/1",
        },
        {
            "title": "完全无关的宏观新闻",
            "content": "消费数据发布",
            "date": "2026-08-17 08:00:00",
            "mediaName": "日报",
            "url": "https://example.com/2",
        },
    ]

    news = normalize_news(articles, keywords={"北方华创", "002371"}, limit=6)

    assert news == [
        {
            "title": "北方华创订单创新高",
            "source": "证券时报",
            "time": "2026-08-17",
            "url": "https://example.com/1",
        }
    ]


def test_normalize_news_deduplicates_and_limits() -> None:
    articles = [
        {"title": "北方华创新闻", "content": "北方华创", "date": "2026-08-17",
         "mediaName": "来源", "url": "u1"},
        {"title": "北方华创新闻", "content": "重复", "date": "2026-08-16",
         "mediaName": "来源", "url": "u2"},
        {"title": "北方华创另一条", "content": "北方华创", "date": "2026-08-15",
         "mediaName": "来源", "url": "u3"},
    ]

    news = normalize_news(articles, keywords={"北方华创"}, limit=2)

    assert [item["title"] for item in news] == ["北方华创新闻", "北方华创另一条"]


def test_search_name_for_strips_trade_status_prefix() -> None:
    assert search_name_for("603893.SH", "XD瑞芯微") == "瑞芯微"
    assert search_name_for("600519.SH", "贵州茅台") == "贵州茅台"
    assert search_name_for("600519.SH", "") == "600519"
