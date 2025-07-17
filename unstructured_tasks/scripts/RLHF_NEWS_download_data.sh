#! /bin/bash
huggingface-cli download clockwork7/hh_golden_ifg_annotated  --repo-type dataset --local-dir=data/hh_annotated
huggingface-cli download clockwork7/reddit_news_articles_comments  --repo-type dataset --local-dir=data/reddit_comments