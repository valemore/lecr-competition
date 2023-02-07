import logging
import pandas as pd


def build_text(title, description, text):
    full_text = ""
    if not pd.isnull(title):
        full_text += title + ". "
    if not pd.isnull(description):
        full_text += description + ". "
    if not pd.isnull(text):
        full_text += text + ". "
    return full_text


def get_content2text(content_df):
    content2text = {}
    for content_id, title, description, text in zip(content_df["id"], content_df["title"], content_df["description"], content_df["text"]):
        content2text[content_id] = build_text(title, description, text)
    return content2text
