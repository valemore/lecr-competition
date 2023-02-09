from typing import Dict

import pandas as pd


def build_text(title: str, description: str, text: str) -> str:
    """
    Builds text representation for a content item.
    :param title: content title
    :param description: content description
    :param text: content text
    :return: content item's text representation
    """
    full_text = ""
    if not pd.isnull(title):
        full_text += title + ". "
    if not pd.isnull(description):
        full_text += description + ". "
    if not pd.isnull(text):
        full_text += text + ". "
    return full_text


def get_content2text(content_df: pd.DataFrame) -> Dict[str, str]:
    """
    Get dictionary mapping content ids to their text representations.
    :param content_df: DataFrame from content.csv
    :return: dictionary mapping content ids to their text representations
    """
    content2text = {}
    for content_id, title, description, text in zip(content_df["id"], content_df["title"], content_df["description"], content_df["text"]):
        content2text[content_id] = build_text(title, description, text)
    return content2text
