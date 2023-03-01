# Topic representation
from typing import Dict

import pandas as pd


def build_text(topic_id: str, topic2title: Dict[str, str], topic2description: Dict[str, str], topic2parent: Dict[str, str]) -> str:
    """
    Builds text representation for a topic.
    :param topic_id: topic id
    :param topic2title: dict mapping topic ids to titles
    :param topic2description: dict mapping topic ids to descriptions
    :param topic2parent: dict mapping topic id to parent topic id
    :return: topic's text representation
    """
    text = ""
    description = topic2description[topic_id]
    while topic_id:
        text += topic2title[topic_id] + ". "
        topic_id = topic2parent[topic_id]
    if description:
        text += description
    return text


def get_topic2text(topics_df: pd.DataFrame) -> Dict[str, str]:
    """
    Get dictionary mapping topic ids to their text representations.
    :param topics_df: DataFrame from topics.csv
    :return: dictionary mapping topic ids to their text representations
    """
    topic2title = {}
    topic2description = {}
    topic2parent = {}
    for topic_id, title, description, parent_id in zip(topics_df["id"], topics_df["title"], topics_df["description"], topics_df["parent"]):
        topic2title[topic_id] = title
        topic2description[topic_id] = description
        topic2parent[topic_id] = parent_id

    topic2text = {}
    for topic_id in topics_df["id"]:
        topic2text[topic_id] = build_text(topic_id, topic2title, topic2description, topic2parent)

    return topic2text
