import pandas as pd


def build_text(topic_id, topic2title, topic2description, topic2parent):
    text = ""
    description = topic2description[topic_id]
    while topic_id is not None:
        text += topic2title[topic_id] + ". "
        topic_id = topic2parent[topic_id]
    if description:
        text += description
    return text


def get_topic2text(topics_df):
    topic2title = {}
    topic2description = {}
    topic2parent = {}
    for topic_id, title, description, parent_id in zip(topics_df["id"], topics_df["title"], topics_df["description"], topics_df["parent"]):
        if not pd.isnull(title):
            topic2title[topic_id] = title
        else:
            topic2title[topic_id] = ""
        if not pd.isnull(description):
            topic2description[topic_id] = description
        else:
            topic2description[topic_id] = ""
        if not pd.isnull(parent_id):
            topic2parent[topic_id] = parent_id
        else:
            topic2parent[topic_id] = None

    topic2text = {}
    for topic_id in topics_df["id"]:
        topic2text[topic_id] = build_text(topic_id, topic2title, topic2description, topic2parent)

    return topic2text
