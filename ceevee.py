def get_topics_in_scope(corr_df):
    topics_in_scope = sorted(list(set(corr_df["topic_id"])))
    return topics_in_scope


def get_corr_df_channels_in_scope(corr_df, topics_df):
    topic2channel = {}
    for topic_id, channel in zip(topics_df["id"], topics_df["channel"]):
        topic2channel[topic_id] = channel
    corr_df["channel"] = [topic2channel[x] for x in corr_df["topic_id"]]
    channels = sorted(list(set(corr_df["channel"])))
    return corr_df, channels


def get_source_nonsource_topics(corr_df, topics_df):
    t2category = {}
    for topic_id, category in zip(topics_df["id"], topics_df["category"]):
        t2category[topic_id] = category

    source_topics = sorted([topic_id for topic_id in corr_df["topic_id"] if t2category[topic_id] == "source"])
    nonsource_topics = sorted([topic_id for topic_id in corr_df["topic_id"] if t2category[topic_id] != "source"])
    return source_topics, nonsource_topics
