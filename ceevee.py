# Helpers for cross validation of experiments and generating splits

def get_topics_in_corr(corr_df):
    """Returns sorted list of topic ids in CORR_DF."""
    topics_in_scope = sorted(list(set(corr_df["topic_id"])))
    return topics_in_scope


def get_corr_df_channels_in_corr(corr_df, topics_df):
    """Returns tuple of CORR_DF with added 'channel' column and sorted list of channels in CORR_DF."""
    topic2channel = {}
    for topic_id, channel in zip(topics_df["id"], topics_df["channel"]):
        topic2channel[topic_id] = channel
    corr_df["channel"] = [topic2channel[x] for x in corr_df["topic_id"]]
    channels = sorted(list(set(corr_df["channel"])))
    return corr_df, channels


def get_source_nonsource_topics(corr_df, topics_df):
    """Returns tuple of sorted list of topics with the source category, and sorted list of non-source topics in CORR_DF."""
    t2category = {}
    for topic_id, category in zip(topics_df["id"], topics_df["category"]):
        t2category[topic_id] = category

    source_topics = sorted([topic_id for topic_id in corr_df["topic_id"] if t2category[topic_id] == "source"])
    nonsource_topics = sorted([topic_id for topic_id in corr_df["topic_id"] if t2category[topic_id] != "source"])
    return source_topics, nonsource_topics


def get_corr_df_source_nonsource_channels(corr_df, topics_df):
    """
    Returns tuple of CORR_DF with added 'channel' column, sorted list of channel ids with the source category, sorted
    list of non-source channel ids.
    """
    corr_df, channels = get_corr_df_channels_in_corr(corr_df, topics_df)

    channel2category = {}
    for channel in channels:
        channel2category[channel] = topics_df.loc[(topics_df["channel"] == channel) & (topics_df["parent"].isna()), "category"].item()

    source_channels = sorted([channel_id for channel_id in channels if channel2category[channel_id] == "source"])
    nonsource_channels = sorted([channeld_id for channeld_id in channels if channel2category[channeld_id] != "source"])
    return corr_df, source_channels, nonsource_channels
