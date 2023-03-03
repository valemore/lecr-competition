import itertools
from typing import List

from utils import is_ordered


def post_process(probs, topic_ids: List[str], content_ids: List[str]):
    """
    Post-process predicted probabilities. For every topic, we modify the probability of the most likely content to 1.0.
    :param probs: numpy array of shape (num_topics,) indicating probability that content is is assigned to topic
    :param topic_ids:
    :param content_ids:
    :return: post-processed probs array of shape (num_topics,) with probabilities of most likely contents set to 1.0
    """
    assert is_ordered(topic_ids)

    def argmax_helper(idxs):
        """Finds the index among topic indices IDXS indicating maximum probability in probs among relevant contents."""
        max_prob = -1.0
        max_i = -1
        for i in idxs:
            prob = probs[i]
            if prob > max_prob:
                max_prob = prob
                max_i = i
        return max_i


    for topic_id, topic_idxs in itertools.groupby(range(len(topic_ids)), lambda i: topic_ids[i]):
        max_i = argmax_helper(topic_idxs)
        probs[max_i] = 1.0

    return probs
