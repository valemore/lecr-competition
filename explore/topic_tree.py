class TopicTree:
    def __init__(self, topics_df):
        self.t2parent = {}
        self.t2title = {}
        self.t2description = {}
        self.t2level = {}
        self.t2language = {}
        self.t2has_content = {}
        for topic_id, parent_id, title, description, level, language, has_content in zip(topics_df["id"],
                                                                                         topics_df["parent"],
                                                                                         topics_df["title"],
                                                                                         topics_df["description"],
                                                                                         topics_df["level"],
                                                                                         topics_df["language"],
                                                                                         topics_df["has_content"]):
            self.t2parent[topic_id] = parent_id
            self.t2title[topic_id] = title
            self.t2description[topic_id] = description
            self.t2level[topic_id] = level
            self.t2language[topic_id] = language
            self.t2has_content[topic_id] = has_content


        self.t2node = {}
        for topic_id in topics_df["id"]:
            self.add_node(topic_id)

        self.root_topics = [topic_id for topic_id, node in self.t2node.items() if node.level == 0]
        self.root_nodes = [self.t2node[t] for t in self.root_topics]

    def add_node(self, topic_id):
        if topic_id not in self.t2node:
            node = TopicNode(topic_id, self)
            parent_id = self.t2parent[topic_id]
            if parent_id != "":
                if parent_id not in self.t2node:
                    self.add_node(parent_id)
                parent_node = self.t2node[parent_id]
                node.parent = parent_node
                parent_node.children.append(node)
            self.t2node[topic_id] = node

    def __repr__(self):
        return f"<Topic tree containing {len(self.t2node)} topic nodes.>"

    def __getitem__(self, item):
        return self.t2node[item]


class TopicNode:
    def __init__(self, topic_id, topic_tree):
        self.topic_id = topic_id
        self.topic_tree = topic_tree
        self.parent = None
        self.children = []
        self.title = self.topic_tree.t2title[topic_id]
        self.description = self.topic_tree.t2description[topic_id]
        self.level = self.topic_tree.t2level[topic_id]
        self.language = self.topic_tree.t2language[topic_id]
        self.has_content = self.topic_tree.t2has_content[topic_id]

    def __repr__(self):
        parent_info = f"with parent {self.parent.topic_id}" if self.parent is not None else "at root"
        return f"<Topic {self.topic_id} {parent_info} and {len(self.children)} children."

    def is_leaf(self):
        return not self.children

    def height(self):
        if self.is_leaf():
            return 0
        return 1 + max(c.height() for c in self.children)

    def num_siblings(self):
        if self.parent:
            return len(self.parent.children) - 1
        return 0

    def num_ancestors_with_content(self):
        n = 0
        node = self.parent
        if not node:
            return 0
        while node is not None:
            n += int(node.has_content)
            node = node.parent
        return n
