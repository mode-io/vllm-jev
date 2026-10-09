"""Logical context ordering with Laya's original marker and length contract."""

from .decision_template import current_template


class ContextOrderMixin:
    def _encode_state(self, state, qids, questions):
        items = super()._encode_state(state, qids, questions)
        template = current_template()
        if template.context_order is None:
            return items
        # An empty state gives exact head/option boundaries, including the SDK's
        # option truncation. Searching SEP tokens would confuse user text with
        # structural separators.
        empty = super()._encode_state("", qids, questions)
        for item, skeleton in zip(items, empty):
            prefix_length = len(skeleton["ids"]) - 1
            option_start = skeleton["markers"][0]
            if item["ids"][:prefix_length] != skeleton["ids"][:prefix_length]:
                raise ValueError(
                    "Laya context boundaries differ from its native encoding"
                )
            blocks = {
                "instructions": item["ids"][1:option_start],
                "criteria": item["ids"][option_start:prefix_length],
                "state": item["ids"][prefix_length:],
            }
            offset = 1 + sum(
                len(blocks[field])
                for field in template.context_order[
                    : template.context_order.index("criteria")
                ]
            )
            item["ids"] = item["ids"][:1] + template.arrange(blocks)
            item["markers"] = [offset + p - option_start for p in item["markers"]]
        return items
