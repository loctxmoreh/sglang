"""Chat template example.

Registers a new conversation template `moreh-chat`. After the plugin
loads, sglang's chat-template machinery will recognise the name; you can
select it on the CLI via `--chat-template moreh-chat`.
"""

from __future__ import annotations

from moreh_plugin import logger


def install() -> None:
    from sglang.srt.parser.conversation import (
        Conversation,
        SeparatorStyle,
        register_conv_template,
    )

    logger.info("[moreh-plugin] registering conv template 'moreh-chat'")

    register_conv_template(
        Conversation(
            name="moreh-chat",
            system_template="<|sys|>{system_message}\n",
            system_message="You are a helpful assistant running on ROCm (demo).",
            roles=("<|user|>", "<|assistant|>"),
            sep_style=SeparatorStyle.CHATML,
            sep="\n",
            stop_str=["<|user|>", "<|end|>"],
        ),
        # Use override=True so re-running install() (e.g. across pytest
        # cases that reload plugins) doesn't trip the duplicate-name
        # assertion inside register_conv_template().
        override=True,
    )
