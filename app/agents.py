import os

from pathlib import Path
from dotenv import load_dotenv

import autogen


# =====================================
# LOAD ENVIRONMENT
# =====================================
BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

load_dotenv(
    BASE_DIR / ".env"
)


# =====================================
# MODEL CONFIG
# =====================================
llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv(
        "OPENAI_API_KEY"
    )
}


# =====================================
# REFLECTION MESSAGE
# =====================================
def reflection_message(
    recipient,
    messages,
    sender,
    config
):

    latest_content = (
        recipient
        .chat_messages_for_summary(
            sender
        )[-1]["content"]
    )

    return f"""
Review the following content.

{latest_content}
"""


# =====================================
# MAIN WORKFLOW
# =====================================
def run_blog_workflow(
    task,
    enable_nested_reviews=True
):

    reviewer_outputs = {}

    # =================================
    # WRITER
    # =================================
    writer = (
        autogen.AssistantAgent(
            name="Writer",

            system_message=
            (
                "You are a writer. "
                "You write engaging "
                "and concise blogposts "
                "(with title) on "
                "given topics. "
                "You must polish "
                "your writing based "
                "on feedback you receive "
                "and give a refined "
                "version. "
                "Only return your "
                "final work without "
                "additional comments."
            ),

            llm_config=
            llm_config,
        )
    )

    # =================================
    # CRITIC
    # =================================
    critic = (
        autogen.AssistantAgent(
            name="Critic",

            is_termination_msg=
            lambda x:
            x.get(
                "content",
                ""
            ).find(
                "TERMINATE"
            ) >= 0,

            llm_config=
            llm_config,

            system_message=
            (
                "You are a critic. "
                "You review the work "
                "of the writer and "
                "provide constructive "
                "feedback to improve "
                "the quality of the content."
            ),
        )
    )

    # =================================
    # NESTED REVIEWERS
    # =================================
    if enable_nested_reviews:

        seo_reviewer = (
            autogen.AssistantAgent(
                name="SEOReviewer",

                llm_config=
                llm_config,

                system_message=
                (
                    "You are an SEO reviewer, "
                    "known for optimizing "
                    "content for search engines. "
                    "Make suggestions concise "
                    "(max 3 bullet points). "
                    "Begin by stating your role."
                ),
            )
        )

        legal_reviewer = (
            autogen.AssistantAgent(
                name="LegalReviewer",

                llm_config=
                llm_config,

                system_message=
                (
                    "You are a legal reviewer, "
                    "known for ensuring content "
                    "is legally compliant. "
                    "Make suggestions concise "
                    "(max 3 bullet points). "
                    "Begin by stating your role."
                ),
            )
        )

        ethics_reviewer = (
            autogen.AssistantAgent(
                name="EthicsReviewer",

                llm_config=
                llm_config,

                system_message=
                (
                    "You are an ethics reviewer, "
                    "known for ensuring content "
                    "is ethically sound. "
                    "Make suggestions concise "
                    "(max 3 bullet points). "
                    "Begin by stating your role."
                ),
            )
        )

        meta_reviewer = (
            autogen.AssistantAgent(
                name="MetaReviewer",

                llm_config=
                llm_config,

                system_message=
                (
                    "You are a meta reviewer. "
                    "Aggregate and review "
                    "the work of other reviewers "
                    "and give final suggestions."
                ),
            )
        )

        review_chats = [

            {
                "recipient":
                seo_reviewer,

                "message":
                reflection_message,

                "summary_method":
                "last_msg",

                "max_turns":
                1,
            },

            {
                "recipient":
                legal_reviewer,

                "message":
                reflection_message,

                "summary_method":
                "last_msg",

                "max_turns":
                1,
            },

            {
                "recipient":
                ethics_reviewer,

                "message":
                reflection_message,

                "summary_method":
                "last_msg",

                "max_turns":
                1,
            },

            {
                "recipient":
                meta_reviewer,

                "message":
                (
                    "Aggregate feedback "
                    "from all reviewers "
                    "and provide final "
                    "suggestions."
                ),

                "summary_method":
                "last_msg",

                "max_turns":
                1,
            },
        ]

        critic.register_nested_chats(
            review_chats,
            trigger=writer
        )

    # =================================
    # RUN WORKFLOW
    # =================================
    result = (
        critic.initiate_chat(
            recipient=writer,

            message=task,

            max_turns=2,

            summary_method=
            "last_msg",
        )
    )

    # =================================
    # CAPTURE REVIEWER OUTPUTS
    # =================================
    if enable_nested_reviews:

        reviewers = [
            seo_reviewer,
            legal_reviewer,
            ethics_reviewer,
            meta_reviewer
        ]

        for reviewer in reviewers:

            history = (
                reviewer.chat_messages
            )

            last_message = ""

            for msgs in history.values():

                if len(msgs) > 0:

                    last_message = (
                        msgs[-1]
                        .get(
                            "content",
                            ""
                        )
                    )

            reviewer_outputs[
                reviewer.name
            ] = last_message

    return {
        "result": result,
        "reviewers": reviewer_outputs
    }