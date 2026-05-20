import streamlit as st

from agents import (
    run_blog_workflow
)


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title=
    "AI Agent Playground — Reflection",

    layout="wide"
)


# =====================================
# HEADER
# =====================================
st.title(
    "AI Agent Playground"
)

st.caption(
    "Reflection and nested multi-agent review using AutoGen and OpenAI"
)

st.divider()


left, right = st.columns(
    [1, 2]
)


# =====================================
# LEFT SIDE
# =====================================
with left:

    st.subheader(
        "Blog Writing Task"
    )

    task = st.text_area(
        "Task",

        value="""
Write a concise but engaging
blogpost about DeepLearning.AI.

Make sure the blogpost
is within 100 words.
""",

        height=180
    )

    enable_nested_reviews = (
        st.toggle(
            "Enable nested reviewers",
            value=True
        )
    )

    run_button = (
        st.button(
            "Generate Blogpost",
            use_container_width=True
        )
    )

    st.divider()

    st.markdown("""
### What this demonstrates

- Reflection loops  
- Critic feedback  
- Nested chats  
- Specialized reviewers  
- Content refinement  
- Multi-agent orchestration  
""")


# =====================================
# RIGHT SIDE
# =====================================
with right:

    st.subheader(
        "Workflow Output"
    )

    if run_button:

        with st.spinner(
            "Agents are collaborating..."
        ):

            try:

                workflow = (
                    run_blog_workflow(
                        task=task,
                        enable_nested_reviews=
                        enable_nested_reviews
                    )
                )

                result = workflow[
                    "result"
                ]

                reviewers = workflow[
                    "reviewers"
                ]

                st.success(
                    "Workflow completed"
                )

                st.divider()

                # ======================
                # FINAL BLOGPOST
                # ======================
                st.subheader(
                    "Final Blogpost"
                )

                st.markdown(
                    result.summary
                )

                st.divider()

                # ======================
                # REVIEWER OUTPUTS
                # ======================
                if reviewers:

                    st.subheader(
                        "Reviewer Contributions"
                    )

                    st.caption(
                        "How each specialized agent improved the content"
                    )

                    for (
                        reviewer_name,
                        feedback
                    ) in reviewers.items():

                        with st.expander(
                            reviewer_name,
                            expanded=False
                        ):

                            st.write(
                                feedback
                            )

                st.divider()

                # ======================
                # FULL CONVERSATION
                # ======================
                with st.expander(
                    "View Agent Conversation"
                ):

                    for msg in (
                        result.chat_history
                    ):

                        role = msg.get(
                            "name",
                            msg.get(
                                "role",
                                "Unknown"
                            )
                        )

                        content = (
                            msg.get(
                                "content",
                                ""
                            )
                        )

                        if not content:
                            continue

                        st.markdown(
                            f"### {role}"
                        )

                        st.write(
                            content
                        )

                        st.divider()

            except Exception as e:

                st.error(
                    str(e)
                )