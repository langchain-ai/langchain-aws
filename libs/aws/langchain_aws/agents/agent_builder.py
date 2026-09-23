'''

The create_agent method of agents/base.py has a large number of parameters. I want
to offer an easier way to create agents without having to enter in
so many parameters or even offer built-in features quicker (confirmations, human-as-a-tool).

I want to get comments on 2 different possible solutions.

First is the builder pattern which is not as common in Python but still used.

Second is config pattern which is a more common pattern.

'''


# builder

agent = (BedrockAgentBuilder("my-agent", "arn:aws:iam::xyz123:role/agent")
    .with_tools([tool1, tool2])
    .with_instructions("You are a helpful financial agent who answers finance related questions using "
                       "the provided tools.")
    .with_idle_timeout(3600)
    .with_tracing()
    .with_human_input()
    .with_confirmations()
    .build())



# config
agent = create_bedrock_agent(
    config=BedrockAgentConfig(
        name="my-agent",
        role_arn="arn:aws:iam::123:role/agent",
        model="anthropic.claude-3",
        instruction="Help users with tasks",
        mixins=[HumanInTheLoopMixin, HumanConfirmationMixin, TracingMixin]
    )
)


##Thoughts on either? Or something else?