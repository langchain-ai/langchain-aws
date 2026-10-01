# helpers

The notebooks import these so their cells stay short. Open them: the prompts and the
span reader are the interesting part of this sample.

| File | What it holds | Framework specific? |
|---|---|---|
| `agentcore_evals.py` | `Trajectory`, span reader, `Check`, `report` | no, reads OpenTelemetry attributes |
| `agentcore_deploy.py` | IAM, packaging, S3, runtime create and delete | no |
| `agentcore_loop.py` | insights, recommendations, configuration bundles | no |
| `edgar_dataset.py` | derives `dataset.json` from tickers, so the answer key is generated | no, it is a data script |
| `research_agent.py` | the agent, and the answer key computed from `dataset.json` | yes, LangChain Deep Agents |
| `research_checks.py` | the checks and the traffic scenarios | yes, this agent |
| `runtime_entrypoint.py` | what gets deployed to AgentCore Runtime | yes, LangChain Deep Agents |
| `negative_control.py` | stand-in tools that force each check to fail | yes, this agent |

`agentcore_evals.py` imports no agent framework. It recognizes tool spans from three
instrumentation conventions, so the same checks work for a Strands or CrewAI agent.
That module is the part that arguably belongs in an SDK, and it is kept separate here
to make that boundary visible.
