# Dependency watch on Amazon Bedrock AgentCore Runtime

This folder hosts the graph from `agentcore_web_search_supply_chain_watch.ipynb` on
[Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html).
`main.py` is the same graph, research branches and evidence checks as the notebook, wrapped in a
`BedrockAgentCoreApp` entrypoint.

| File | Purpose |
| --- | --- |
| `main.py` | The graph, served as the Runtime entrypoint |
| `pyproject.toml` | Python dependencies for the Runtime build |
| `invoke-gateway-policy.json` | Grants the Runtime execution role `bedrock-agentcore:InvokeGateway` on your gateway only |

## Prerequisites

- Run `agentcore_web_search_supply_chain_watch.ipynb` through Step 1 first. It creates the gateway and
  the `security-sources` web search target this agent searches through. Note the gateway ID it prints.
- The [AgentCore CLI](https://github.com/aws/agentcore-cli), Node.js 20 or later, and
  [uv](https://docs.astral.sh/uv/getting-started/installation/). These steps were tested with
  CLI version 0.31.1; check the CLI's README if a command has changed.
- AWS credentials for the same account and Region as the gateway.

## Deploy

1. Create a LangGraph project:

   ```bash
   agentcore create --name SupplyChainWatch --project-name SupplyChainWatch \
     --framework LangChain_LangGraph --model-provider Bedrock --memory none \
     --build CodeZip --protocol HTTP
   ```

2. Copy the three files in this folder into `SupplyChainWatch/app/SupplyChainWatch/`, replacing the
   template's `main.py` and `pyproject.toml`.

3. In `invoke-gateway-policy.json`, replace `<region>`, `<account-id>` and `<gateway-id>`.

4. In `SupplyChainWatch/agentcore/agentcore.json`, add the gateway settings and the policy to the
   runtime entry:

   ```json
   "envVars": [
     {"name": "WEB_SEARCH_GATEWAY_ID", "value": "<gateway-id>"},
     {"name": "WEB_SEARCH_TARGET_NAME", "value": "security-sources"}
   ],
   "additionalPolicies": ["invoke-gateway-policy.json"]
   ```

5. Deploy and invoke:

   ```bash
   cd SupplyChainWatch
   agentcore deploy -y
   agentcore invoke --session-id "watch-$(uuidgen | tr -d -)" --json "run the daily watch"
   ```

The response holds `wall_seconds`, per-branch statistics, the number of findings kept and new, and the
cited report as a markdown table.

## State between runs

`main.py` uses `InMemoryStore`, which lasts as long as the Runtime session. A second invocation with the
same `--session-id` reports nothing already reported; a new session starts with an empty store and
reports everything again. For a daily job, invoke on a schedule and replace `InMemoryStore` with a
durable store with exact key lookups, such as `DynamoDBStore` from `langgraph-checkpoint-aws`.

## Clean up

`agentcore remove all`, then `agentcore deploy -y`, deletes the Runtime resources this project created.
The gateway and target belong to the notebook; delete them from its Clean up cell.
