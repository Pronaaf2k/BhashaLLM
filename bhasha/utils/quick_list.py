from agentmail import AgentMail
import os

api_key = os.environ.get("AGENTMAIL_API_KEY")
client = AgentMail(api_key=api_key)

print(client.inboxes.list())
