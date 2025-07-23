import os
import requests
from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

# === ENVIRONMENT VARIABLES / CONFIG ===
IEXPENSE_API_BASE = ""
API_KEY = "2A36F1D2-9757-4ED3-AABC-EB096B1850EE"   # Place your actual key here or use os.environ

HEADERS = {
    "API_KEY": API_KEY,
    "Content-Type": "application/json"
}

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], add_messages]

class IExpenseAgent:
    """
    EXL iExpense LLM-powered conversational agent.
    - Handles listing, counting, approval/rejection for manager workflows.
    - Uses LLM prompt for ALL reference resolution (no rules in code!).
    - Connects to real EXL APIs with dynamic user, paging, and robust error handling.
    """

    def __init__(self, lanid: str):
        """
        Args:
            lanid (str): Authenticated user's LAN ID
        """
        self.lanid = lanid.strip()
        self.model = init_chat_model(
            "azure_openai:gpt-4o",
            azure_deployment="gpt-4o"
        )
        self.memory = []
        self.tools = self.make_tools()
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.graph = self._build_graph()

    # ========== REAL API CALLS WITH ERROR HANDLING ==========

    def get_notification_count(self) -> str:
        """
        Fetch the count of pending iExpense notifications for the current user via API.
        """
        url = IEXPENSE_API_BASE + "NotificationCount"
        body = { "User_ID": self.lanid }
        try:
            resp = requests.post(url, json=body, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            count = data.get("Total_Records") or 0
            if not count or count == 0:
                return "You have no pending iExpense notifications for approval."
            return f"You have {count} pending iExpense notification(s) for approval."
        except Exception as e:
            return f"Sorry, failed to fetch notification count due to a system error: {str(e)}"

    def list_notifications(self) -> str:
        """
        Fetch all PENDING iExpense notifications for this user from the API (first page only for now).
        """
        url = IEXPENSE_API_BASE + "NotificationExpense"
        body = {
            "User_ID": self.lanid,
            "P_NUM": 1,     # First page
            "R_PAGE": 20    # Number per page (adjust as needed)
        }
        try:
            resp = requests.post(url, json=body, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                return "There are no pending iExpense notifications."
            msg = "Here are your pending iExpense notifications:\n"
            for i, exp in enumerate(data, 1):
                msg += (
                    f"{i}. Notification ID: {exp.get('NOTIFICATION_ID')}, "
                    f"Report: {exp.get('Report_Number')}, "
                    f"From: {exp.get('FROM_USER')}, "
                    f"Amount: {exp.get('Amount', 'N/A')} INR, "
                    f"Subject: {exp.get('SUBJECT')}\n"
                )
            return msg.strip()
        except Exception as e:
            return f"Sorry, failed to fetch pending notifications due to a system error: {str(e)}"

    def approve_notification(self, notification_id: str) -> str:
        """
        Approve a specific iExpense notification by its Notification ID via API.
        """
        url = IEXPENSE_API_BASE + "PutExpenseApproveReject"
        body = {
            "USER_IDS": self.lanid,
            "I_NOTIFICATION_ID": notification_id,
            "I_ACTION": "APPROVE",
            "I_ACTION_COMMENT": "Approved via assistant"
        }
        try:
            resp = requests.post(url, json=body, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data or "response" not in data:
                return "Failed to process your approval. Try again later."
            res = data["response"]
            if res.get("isSuccess") and res["data"].get("P_STATUS", "").upper() == "SUCCESS":
                return f"Notification {notification_id} has been APPROVED successfully."
            else:
                err = res["data"].get("P_ERROR_MSG", "Approval failed.")
                return f"Approval for notification {notification_id} failed: {err}"
        except Exception as e:
            return f"Sorry, failed to approve notification {notification_id} due to a system error: {str(e)}"

    def reject_notification(self, notification_id: str) -> str:
        """
        Reject a specific iExpense notification by its Notification ID via API.
        """
        url = IEXPENSE_API_BASE + "PutExpenseApproveReject"
        body = {
            "USER_IDS": self.lanid,
            "I_NOTIFICATION_ID": notification_id,
            "I_ACTION": "REJECT",
            "I_ACTION_COMMENT": "Rejected via assistant"
        }
        try:
            resp = requests.post(url, json=body, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data or "response" not in data:
                return "Failed to process your rejection. Try again later."
            res = data["response"]
            if res.get("isSuccess") and res["data"].get("P_STATUS", "").upper() == "SUCCESS":
                return f"Notification {notification_id} has been REJECTED successfully."
            else:
                err = res["data"].get("P_ERROR_MSG", "Rejection failed.")
                return f"Rejection for notification {notification_id} failed: {err}"
        except Exception as e:
            return f"Sorry, failed to reject notification {notification_id} due to a system error: {str(e)}"

    # ========== LLM TOOL BINDINGS (NO RULES, FULL CONTEXT) ==========
    def make_tools(self):
        def count_tool():
            """Return the count of pending iExpense notifications for the current user."""
            return self.get_notification_count()
        count_tool.__doc__ = self.get_notification_count.__doc__

        def list_tool():
            """List all pending iExpense notifications for the current user."""
            return self.list_notifications()
        list_tool.__doc__ = self.list_notifications.__doc__

        def approve_tool(notification_id: str):
            """
            Approve a pending iExpense notification by its unique Notification ID.
            """
            return self.approve_notification(notification_id)
        approve_tool.__doc__ = self.approve_notification.__doc__

        def reject_tool(notification_id: str):
            """
            Reject a pending iExpense notification by its unique Notification ID.
            """
            return self.reject_notification(notification_id)
        reject_tool.__doc__ = self.reject_notification.__doc__

        return [count_tool, list_tool, approve_tool, reject_tool]

    @staticmethod
    def system_prompt(context: Optional[str] = None) -> str:
        """
        Ultra-detailed system prompt to maximize LLM context handling.
        """
        prompt = (
            "You are EXL's enterprise iExpense Assistant Agent. "
            "You help the user with their iExpense notifications using ONLY the available tools.\n"
            "Instructions:\n"
            "- For any pending expenses request (count, list, details), always use the relevant tool.\n"
            "- When the user requests to approve/reject, ALWAYS resolve the correct Notification ID from the conversation context, numbered lists, previous tool outputs, or user messages, and pass ONLY the ID to the backend tool. Never hardcode or guess in code!\n"
            "- If user refers ambiguously (e.g., 'approve the second one', 'approve Kunal's'), try to infer the Notification ID from list/context; if still unclear, politely ask for clarification with a list.\n"
            "- Accept any natural language ('approve 47052068', 'please approve the last', 'reject Kunal’s', etc.) but always use only Notification ID for backend calls.\n"
            "- For anything not related to iExpense, reply: 'Sorry, I can only help with iExpense approvals and requests.'\n"
            "- Always return clear, professional, actionable responses. Be human-like, robust, and context-aware.\n"
            "- If a tool/API call fails, inform the user professionally and suggest a retry if needed.\n"
            + (f"\nRecent context:\n{context}" if context else "")
        )
        return prompt

    def agent_node(self, state: State):
        """
        LangGraph node to run the LLM with full context and system prompt.
        """
        notifications_context = self.list_notifications()
        msgs = [SystemMessage(content=self.system_prompt(notifications_context))] + state["messages"]
        response = self.model_with_tools.invoke(msgs)
        return {"messages": [response]}

    def _build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("agent", self.agent_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_conditional_edges(
            "agent",
            tools_condition,
            path_map={"tools": "tools", "end": END, "__end__": END}
        )
        graph_builder.add_edge("tools", "agent")
        graph_builder.set_entry_point("agent")
        return graph_builder.compile()

    def run_chat(self, user_messages: List[str]) -> str:
        """
        Run a chat session (CLI or backend). Keeps full memory for robust context.
        Args:
            user_messages (List[str]): List of user input strings for this turn.
        Returns:
            str: Assistant's latest response.
        """
        for user_input in user_messages:
            self.memory.append(HumanMessage(content=user_input))
            response_text = ""
            for msg in self.graph.stream({"messages": self.memory}):
                if isinstance(msg, dict):
                    ai_msg = msg.get("agent", {}).get("messages", [None])[0]
                    if ai_msg and hasattr(ai_msg, "content") and ai_msg.content:
                        response_text = ai_msg.content
                elif hasattr(msg, "content") and msg.content:
                    response_text = msg.content
            self.memory.append(AIMessage(content=response_text))
        return response_text

# ========== CLI DEMO ==========
def main():
    print("=== EXL iExpense LangGraph Agent (LLM-driven, Real API) ===")
    lanid = input("Enter your LAN ID (SSO simulation): ").strip()
    agent = IExpenseAgent(lanid=lanid)
    print("Tip: Try: 'approve the second one', 'reject it', 'list', 'show pending', 'approve Kunal's', etc.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        response = agent.run_chat([user_input])
        print("Assistant:", response)

if __name__ == "__main__":
    main()
