from langchain.tools import tool
from langchain_community.tools import ShellTool

class CyberActions:
    @tool
    def isolate_host(host_id: str):
        """Isolate compromised host from network"""
        return ShellTool().run(f"iptables -A INPUT -s {host_id} -j DROP")

    @tool
    def collect_forensics(host_id: str):
        """Initiate forensic data collection"""
        return {
            "memory_dump": ShellTool().run(f"avml {host_id}-memory.raw"),
            "disk_image": ShellTool().run(f"dd if=/dev/sda of={host_id}-disk.img")
        }

class ResponseOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-0125-preview")
        self.tools = [CyberActions.isolate_host, 
                     CyberActions.collect_forensics]
        
    def build_response_plan(self, threat_data):
        prompt = CustomPromptBuilder.build_response_prompt(threat_data)
        chain = prompt | self.llm.bind_tools(self.tools)
        return chain.invoke({"threat_data": threat_data})
