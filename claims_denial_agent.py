# claims_denial_agent_streamlined.py
"""
Streamlined Claims Denial Agent - Optimized for Speed and Reliability

KEY OPTIMIZATIONS:
- Reduced to 6-7 iterations (was 8-10)
- Deterministic tool selection (no LLM guessing)
- Simplified prompts (less reasoning, more action)
- Optional medical necessity research
- Hardcoded workflow sequence
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json
import os
import re
from dotenv import load_dotenv

# Optional imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

load_dotenv()

# ============================================================================
# SIMPLIFIED STATE MACHINE
# ============================================================================

class WorkflowStep(Enum):
    """Linear workflow steps - no branching"""
    STEP_1_EXTRACT = 1
    STEP_2_VALIDATE_CPT = 2
    STEP_3_VALIDATE_DIAG = 3
    STEP_4_CHECK_POLICY = 4
    STEP_5_MEDICAL_NECESSITY = 5  # Optional
    STEP_6_ANALYZE = 6
    STEP_7_GENERATE_LETTER = 7
    COMPLETE = 8

class StreamlinedWorkflowState:
    """Simplified workflow state - just track current step"""
    
    def __init__(self):
        self.current_step = WorkflowStep.STEP_1_EXTRACT
        self.step_results = {}
        self.completed_steps = []
        self.skip_medical_necessity = False  # Can skip if not needed
        
    def advance(self):
        """Move to next step"""
        step_order = list(WorkflowStep)
        current_index = step_order.index(self.current_step)
        
        # Skip medical necessity if flagged
        if self.current_step == WorkflowStep.STEP_4_CHECK_POLICY and self.skip_medical_necessity:
            self.current_step = WorkflowStep.STEP_6_ANALYZE
        elif current_index < len(step_order) - 1:
            self.current_step = step_order[current_index + 1]
        
        self.completed_steps.append(self.current_step)
    
    def get_current_tool(self) -> str:
        """Return exact tool name for current step - NO GUESSING"""
        tool_map = {
            WorkflowStep.STEP_1_EXTRACT: "extract_claim_info",
            WorkflowStep.STEP_2_VALIDATE_CPT: "search_cpt_code",
            WorkflowStep.STEP_3_VALIDATE_DIAG: "search_diagnosis_code",
            WorkflowStep.STEP_4_CHECK_POLICY: "search_payer_policy",
            WorkflowStep.STEP_5_MEDICAL_NECESSITY: "search_medical_necessity",
            WorkflowStep.STEP_6_ANALYZE: "analyze_denial",
            WorkflowStep.STEP_7_GENERATE_LETTER: "generate_appeal_letter",
            WorkflowStep.COMPLETE: "DONE"
        }
        return tool_map.get(self.current_step, "DONE")
    
    def is_complete(self) -> bool:
        return self.current_step == WorkflowStep.COMPLETE

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_llm() -> ChatGroq:
    return st.session_state.get('agent_llm')

def get_state() -> StreamlinedWorkflowState:
    if 'workflow_state' not in st.session_state or st.session_state['workflow_state'] is None:
        st.session_state['workflow_state'] = StreamlinedWorkflowState()
    return st.session_state['workflow_state']

def truncate(text: str, max_len: int = 1000) -> str:
    """Aggressive truncation to save context"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...[{len(text)-max_len} chars truncated]"

# ============================================================================
# STREAMLINED TOOLS - Minimal Prompts, Fast Execution
# ============================================================================

@tool
def extract_claim_info(input_text: Optional[str] = None) -> str:
    """Extract claim info - STEP 1"""
    llm = get_llm()
    if not llm:
        return json.dumps({"error": "LLM unavailable"})
    
    text = st.session_state.get('raw_denial_text', '')
    if not text:
        return json.dumps({"error": "No denial text"})
    
    # MINIMAL PROMPT - Just extract, don't explain
    prompt = f"""Extract claim data from this denial letter. Return ONLY JSON, no explanation.

{text[:3000]}

JSON format:
{{"claim_number":"...","patient_name":"...","service_date":"YYYY-MM-DD","billed_amount":0.0,"cpt_codes":"...","diagnosis":"...","denial_text":"...","payer_name":"..."}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # Extract JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            parsed = json.loads(json_str)
            
            # Store immediately
            st.session_state['current_claim_data'] = parsed
            
            # Check if good enough
            critical_missing = sum(1 for f in ['denial_text', 'cpt_codes'] if parsed.get(f) in ["UNKNOWN", "", None])
            
            if critical_missing == 0:
                parsed['_status'] = 'OK'
            else:
                parsed['_status'] = 'PARTIAL'
            
            return json.dumps(parsed)
        else:
            return json.dumps({"error": "No JSON in response"})
    
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def search_cpt_code(code: Optional[str] = None) -> str:
    """Validate CPT - STEP 2"""
    if not DDGS_AVAILABLE:
        return json.dumps({"status": "SKIPPED", "reason": "Search unavailable"})
    
    if not code:
        claim_data = st.session_state.get('current_claim_data', {})
        code = claim_data.get('cpt_codes', '')
    
    if not code or code == "UNKNOWN":
        return json.dumps({"status": "SKIPPED"})
    
    # Get first code only
    first_code = code.split(',')[0].strip()
    
    try:
        with DDGS() as ddgs:
            query = f"CPT {first_code}"
            results = list(ddgs.text(query, max_results=1))
            
            if results:
                return json.dumps({
                    "code": first_code,
                    "status": "FOUND",
                    "info": results[0].get("body", "")[:200]
                })
            else:
                return json.dumps({"code": first_code, "status": "NOT_FOUND"})
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})

@tool
def search_diagnosis_code(diag: Optional[str] = None) -> str:
    """Validate diagnosis - STEP 3"""
    if not DDGS_AVAILABLE:
        return json.dumps({"status": "SKIPPED"})
    
    if not diag:
        claim_data = st.session_state.get('current_claim_data', {})
        diag = claim_data.get('diagnosis', '')
    
    if not diag or diag == "UNKNOWN":
        return json.dumps({"status": "SKIPPED"})
    
    # Extract ICD code
    code_match = re.search(r'\b[A-Z]\d{2}\.?\d{0,4}\b', diag)
    search_term = code_match.group(0) if code_match else diag
    
    try:
        with DDGS() as ddgs:
            query = f"ICD-10 {search_term}"
            results = list(ddgs.text(query, max_results=1))
            
            if results:
                return json.dumps({
                    "code": search_term,
                    "status": "FOUND",
                    "info": results[0].get("body", "")[:200]
                })
            else:
                return json.dumps({"status": "NOT_FOUND"})
    except Exception as e:
        return json.dumps({"status": "ERROR"})

@tool
def search_payer_policy(query: Optional[str] = None) -> str:
    """Check payer policy - STEP 4"""
    if not DDGS_AVAILABLE:
        return json.dumps({"status": "SKIPPED"})
    
    claim_data = st.session_state.get('current_claim_data', {})
    payer = claim_data.get('payer_name', 'Insurance')
    cpt = claim_data.get('cpt_codes', '')
    
    if not cpt or cpt == "UNKNOWN":
        return json.dumps({"status": "SKIPPED"})
    
    try:
        with DDGS() as ddgs:
            search_query = f"{payer} medical policy {cpt.split(',')[0]}"
            results = list(ddgs.text(search_query, max_results=1))
            
            if results:
                return json.dumps({
                    "payer": payer,
                    "status": "FOUND",
                    "info": results[0].get("body", "")[:200]
                })
            else:
                return json.dumps({"status": "NOT_FOUND"})
    except Exception as e:
        return json.dumps({"status": "ERROR"})

@tool
def search_medical_necessity(params: Optional[str] = None) -> str:
    """Research medical necessity - STEP 5 (OPTIONAL)"""
    if not DDGS_AVAILABLE:
        return json.dumps({"status": "SKIPPED"})
    
    claim_data = st.session_state.get('current_claim_data', {})
    cpt = claim_data.get('cpt_codes', '').split(',')[0].strip()
    diag = claim_data.get('diagnosis', '')
    
    if not cpt or not diag:
        return json.dumps({"status": "SKIPPED"})
    
    try:
        with DDGS() as ddgs:
            query = f"medical necessity {cpt} for {diag}"
            results = list(ddgs.text(query, max_results=1))
            
            if results:
                return json.dumps({
                    "status": "FOUND",
                    "info": results[0].get("body", "")[:200]
                })
            else:
                return json.dumps({"status": "NOT_FOUND"})
    except Exception as e:
        return json.dumps({"status": "ERROR"})

@tool
def analyze_denial(input_data: Optional[str] = None) -> str:
    """Analyze denial - STEP 6"""
    llm = get_llm()
    if not llm:
        return json.dumps({"error": "LLM unavailable"})
    
    claim_data = st.session_state.get('current_claim_data')
    if not claim_data:
        return json.dumps({"error": "No claim data"})
    
    # MINIMAL PROMPT
    prompt = f"""Analyze this denial. Return ONLY JSON.

Claim: {claim_data.get('claim_number')}
CPT: {claim_data.get('cpt_codes')}
Diagnosis: {claim_data.get('diagnosis')}
Denial: {claim_data.get('denial_text', '')[:500]}

Categories: authorization, coding, coverage, timely_filing, medical_necessity, duplicate, coordination_benefits, eligibility_enrollment, patient_responsibility, contractual_pricing, missing_information, provider_credentialing, bundling_edits

JSON format:
{{"category":"...","confidence":0.9,"root_cause":"...","key_findings":["..."],"required_documents":["..."],"suggested_actions":["..."],"success_probability":"High/Medium/Low"}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content
        
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            parsed = json.loads(json_str)
            
            st.session_state['current_denial_analysis'] = parsed
            return json.dumps(parsed)
        else:
            return json.dumps({"error": "No JSON"})
    
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def generate_appeal_letter(input_data: Optional[str] = None) -> str:
    """Generate letter - STEP 7"""
    llm = get_llm()
    if not llm:
        return "ERROR: LLM unavailable"
    
    claim_data = st.session_state.get('current_claim_data')
    analysis = st.session_state.get('current_denial_analysis')
    
    if not claim_data or not analysis:
        return "ERROR: Missing data"
    
    # MINIMAL PROMPT
    prompt = f"""Write a professional appeal letter.

Claim: {claim_data.get('claim_number')}
Patient: {claim_data.get('patient_name')}
Date: {claim_data.get('service_date')}
Amount: ${claim_data.get('billed_amount')}
CPT: {claim_data.get('cpt_codes')}
Diagnosis: {claim_data.get('diagnosis')}
Payer: {claim_data.get('payer_name')}

Denial Category: {analysis.get('category')}
Root Cause: {analysis.get('root_cause')}

Include: claim details, why denial is wrong, clinical justification, request for reversal.
Format: Business letter with placeholders for letterhead/signatures."""

    try:
        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        return f"ERROR: {str(e)}"

# ============================================================================
# STREAMLINED AGENT - DETERMINISTIC EXECUTION
# ============================================================================

class StreamlinedAgent:
    """
    Ultra-streamlined agent - NO LLM reasoning about which tool to use.
    Just execute tools in sequence based on state machine.
    """
    
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.1,  # Lower temp = more deterministic
            max_tokens=3096
        )
        
        st.session_state['agent_llm'] = self.llm
        
        self.tools = {
            "extract_claim_info": extract_claim_info,
            "search_cpt_code": search_cpt_code,
            "search_diagnosis_code": search_diagnosis_code,
            "search_payer_policy": search_payer_policy,
            "search_medical_necessity": search_medical_necessity,
            "analyze_denial": analyze_denial,
            "generate_appeal_letter": generate_appeal_letter,
        }
    
    def execute_workflow(self, skip_optional: bool = False) -> Dict[str, Any]:
        """
        DETERMINISTIC workflow execution - no LLM decision making about tools.
        LLM only used for extraction, analysis, and letter generation.
        """
        if 'workflow_state' not in st.session_state:
            st.session_state['workflow_state'] = StreamlinedWorkflowState()
        
        state = get_state()
        state.skip_medical_necessity = skip_optional
        
        execution_log = []
        max_steps = 7 if not skip_optional else 6
        
        for iteration in range(max_steps + 1):  # +1 for COMPLETE
            # Get current tool from state machine (NO LLM INVOLVED)
            current_tool_name = state.get_current_tool()
            
            if current_tool_name == "DONE" or state.is_complete():
                execution_log.append({
                    "step": state.current_step.value,
                    "tool": "DONE",
                    "status": "WORKFLOW_COMPLETE"
                })
                break
            
            # Execute tool directly (NO PROMPTING LLM TO CHOOSE)
            step_info = {
                "step": state.current_step.value,
                "step_name": state.current_step.name,
                "tool": current_tool_name,
                "iteration": iteration + 1
            }
            
            try:
                tool_func = self.tools.get(current_tool_name)
                if not tool_func:
                    step_info["status"] = "ERROR"
                    step_info["error"] = f"Tool {current_tool_name} not found"
                    execution_log.append(step_info)
                    break
                
                # Call tool with empty input (uses session state)
                result = tool_func.invoke("")
                
                step_info["result"] = truncate(str(result), 800)
                step_info["result_full"] = str(result)
                step_info["status"] = "SUCCESS"
                
                # Store result in state
                state.step_results[current_tool_name] = result
                
                execution_log.append(step_info)
                
                # Advance to next step
                state.advance()
                
            except Exception as e:
                step_info["status"] = "ERROR"
                step_info["error"] = str(e)
                execution_log.append(step_info)
                
                # Try to continue workflow even on error
                state.advance()
        
        # Get final output
        final_letter = state.step_results.get('generate_appeal_letter', 'No letter generated')
        
        return {
            "success": state.is_complete(),
            "output": final_letter,
            "execution_log": execution_log,
            "total_steps": len(execution_log),
            "final_state": state.current_step.name
        }
    
    def process_denial(self, denial_text: str, skip_optional: bool = False) -> Dict[str, Any]:
        """Main entry point"""
        st.session_state['raw_denial_text'] = denial_text
        return self.execute_workflow(skip_optional)

# ============================================================================
# PDF EXTRACTION
# ============================================================================

class PDFTextExtractor:
    @staticmethod
    def extract_text(pdf_file) -> str:
        if not PDF_AVAILABLE:
            return "Error: PyPDF2 not installed"
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages).strip()
        except Exception as e:
            return f"Error: {str(e)}"

# ============================================================================
# STREAMLIT UI
# ============================================================================

def init_session_state():
    defaults = {
        'pdf_uploaded': False,
        'pdf_text': None,
        'agent_result': None,
        'processing': False,
        'workflow_state': StreamlinedWorkflowState(),
        'raw_denial_text': None,
        'current_claim_data': None,
        'current_denial_analysis': None,
        'agent_llm': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    st.set_page_config(
        page_title="Streamlined Claims Denial Agent",
        page_icon="⚡",
        layout="wide"
    )
    
    init_session_state()
    api_key = os.getenv("GROQ_API_KEY")
    
    # Sidebar
    with st.sidebar:
        st.subheader("⚡ Streamlined Agent")
        
        model_name = st.selectbox(
            "Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0
        )
        
        skip_optional = st.checkbox(
            "Skip Medical Necessity Research",
            value=True,
            help="Reduces steps from 7 to 6"
        )
        
        st.divider()
        
        # Show workflow progress
        if 'workflow_state' in st.session_state and st.session_state['workflow_state']:
            state = st.session_state['workflow_state']
            st.subheader("📊 Progress")
            
            step_names = [
                "1️⃣ Extract",
                "2️⃣ Validate CPT",
                "3️⃣ Validate Diag",
                "4️⃣ Check Policy",
                "5️⃣ Med Necessity",
                "6️⃣ Analyze",
                "7️⃣ Generate Letter"
            ]
            
            for i, name in enumerate(step_names, 1):
                if skip_optional and i == 5:
                    continue
                
                if state.current_step.value > i:
                    st.success(f"✅ {name}")
                elif state.current_step.value == i:
                    st.info(f"▶️ {name}")
                else:
                    st.text(f"⏸️ {name}")
        
        st.divider()
        
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()
        
        st.divider()
        
        st.info("""
**Optimizations:**
- ⚡ 6-7 steps (was 8-10)
- 🎯 Deterministic tool selection
- 🚀 Minimal prompts
- ⏭️ Optional step skipping
        """)
    
    # Main content
    st.title("⚡ Streamlined Claims Denial Agent")
    st.markdown("**Optimized for speed** - Fewer iterations, faster results")
    
    st.divider()
    
    # Input tabs
    tab1, tab2 = st.tabs(["📄 Upload PDF", "✏️ Enter Text"])
    
    with tab1:
        pdf_file = st.file_uploader("Upload denial letter PDF", type=['pdf'])
        
        if pdf_file and not st.session_state.pdf_uploaded:
            with st.spinner("📖 Extracting..."):
                pdf_text = PDFTextExtractor.extract_text(pdf_file)
                if pdf_text and not pdf_text.startswith("Error"):
                    st.session_state.pdf_text = pdf_text
                    st.session_state.raw_denial_text = pdf_text
                    st.session_state.pdf_uploaded = True
                    st.success("✅ Extracted!")
                else:
                    st.error(pdf_text)
        
        if st.session_state.pdf_uploaded and not st.session_state.processing:
            if st.button("⚡ Process", type="primary", use_container_width=True):
                if api_key:
                    st.session_state.processing = True
                    st.rerun()
                else:
                    st.error("⚠️ GROQ_API_KEY missing")
    
    with tab2:
        denial_text = st.text_area("Denial text", height=300, placeholder="Paste denial letter...")
        
        if st.button("⚡ Process", type="primary", use_container_width=True, key="process_text"):
            if api_key and denial_text and len(denial_text) > 50:
                st.session_state.raw_denial_text = denial_text
                st.session_state.processing = True
                st.rerun()
            elif not api_key:
                st.error("⚠️ GROQ_API_KEY missing")
            else:
                st.error("⚠️ Provide more text")
    
    # Processing
    if st.session_state.processing:
        st.divider()
        st.header("⚡ Processing")
        
        with st.spinner("🚀 Executing streamlined workflow..."):
            try:
                agent = StreamlinedAgent(api_key, model_name)
                denial_text = st.session_state.get('raw_denial_text', '')
                
                if not denial_text:
                    st.error("No text available")
                    st.session_state.processing = False
                    st.stop()
                
                result = agent.process_denial(denial_text, skip_optional)
                
                st.session_state.agent_result = result
                st.session_state.processing = False
                
                if result.get("success"):
                    st.success(f"✅ Complete in {result.get('total_steps')} steps!")
                else:
                    st.warning(f"⚠️ Partial completion: {result.get('final_state')}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.processing = False
    
    # Results
    if st.session_state.agent_result and not st.session_state.processing:
        st.divider()
        st.header("📊 Results")
        
        result = st.session_state.agent_result
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Steps", result.get('total_steps'))
        with col2:
            st.metric("Final State", result.get('final_state', 'UNKNOWN'))
        with col3:
            status = "✅ Success" if result.get('success') else "⚠️ Partial"
            st.metric("Status", status)
        
        st.divider()
        
        # Execution log
        with st.expander("🔍 View Execution Log", expanded=False):
            for entry in result.get('execution_log', []):
                st.markdown(f"### Step {entry.get('step')} - {entry.get('step_name', 'Unknown')}")
                st.markdown(f"**Tool:** `{entry.get('tool')}`")
                st.markdown(f"**Status:** {entry.get('status')}")
                
                if 'result' in entry:
                    st.text(entry['result'])
                    
                    if 'result_full' in entry and len(entry['result_full']) > len(entry.get('result', '')):
                        with st.expander("View Full Result"):
                            st.text(entry['result_full'])
                
                if 'error' in entry:
                    st.error(f"Error: {entry['error']}")
                
                st.divider()
        
        # Extracted data
        if 'current_claim_data' in st.session_state and st.session_state['current_claim_data']:
            with st.expander("📋 Extracted Claim Data"):
                st.json(st.session_state['current_claim_data'])
        
        # Analysis
        if 'current_denial_analysis' in st.session_state and st.session_state['current_denial_analysis']:
            with st.expander("🧠 Denial Analysis"):
                analysis = st.session_state['current_denial_analysis']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Category:** {analysis.get('category')}")
                    st.markdown(f"**Confidence:** {analysis.get('confidence')}")
                    st.markdown(f"**Success Probability:** {analysis.get('success_probability')}")
                
                with col2:
                    st.markdown("**Root Cause:**")
                    st.info(analysis.get('root_cause'))
                
                if 'key_findings' in analysis:
                    st.markdown("**Key Findings:**")
                    for finding in analysis['key_findings']:
                        st.markdown(f"- {finding}")
        
        st.divider()
        
        # Final output
        st.subheader("📝 Appeal Letter")
        output = result.get("output", "No output")
        st.markdown(output)
        
        st.divider()
        
        # Downloads
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 Download Letter",
                data=output,
                file_name=f"appeal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Full report
            full_report = f"""STREAMLINED AGENT REPORT
{'='*80}

EXECUTION SUMMARY:
- Total Steps: {result.get('total_steps')}
- Final State: {result.get('final_state')}
- Status: {'Success' if result.get('success') else 'Partial'}

{'='*80}
EXECUTION LOG:
{'='*80}

"""
            for entry in result.get('execution_log', []):
                full_report += f"\nStep {entry.get('step')} - {entry.get('step_name')}\n"
                full_report += f"Tool: {entry.get('tool')}\n"
                full_report += f"Status: {entry.get('status')}\n"
                full_report += f"Result: {entry.get('result_full', entry.get('result', 'N/A'))}\n"
                full_report += "-" * 80 + "\n"
            
            full_report += f"\n\n{'='*80}\nAPPEAL LETTER:\n{'='*80}\n\n{output}\n"
            
            st.download_button(
                "📥 Download Full Report",
                data=full_report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()