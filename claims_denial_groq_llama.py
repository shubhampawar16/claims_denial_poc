# claims_denial_analyzer_groq.py
"""
Claims Denial Reason Analyzer & Appeal Letter Drafting System
Python implementation with LangChain and Groq (llama-3.3-70b-versatile)
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import json
import PyPDF2
import io
import os
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# DATA MODELS
# ============================================================================

class DenialAnalysis(BaseModel):
    """Structured output for denial analysis"""
    category: str = Field(description="Primary denial category")
    confidence: float = Field(description="Confidence score between 0 and 1")
    root_cause: str = Field(description="Root cause explanation")
    key_findings: List[str] = Field(description="List of key findings from denial")
    required_documents: List[str] = Field(description="Documents needed for appeal")
    suggested_actions: List[str] = Field(description="Recommended action steps")
    success_probability: str = Field(description="Appeal success likelihood: High/Medium/Low")

class ClaimData(BaseModel):
    """Claim information structure"""
    claim_number: str
    patient_name: str
    service_date: str
    billed_amount: float
    cpt_codes: str
    diagnosis: str
    denial_text: str
    payer_name: Optional[str] = None

# ============================================================================
# LANGCHAIN COMPONENTS WITH GROQ LLAMA
# ============================================================================

class DenialAnalyzer:
    """Core LangChain-based denial analysis engine using Groq Llama-3.3-70b-versatile"""
    
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize with Groq Llama model
        
        Args:
            api_key: Groq API key
            model_name: Groq model name (llama-3.3-70b-versatile, llama-3.1-70b-versatile, etc.)
        """
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.3,
            max_tokens=4096
        )
        self.parser = PydanticOutputParser(pydantic_object=DenialAnalysis)
        
    def create_analysis_chain(self):
        """Create the denial analysis chain"""
        
        analysis_template = """You are an expert healthcare revenue cycle analyst specializing in claims denial management.

Analyze the following claim denial and provide a structured analysis:

CLAIM INFORMATION:
- Claim Number: {claim_number}
- Patient: {patient_name}
- Service Date: {service_date}
- Billed Amount: ${billed_amount}
- CPT Codes: {cpt_codes}
- Diagnosis: {diagnosis}
- Payer: {payer_name}

DENIAL TEXT:
{denial_text}

DENIAL CATEGORIES (use one):
- authorization: Prior Authorization Required
- coding: Coding/Documentation Error
- coverage: Service Not Covered
- timely_filing: Timely Filing Limit Exceeded
- medical_necessity: Medical Necessity Not Established
- duplicate: Duplicate Claim
- coordination_benefits: Coordination of Benefits Issue

YOUR TASK:
1. Categorize this denial into one of the standard categories
2. Identify the root cause with specific details
3. Extract key findings from the denial language
4. List all documents needed to support an appeal
5. Provide actionable steps for the revenue cycle team
6. Assess the probability of successful appeal (High/Medium/Low)

{format_instructions}

Provide your analysis in the specified JSON format."""

        prompt = ChatPromptTemplate.from_template(analysis_template)
        
        chain = prompt | self.llm | self.parser
        return chain
    
    def analyze_denial(self, claim_data: ClaimData) -> DenialAnalysis:
        """Analyze a claim denial"""
        chain = self.create_analysis_chain()
        
        result = chain.invoke({
            "claim_number": claim_data.claim_number,
            "patient_name": claim_data.patient_name,
            "service_date": claim_data.service_date,
            "billed_amount": claim_data.billed_amount,
            "cpt_codes": claim_data.cpt_codes,
            "diagnosis": claim_data.diagnosis,
            "payer_name": claim_data.payer_name or "Insurance Payer",
            "denial_text": claim_data.denial_text,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result

class AppealLetterGenerator:
    """LangChain-based appeal letter generator using Groq Llama-3.3-70b-versatile"""
    
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize with Groq Llama model
        
        Args:
            api_key: Groq API key
            model_name: Groq model name
        """
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.5,
            max_tokens=4096
        )
    
    def create_letter_chain(self):
        """Create appeal letter generation chain"""
        
        letter_template = """You are an expert medical billing specialist writing a professional appeal letter.

CLAIM INFORMATION:
- Claim Number: {claim_number}
- Patient: {patient_name}
- Service Date: {service_date}
- Billed Amount: ${billed_amount}
- CPT Codes: {cpt_codes}
- Diagnosis: {diagnosis}
- Payer: {payer_name}

DENIAL ANALYSIS:
- Category: {category}
- Root Cause: {root_cause}
- Key Findings: {key_findings}
- Required Documents: {required_documents}

HOSPITAL TEMPLATE GUIDELINES:
- Use professional, formal business letter format
- Include specific clinical rationale
- Reference payer's medical policies
- Cite relevant documentation
- Request specific action (reversal and payment)
- Set expectation for response timeframe
- Maintain respectful but firm tone

Write a comprehensive, professional appeal letter that:
1. Clearly states the appeal purpose and claim details
2. Explains why the denial is incorrect
3. Provides clinical justification for medical necessity
4. Lists supporting documentation being provided
5. References applicable medical policies
6. Requests reversal and payment
7. Includes appropriate placeholders for hospital letterhead and signatures

Format the letter as a complete, ready-to-use document."""

        prompt = ChatPromptTemplate.from_template(letter_template)
        chain = prompt | self.llm
        
        return chain
    
    def generate_letter(self, claim_data: ClaimData, analysis: DenialAnalysis) -> str:
        """Generate appeal letter"""
        chain = self.create_letter_chain()
        
        result = chain.invoke({
            "claim_number": claim_data.claim_number,
            "patient_name": claim_data.patient_name,
            "service_date": claim_data.service_date,
            "billed_amount": claim_data.billed_amount,
            "cpt_codes": claim_data.cpt_codes,
            "diagnosis": claim_data.diagnosis,
            "payer_name": claim_data.payer_name or "Insurance Payer",
            "category": analysis.category,
            "root_cause": analysis.root_cause,
            "key_findings": "\n".join(f"- {finding}" for finding in analysis.key_findings),
            "required_documents": "\n".join(f"- {doc}" for doc in analysis.required_documents)
        })
        
        return result.content

# ============================================================================
# FILE PARSERS
# ============================================================================

class EDIParser:
    """Parser for 835/837 EDI files"""
    
    @staticmethod
    def parse_835_remittance(file_content: str) -> dict:
        """Parse 835 remittance advice file"""
        # Simplified parser - in production use python-edi library
        lines = file_content.strip().split('\n')
        
        data = {
            "claim_number": "",
            "patient_name": "",
            "service_date": "",
            "billed_amount": 0.0,
            "paid_amount": 0.0,
            "denial_codes": [],
            "denial_text": ""
        }
        
        for line in lines:
            segments = line.split('*')
            
            if segments[0] == 'CLP':  # Claim Payment Information
                data["claim_number"] = segments[1]
                data["billed_amount"] = float(segments[3])
                data["paid_amount"] = float(segments[4])
            
            elif segments[0] == 'NM1':  # Patient Name
                data["patient_name"] = f"{segments[3]} {segments[4]}"
            
            elif segments[0] == 'DTM':  # Service Date
                if segments[1] == '232':
                    data["service_date"] = segments[2]
            
            elif segments[0] == 'CAS':  # Claim Adjustment
                data["denial_codes"].append(segments[2])
                if len(segments) > 3:
                    data["denial_text"] += f" {segments[2]}: Adjustment amount ${segments[3]}"
        
        return data
    
    @staticmethod
    def parse_837_claim(file_content: str) -> dict:
        """Parse 837 claim file"""
        lines = file_content.strip().split('\n')
        
        data = {
            "claim_number": "",
            "patient_name": "",
            "service_date": "",
            "billed_amount": 0.0,
            "cpt_codes": [],
            "diagnosis": ""
        }
        
        for line in lines:
            segments = line.split('*')
            
            if segments[0] == 'CLM':  # Claim Information
                data["claim_number"] = segments[1]
                data["billed_amount"] = float(segments[2])
            
            elif segments[0] == 'NM1':  # Name
                if segments[1] == 'IL':  # Insured/Patient
                    data["patient_name"] = f"{segments[3]} {segments[4]}"
            
            elif segments[0] == 'DTP':  # Date
                if segments[1] == '472':  # Service Date
                    data["service_date"] = segments[3]
            
            elif segments[0] == 'SV1':  # Professional Service
                cpt = segments[1].split(':')[1] if ':' in segments[1] else segments[1]
                data["cpt_codes"].append(cpt)
            
            elif segments[0] == 'HI':  # Health Care Diagnosis Code
                dx = segments[1].split(':')[1] if ':' in segments[1] else segments[1]
                data["diagnosis"] = dx
        
        return data

class PDFParser:
    """Parser for denial letters in PDF format"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            st.error(f"Error parsing PDF: {str(e)}")
            return ""

class PDFClaimExtractor:
    """Extract structured claim information from denial letter PDFs using Groq Llama AI"""
    
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize with Groq Llama model
        
        Args:
            api_key: Groq API key
            model_name: Groq model name
        """
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.1,
            max_tokens=2048
        )
        self.parser = PydanticOutputParser(pydantic_object=ClaimData)
    
    def create_extraction_chain(self):
        """Create the claim information extraction chain"""
        
        extraction_template = """You are an expert medical billing specialist analyzing a denial letter.

Extract the following claim information from the denial letter text below:

DENIAL LETTER TEXT:
{denial_letter_text}

YOUR TASK:
Extract and structure the following information:
1. Claim Number - Look for claim ID, reference number, or claim number
2. Patient Name - Full name of the patient
3. Service Date - Date when service was provided (format as YYYY-MM-DD)
4. Billed Amount - Total amount billed (extract number only, no $ sign)
5. CPT Codes - Procedure codes (comma-separated if multiple)
6. Diagnosis - Diagnosis codes or description
7. Denial Text - The actual denial reason/explanation from the letter
8. Payer Name - Insurance company name

IMPORTANT INSTRUCTIONS:
- If you cannot find a specific field, use reasonable defaults:
  - claim_number: "UNKNOWN"
  - patient_name: "UNKNOWN"
  - service_date: "{default_date}"
  - billed_amount: 0.0
  - cpt_codes: "UNKNOWN"
  - diagnosis: "UNKNOWN"
  - payer_name: "UNKNOWN"
- For denial_text, extract the complete denial reason/explanation
- Be as accurate as possible with the information you find
- For dates, convert to YYYY-MM-DD format

{format_instructions}

Provide the extracted information in the specified JSON format."""

        prompt = ChatPromptTemplate.from_template(extraction_template)
        chain = prompt | self.llm | self.parser
        
        return chain
    
    def extract_claim_info(self, pdf_text: str) -> ClaimData:
        """Extract claim information from PDF text"""
        chain = self.create_extraction_chain()
        
        # Use today's date as default
        default_date = datetime.now().strftime("%Y-%m-%d")
        
        result = chain.invoke({
            "denial_letter_text": pdf_text,
            "default_date": default_date,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = None
    if 'extraction_complete' not in st.session_state:
        st.session_state.extraction_complete = False
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'appeal_letter' not in st.session_state:
        st.session_state.appeal_letter = None
    if 'claim_data' not in st.session_state:
        st.session_state.claim_data = None

def main():
    st.set_page_config(
        page_title="Claims Denial Analyzer - Groq Llama",
        page_icon="📋",
        layout="wide"
    )
    
    init_session_state()
    
    # Load API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    model_name = "llama-3.3-70b-versatile"  # Groq Llama model
    
    # Sidebar with minimal controls
    with st.sidebar:
        st.subheader("🤖 Model Information")
        st.info(f"**Model:** {model_name}\n\n**Provider:** Groq")
        
        st.divider()
        
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.pdf_uploaded = False
            st.session_state.pdf_text = None
            st.session_state.extraction_complete = False
            st.session_state.extracted_data = None
            st.session_state.analysis = None
            st.session_state.appeal_letter = None
            st.session_state.claim_data = None
            st.rerun()
        
        st.divider()
        
        st.subheader("ℹ️ About")
        st.info(
            "This tool uses LangChain and Groq's Llama-3.3-70b-versatile to analyze "
            "claim denials and generate professional appeal letters."
        )
        
        st.markdown("**Powered by:**")
        st.markdown("- 🤖 Groq Llama-3.3-70b")
        st.markdown("- 🔗 LangChain")
        st.markdown("- 🎈 Streamlit")
    
    # Main header
    st.title("📋 Claims Denial Analyzer & Appeal Letter Drafter")
    st.markdown(
        "AI-powered denial analysis and appeal generation using **Groq Llama-3.3-70b-versatile**"
    )
    
    st.divider()
    
    # ========================================================================
    # SECTION 1: PDF Upload
    # ========================================================================
    st.header("📄 Step 1: Upload Denial Letter PDF")
    
    pdf_file = st.file_uploader(
        "Upload your denial letter in PDF format",
        type=['pdf'],
        help="Upload the payer denial letter PDF to automatically extract claim information"
    )
    
    if pdf_file:
        if not st.session_state.pdf_uploaded:
            with st.spinner("📖 Extracting text from PDF..."):
                pdf_text = PDFParser.extract_text_from_pdf(pdf_file)
                
                if pdf_text:
                    st.session_state.pdf_text = pdf_text
                    st.session_state.pdf_uploaded = True
                    st.success("✅ PDF text extracted successfully!")
                    
                    # Auto-extract claim information if API key is available
                    if api_key:
                        with st.spinner("🤖 Extracting claim information with Groq Llama AI..."):
                            try:
                                extractor = PDFClaimExtractor(api_key, model_name)
                                extracted_data = extractor.extract_claim_info(pdf_text)
                                st.session_state.extracted_data = extracted_data
                                st.session_state.extraction_complete = True
                                st.success("✅ Claim information extracted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error extracting claim information: {str(e)}")
                                st.info("💡 Please enter claim information manually below.")
                    else:
                        st.warning("⚠️ Please set GROQ_API_KEY in your .env file to auto-extract claim information.")
                else:
                    st.error("❌ Could not extract text from PDF. Please try a different file.")
        
        # Show extracted text in expander
        if st.session_state.pdf_text:
            with st.expander("📄 View Extracted PDF Text"):
                st.text_area("PDF Content", st.session_state.pdf_text, height=200, disabled=True)
    
    # ========================================================================
    # SECTION 2: Claim Information (Auto-populated or Manual)
    # ========================================================================
    if st.session_state.pdf_uploaded:
        st.divider()
        st.header("📝 Step 2: Review & Edit Claim Information")
        
        if st.session_state.extraction_complete and st.session_state.extracted_data:
            st.success("✅ Information auto-extracted from PDF. Please review and edit if needed.")
            extracted = st.session_state.extracted_data
        else:
            st.info("ℹ️ Please enter claim information manually.")
            extracted = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            claim_number = st.text_input(
                "Claim Number",
                value=extracted.claim_number if extracted else "",
                placeholder="CLM-2024-001234"
            )
            patient_name = st.text_input(
                "Patient Name",
                value=extracted.patient_name if extracted else "",
                placeholder="John Doe"
            )
            service_date = st.text_input(
                "Service Date (YYYY-MM-DD)",
                value=extracted.service_date if extracted else "",
                placeholder="2024-01-15"
            )
            billed_amount = st.number_input(
                "Billed Amount ($)",
                min_value=0.0,
                value=float(extracted.billed_amount) if extracted else 0.0,
                step=100.0,
                format="%.2f"
            )
        
        with col2:
            payer_name = st.text_input(
                "Payer Name",
                value=extracted.payer_name if extracted and extracted.payer_name else "",
                placeholder="Blue Cross Blue Shield"
            )
            cpt_codes = st.text_input(
                "CPT Codes",
                value=extracted.cpt_codes if extracted else "",
                placeholder="99213, 36415"
            )
            diagnosis = st.text_input(
                "Diagnosis",
                value=extracted.diagnosis if extracted else "",
                placeholder="Chronic back pain (M54.5)"
            )
        
        st.divider()
        
        denial_text = st.text_area(
            "Denial Reason / Text",
            value=extracted.denial_text if extracted else "",
            height=200,
            placeholder="Paste or edit the denial reason from the payer's EOB or remittance advice here..."
        )
        
        st.divider()
        
        if st.button("🧠 Analyze Denial with Groq Llama AI", type="primary", use_container_width=True):
            if not api_key:
                st.error("⚠️ GROQ_API_KEY not found in .env file. Please add GROQ_API_KEY to your .env file.")
            elif not denial_text:
                st.error("⚠️ Please provide at least the denial text to analyze")
            else:
                with st.spinner("🔄 Analyzing denial with Groq Llama-3.3-70b..."):
                    try:
                        # Create claim data
                        claim_data = ClaimData(
                            claim_number=claim_number,
                            patient_name=patient_name,
                            service_date=service_date,
                            billed_amount=billed_amount,
                            cpt_codes=cpt_codes,
                            diagnosis=diagnosis,
                            denial_text=denial_text,
                            payer_name=payer_name if payer_name else None
                        )
                        
                        # Analyze denial
                        analyzer = DenialAnalyzer(api_key, model_name)
                        analysis = analyzer.analyze_denial(claim_data)
                        
                        # Store in session state
                        st.session_state.claim_data = claim_data
                        st.session_state.analysis = analysis
                        
                        st.success("✅ Analysis complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.info("💡 Tip: Make sure your GROQ_API_KEY is valid and you have sufficient quota.")
    
    # ========================================================================
    # SECTION 3: Analysis Results
    # ========================================================================
    if st.session_state.analysis is not None:
        st.divider()
        st.header("🔍 Step 3: AI Analysis Results")
        
        analysis = st.session_state.analysis
        claim_data = st.session_state.claim_data
        
        # Category and confidence
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Denial Category", analysis.category.replace('_', ' ').title())
        with col2:
            st.metric("Confidence", f"{analysis.confidence * 100:.0f}%")
        with col3:
            color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
            st.metric(
                "Success Probability",
                f"{color.get(analysis.success_probability, '⚪')} {analysis.success_probability}"
            )
        
        st.divider()
        
        # Root cause
        st.subheader("🎯 Root Cause Analysis")
        st.error(analysis.root_cause)
        
        # Key findings and documents
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔎 Key Findings")
            for finding in analysis.key_findings:
                st.markdown(f"- {finding}")
        
        with col2:
            st.subheader("📋 Required Documents")
            for doc in analysis.required_documents:
                st.markdown(f"- {doc}")
        
        st.divider()
        
        # Suggested actions
        st.subheader("✅ Suggested Action Steps")
        for i, action in enumerate(analysis.suggested_actions, 1):
            st.success(f"**Step {i}:** {action}")
        
        st.divider()
        
        # Appeal deadline
        deadline = datetime.now() + timedelta(days=30)
        st.warning(f"⏰ **Appeal Deadline:** {deadline.strftime('%B %d, %Y')} (30 days from today)")
        
        # Auto-generate appeal letter if not already generated
        if st.session_state.appeal_letter is None:
            with st.spinner("📝 Generating appeal letter with Groq Llama..."):
                try:
                    generator = AppealLetterGenerator(api_key, model_name)
                    letter = generator.generate_letter(claim_data, analysis)
                    
                    st.session_state.appeal_letter = letter
                    st.success("✅ Appeal letter generated!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating letter: {str(e)}")
    
    # ========================================================================
    # SECTION 4: Appeal Letter
    # ========================================================================
    if st.session_state.appeal_letter is not None:
        st.divider()
        st.header("✉️ Step 4: Appeal Letter Draft")
        
        st.warning(
            "⚠️ **Important:** This is a draft letter. Please review and edit all sections "
            "before submission. Ensure all placeholders are replaced with actual information."
        )
        
        # Editable letter
        edited_letter = st.text_area(
            "Appeal Letter (Editable)",
            value=st.session_state.appeal_letter,
            height=600
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.download_button(
                "📥 Download as TXT",
                data=edited_letter,
                file_name=f"appeal_{st.session_state.claim_data.claim_number}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            ):
                st.success("✅ Letter downloaded!")
        
        with col2:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.code(edited_letter, language=None)
                st.info("Copy the text from the box above")

if __name__ == "__main__":
    main()
