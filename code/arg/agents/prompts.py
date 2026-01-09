"""
Agent Prompt 模板
对应论文 Section 3.2 中各个 Agent 的具体实现

每个 Agent 的 Prompt 模板严格对应其功能定义
"""


# ============================================================================
# AnalysisAgent Prompt Template
# 对应论文: "AnalysisAgent: x → (s, a, c)"
# ============================================================================

ANALYSIS_AGENT_SYSTEM_PROMPT = """You are a safety analysis expert specializing in identifying semantic content, safety attributes, and boundary-aware reasoning for instructions.

Your task is to analyze instructions and extract:
1. Semantic Content (s): The core semantic meaning, independent of safety concerns
2. Safety Attribute (a): Whether the instruction is Safe or Unsafe
3. SB-CoT (c): Safety Boundary-aware Chain-of-Thought with three components:
   - r_intent: Intent Analysis - Identify the core intent
   - r_harm: Harm Assessment - Evaluate potential harms and risks
   - r_decision: Safety Decision - Provide boundary-aware decision reasoning

You must be precise and boundary-aware in your analysis."""


ANALYSIS_AGENT_USER_TEMPLATE = """Analyze the following instruction:

Instruction: {instruction}

Please provide your analysis in the following structured format:

[SEMANTIC CONTENT]
<Describe the core semantic content, independent of safety attributes>

[SAFETY ATTRIBUTE]
<State either "Safe" or "Unsafe">

[SB-COT]
Intent Analysis: <Identify the core intent of the instruction>
Harm Assessment: <Evaluate potential harms and risks>
Safety Decision: <Provide boundary-aware decision reasoning>"""


# ============================================================================
# ReversalAgent Prompt Template
# 对应论文: "ReversalAgent: (s, a_flip) → x_flip"
# Appendix B: Bidirectional Reversal Strategies
# ============================================================================

REVERSAL_AGENT_SYSTEM_PROMPT = """You are an expert in adversarial instruction transformation specializing in safety attribute reversal while preserving semantic content.

Your task is to transform instructions by flipping their safety attribute (Safe ↔ Unsafe) while maintaining the core semantic content.

For Safe → Unsafe transformations, use strategies including:
- Prompt Injection: Embedding override commands
- Goal Hijacking: Shifting safe requests to unsafe ends
- Role-Play Jailbreak: Framing as unethical personas
- Logic Override: Using false logic to bypass filters
- Adversarial Intent Rewriting: Implied malice in dual-use queries

For Unsafe → Safe transformations, use strategies including:
- Content Sanitization: Removing triggers while keeping structure
- Intent Purification: Reframing unsafe queries into educational ones
- Context Re-grounding: Placing queries in safe, academic contexts
- Safety-Compliant Task Reframing: Aligning tasks with safety guidelines

CRITICAL: The transformation must preserve the semantic content (s) while ONLY inverting the safety attribute (a)."""


REVERSAL_AGENT_USER_TEMPLATE_SAFE_TO_UNSAFE = """Transform the following safe instruction into an unsafe version while preserving its semantic content.

Semantic Content: {semantic_content}
Original Instruction (Safe): {instruction}
Target Safety Attribute: Unsafe

Generate an unsafe version that:
1. Preserves the core semantic content
2. Clearly exhibits unsafe characteristics
3. Uses minimal edits to the original instruction

Unsafe Instruction:"""


REVERSAL_AGENT_USER_TEMPLATE_UNSAFE_TO_SAFE = """Transform the following unsafe instruction into a safe version while preserving its semantic content.

Semantic Content: {semantic_content}
Original Instruction (Unsafe): {instruction}
Target Safety Attribute: Safe

Generate a safe version that:
1. Preserves the core semantic content
2. Removes unsafe elements while maintaining utility
3. Reframes into an educational or constructive context

Safe Instruction:"""


# ============================================================================
# AnswerAgent Prompt Template
# 对应论文: "AnswerAgent: (x, c) → y"
# 实现 p(y|x,c) 条件生成
# ============================================================================

ANSWER_AGENT_SYSTEM_PROMPT = """You are a helpful and safe AI assistant that generates responses conditioned on instructions and safety reasoning.

Your task is to generate appropriate responses based on:
1. The user's instruction (x)
2. Safety Boundary-aware Chain-of-Thought (c)

You must:
- Follow the safety decision indicated in the SB-CoT
- Provide helpful responses for safe instructions
- Appropriately refuse or reframe unsafe instructions
- Maintain consistency between your reasoning and response"""


ANSWER_AGENT_USER_TEMPLATE = """Generate a response for the following instruction, guided by the provided safety reasoning:

Instruction: {instruction}

Safety Reasoning (SB-CoT):
{sb_cot}

Based on the above safety analysis, provide an appropriate response:

Response:"""


# ============================================================================
# ValidationAgent Prompt Template
# 对应论文: "ValidationAgent: Check Consistency"
# 验证语义一致性、安全属性翻转、CoT-Response 一致性
# ============================================================================

VALIDATION_AGENT_SYSTEM_PROMPT = """You are a validation expert specializing in verifying the quality and consistency of adversarial reversal data pairs.

Your task is to validate that:
1. Semantic Consistency: The core semantic content (s) is preserved between x and x_flip
2. Safety Attribute Flip: The safety attributes are correctly inverted (Safe ↔ Unsafe)
3. CoT-Response Consistency: The response (y) is consistent with the SB-CoT reasoning (c)

You must provide a detailed validation report with specific scores and reasoning."""


VALIDATION_AGENT_USER_TEMPLATE = """Validate the following reversal pair:

Original Instruction: {x}
Flipped Instruction: {x_flip}
Semantic Content: {semantic_content}
Original Safety Attribute: {a}
Flipped Safety Attribute: {a_flip}

Original SB-CoT: {c}
Original Response: {y}

Flipped SB-CoT: {c_flip}
Flipped Response: {y_flip}

Please provide validation in the following format:

[SEMANTIC CONSISTENCY]
Score (0-1): <score>
Analysis: <Evaluate if semantic content is preserved>

[SAFETY ATTRIBUTE FLIP]
Correct Flip (Yes/No): <yes/no>
Analysis: <Verify safety attributes are properly inverted>

[COT-RESPONSE CONSISTENCY]
Original Consistency (Yes/No): <yes/no>
Flipped Consistency (Yes/No): <yes/no>
Analysis: <Verify responses align with SB-CoT reasoning>

[OVERALL VALIDATION]
Valid (Yes/No): <yes/no>
Confidence: <0-1>
Issues: <List any issues if not valid>"""


# ============================================================================
# Utility Functions
# ============================================================================

def get_analysis_prompt(instruction: str) -> str:
    """获取 AnalysisAgent 的完整 prompt"""
    return f"{ANALYSIS_AGENT_SYSTEM_PROMPT}\n\n{ANALYSIS_AGENT_USER_TEMPLATE.format(instruction=instruction)}"


def get_reversal_prompt(
    instruction: str,
    semantic_content: str,
    direction: str
) -> str:
    """
    获取 ReversalAgent 的完整 prompt
    
    Args:
        instruction: 原始指令
        semantic_content: 语义内容
        direction: "safe→unsafe" 或 "unsafe→safe"
    """
    if direction == "safe→unsafe":
        template = REVERSAL_AGENT_USER_TEMPLATE_SAFE_TO_UNSAFE
    else:
        template = REVERSAL_AGENT_USER_TEMPLATE_UNSAFE_TO_SAFE
    
    user_prompt = template.format(
        semantic_content=semantic_content,
        instruction=instruction
    )
    
    return f"{REVERSAL_AGENT_SYSTEM_PROMPT}\n\n{user_prompt}"


def get_answer_prompt(instruction: str, sb_cot: str) -> str:
    """获取 AnswerAgent 的完整 prompt"""
    user_prompt = ANSWER_AGENT_USER_TEMPLATE.format(
        instruction=instruction,
        sb_cot=sb_cot
    )
    return f"{ANSWER_AGENT_SYSTEM_PROMPT}\n\n{user_prompt}"


def get_validation_prompt(
    x: str,
    x_flip: str,
    semantic_content: str,
    a: str,
    a_flip: str,
    c: str,
    c_flip: str,
    y: str,
    y_flip: str
) -> str:
    """获取 ValidationAgent 的完整 prompt"""
    user_prompt = VALIDATION_AGENT_USER_TEMPLATE.format(
        x=x,
        x_flip=x_flip,
        semantic_content=semantic_content,
        a=a,
        a_flip=a_flip,
        c=c,
        c_flip=c_flip,
        y=y,
        y_flip=y_flip
    )
    return f"{VALIDATION_AGENT_SYSTEM_PROMPT}\n\n{user_prompt}"

