"""Molmo2-specific reasoning instruction variants for prompt augmentation."""

MOLMO2_LEGACY_REASONING_SUFFIX = (
    " Think step by step before answering. Present the final answer in LaTeX using \\boxed."
)

MOLMO2_REASONING_REWARD_TYPES = {
    "number_list",
    "numeric",
    "counting",
    "multiple_choice",
    "list_string_match",
    "string_match",
    "search",
}

MOLMO2_REASONING_INSTRUCTION_VARIANTS = (
    "Think through the solution step by step before you respond. Present the final result in LaTeX using \\boxed.",
    "Work carefully through each step before giving your answer. Format the final result in LaTeX using \\boxed.",
    "Reason through the problem step by step before answering. Provide the final answer in LaTeX using \\boxed.",
    "Go step by step in your reasoning before responding. Write the final answer in LaTeX using \\boxed.",
    "Think step by step before answering. Present the final result in LaTeX using \\boxed.",
    "Think step by step before answering. Express the final answer in LaTeX using \\boxed.",
    "Break down your reasoning step by step before responding. Display the final answer in LaTeX using \\boxed.",
    "Proceed step by step in your reasoning before giving the answer. Show the final result in LaTeX using \\boxed.",
    "Think step by step before answering. Present the conclusion in LaTeX using \\boxed.",
    "Think step by step before answering. State the final answer in LaTeX using \\boxed.",
    "Think carefully through every step before responding, and present the final result in LaTeX using \\boxed.",
    "Think step by step before answering, and provide the final answer in LaTeX using \\boxed.",
    "Analyze the problem step by step before replying, and write the final result in LaTeX using \\boxed.",
    "Think step by step before answering, and format the final answer in LaTeX using \\boxed.",
    "Think step by step before answering, and express the final result in LaTeX using \\boxed.",
    "Move through the reasoning step by step before giving your answer, and present the final expression in LaTeX using \\boxed.",
    "Examine the problem carefully in stages before replying, and provide the final answer in LaTeX using \\boxed.",
    "Lay out your reasoning step by step before answering, and show the final result in LaTeX using \\boxed.",
    "Approach the task methodically before responding, and conclude with the final answer in LaTeX using \\boxed.",
    "Solve it carefully step by step before replying, and present the final result in LaTeX using \\boxed.",
    "Think step by step before answering. Clearly state the final result.",
    "Work through the problem methodically before responding. Provide a clear final answer.",
    "Think step by step before answering. Conclude with a definitive result.",
    "Break down your thinking step by step before replying. End with a concise final answer.",
    "Analyze the problem in a stepwise manner before answering. Present a clear conclusion.",
    "Proceed carefully through each step before responding. Then state the final answer.",
    "Develop your reasoning step by step before replying. Finish with the final result.",
    "Consider the solution carefully in stages before answering. Provide the conclusion clearly.",
    "Work systematically through the reasoning before responding. Present the final outcome.",
    "Think carefully through each step before answering. Indicate the final result clearly.",
    "Solve the problem step by step before replying and end with a clear answer.",
    "Think step by step before answering and then give the final result.",
    "Examine the problem step by step before responding and conclude with a precise answer.",
    "Approach the task methodically before replying and provide a definitive final answer.",
    "Carefully reason through the solution before answering and clearly state the conclusion.",
    "Move through the analysis step by step before responding and then give the final result.",
    "Think through the reasoning in stages before answering and present a clear conclusion.",
    "Analyze each component carefully before replying and finish with the final answer.",
    "Work through the logic step by step before responding and clearly present the result.",
    "Think step by step before answering and then state the final conclusion.",
    "Think step by step before answering and end with a clear final answer.",
    "Break the solution into steps before answering and then provide the concluding result.",
    "Go through the reasoning carefully before responding and present the final answer clearly.",
    "Consider each part methodically before answering and conclude with the final result.",
    "Think carefully and sequentially before replying and then state the final answer clearly.",
    "Work step by step before answering and present a clear final result.",
    "Analyze the problem thoroughly before responding and give the final answer clearly.",
    "Develop the reasoning carefully before replying and provide the final conclusion.",
    "Think step by step before answering and end with the final result.",
    "Think step by step before answering and clearly state the final answer.",
)

if len(MOLMO2_REASONING_INSTRUCTION_VARIANTS) != 50:
    raise ValueError("Expected exactly 50 molmo2 reasoning instruction variants.")
if len(set(MOLMO2_REASONING_INSTRUCTION_VARIANTS)) != 50:
    raise ValueError("Molmo2 reasoning instruction variants must be unique.")
