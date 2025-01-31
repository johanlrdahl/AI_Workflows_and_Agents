import os
from groq import Groq
import textwrap
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

# Based on https://media.licdn.com/dms/document/media/v2/D4D1FAQGI0EiS0GWJXg/feedshare-document-pdf-analyzed/B4DZQ_6TywHYAY-/0/1736239046163?e=1739404800&v=beta&t=LbDOMG3-WhmPp4sw-d8mzu5Rflhkr7N1VZ13wJpyQiA
# Baed on https://www.anthropic.com/research/building-effective-agents

# PARAMETERS
#model_name = "llama3-8b-8192"
#model_name_agent = "mixtral-8x7b-32768"
model_name = "mixtral-8x7b-32768"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def llm_call(prompt: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model = model_name
    )

    return chat_completion.choices[0].message.content


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text.
    Used for parsing structured responses
    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.
    
    Returns:
        str: The content of the specified XML tag,
        or an empty string if the tag is not found.
    """
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)

    extraction = ''
    if match:
        extraction = str(match.group(1)).strip()

    return extraction 


def run_simple_query() -> None:

    task_prompt = "Explain me briefly the agentic AI concept"

    response = llm_call(task_prompt)

    wrapped_response = textwrap.fill(response, width=80)
    print(wrapped_response)


def prompt_chaining(input_text: str, prompts: List[str]) -> str:
    """
    Execute a sequence of LLM calls where each step's output
    becomes the next step's input.
    Args:
        input_text: Initial text to process
        prompts: List of prompts/instructions for each step
    Returns:
        Final processed text after all steps
    """
    current_text = input_text
    
    for step, prompt in enumerate(prompts, 1):
        print(f"\nStep {step}:")

        # Combine the prompt with current text
        full_prompt = f"{prompt}\nInput: {current_text}"

        # Process through LLM
        current_text = llm_call(full_prompt)
        print(current_text)

    return current_text

# Workflow 1: Prompt Chaining
data_processing_steps = [
    """Extract only the numerical values and their associated metrics from the text.
    Format each as 'value: metric' on a new line.
    Example format:
    92: customer satisfaction
    45%: revenue growth"""
    ,
    """Convert all numerical values to percentages where possible.
    If not a percentage or points, convert to decimal (e.g., 92 points -> 92%).
    Keep one number per line.
    Example format:
    92%: customer satisfaction
    45%: revenue growth"""
    ,
    """Sort all lines in descending order by numerical value.
    Keep the format'value: metric' on each line.
    Example:
    92%: customer satisfaction
    87%: employee satisfaction"""
    ,
    """Format the sorted data as a markdown table with columns:
    | Metric | Value |
    |:--|--:|
    | Customer Satisfaction | 92% |"""
]

report = """
    Q3 Performance Summary:
    Our customer satisfaction score rose to 92 points this quarter.
    Revenue grew by 45% compared to last year.
    Market share is now at 23% in our primary market.
    Customer churn decreased to 5% from 8%.
    New user acquisition cost is $43 per user.report
    Product adoption rate increased to 78%.
    Employee satisfaction is at 87 points.
    Operating margin improved to 34%.
"""

wf1 = False
if wf1:
    final_output = prompt_chaining(input_text = report, prompts = data_processing_steps)

    print("\n---\nFinal Output:\n")
    print(final_output)

# Workflow 2: Parallelization
wf2 = False
if wf2:
    def parallel(prompt: str, inputs: List[str], n_workers: int = 3) -> List[str]:
        """
        Execute a function in parallel on a list of inputs.
        Args:
            prompt: Prompt to be used in each function call.
            inputs: List of input strings to process.
            func: Function to be executed on each input.
        Returns:
            List of outputs from the function calls.
        """
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(llm_call, f"{prompt}\nInput: {input_text}") for input_text in inputs]
        
        return [f.result() for f in futures]

    stakeholders = [
        """Customers:
        - Price sensitive
        - Want better tech
        - Environmental concerns""" 
        ,
        """Employees:
        - Job security worries
        - Need new skills
        - Want clear direction""" 
        ,
        """Investors:
        - Expect growth
        - Want cost control
        - Risk concerns""" 
        ,
        """Suppliers:
        - Capacity constraints
        - Price pressures
        - Tech transitions"""
    ]

    impact_results_prompt = """
    Analyze how market changes will impact this stakeholder group.
    Provide specific impacts and recommended actions.
    Format with clear sections and priorities.
    """
    impact_results = parallel(prompt = impact_results_prompt, inputs=stakeholders)  

    print("Analysis Results for Each Stakeholder Group:")
    print('='*50)

    for i, result in enumerate(impact_results, 1):
        print(f"\nStakeholder Group {i}:")
        print('-'*20)
        print(result)
        print('='*50)

# Workflow 3: Routing
wf3 = False
if wf3:
    def routing(input_text: str, routes: Dict[str, str]) -> str:
        """
        Route input text to specialsed prompt using content classification.
        """

        print(f'\nAvailable routes: {list(routes.keys())}')
        selector_prompt = f"""
        Analyze the input and select the most appropriate support team from these options: {list(routes.keys())}

        First explain your reasoning, then provide your selection in this XML format:
        <reasoning>
        Brief explanation of why this ticket should be routed to a specific team.
        Consider key terms, user intent, and urgency level.
        </reasoning>

        <selection>
        The chosen team name
        </selection>

        Input: {input_text}
        """.strip()

        route_response = llm_call(selector_prompt)
        reasoning = extract_xml(route_response, 'reasoning')
        route_key = extract_xml(route_response, 'selection').strip().lower()

        print("Routing Analysis:")
        print(reasoning)
        print(f"\nSelected route: {route_key}")

        # Process input with selected specialised prompt
        selected_prompt = routes.get(route_key, "Default Support Team") 

        return llm_call(f"{selected_prompt}\nInput: {input_text}")

    support_routes = {
        "billing": """You are a billing support specialist. Follow these guidelines:
        1. Always start with "Billing Support Response:"
        2. First acknowledge the specific billing issue
        3. Explain any charges or discrepancies clearly
        4. List concrete next steps with timeline
        5. End with payment options if relevant
        Keep responses professional but friendly.
        Input: """
        ,
        "technical": """You are a technical support engineer. Follow these guidelines:
        1. Always start with "Technical Support Response:"    
        2. List exact steps to resolve the issue
        3. Include system requirements if relevant
        4. Provide workarounds for common problems
        5. End with escalation path if needed
        Use clear, numbered steps and technical details.
        Input: """
        ,
        "account": """You are an account security specialist. Follow these guidelines:
        1. Always start with "Account Support Response:"
        2. Prioritize account security and verification
        3. Provide clear steps for account recovery/changes
        4. Include security tips and warnings
        5. Set clear expectations for resolution time
        Maintain a serious, security-focused tone.
        Input: """
        ,
        "product": """You are a product specialist. Follow these guidelines:
        1. Always start with "Product Support Response:"
        2. Focus on feature education and best practices
        3. Include specific examples of usage
        4. Link to relevant documentation sections
        5. Suggest related features that might help
        Be educational and encouraging in tone.
        Input: """
    }
        
    # Test with different support tickets
    tickets = [
        """Subject: Can't access my account
        Message: Hi, I've been trying to log in for the past hour but keep
        getting an'invalid password' error.
        I'm sure I'm using the right password. Can you help me regain access?
        This is urgent as I need to
        submit a report by end of day.
        - John""" 
        ,
        """Subject: Unexpected charge on my card
        Message: Hello, I just noticed a charge of $49.99 on my credit card from
        your company, but I thought
        I was on the $29.99 plan. Can you explain this charge and adjust
        it if it's a mistake?
        Thanks,
        Sarah""" 
        ,
        """Subject: How to export data?
        Message: I need to export all my project data to Excel.
        I've looked through the docs but can't
        figure out how to do a bulk export. Is this possible?
        If so, could you walk me through the steps?
        Best regards,
        Mike"""
    ]

    print("Processing support tickets...\n")
    for i, ticket in enumerate(tickets, 1):
        print(f"\nTicket {i}:")
        print('-'*20)
        print(ticket)
        print(f"\nResponse:")
        print('-'*20)
        response = routing(ticket, support_routes)
        print(response)
   
# Workflow 4: Evaluator-Optimizer
wf4 = True
if wf4:
    def generate(prompt: str, task: str, context: str = "") -> Tuple[str, str]:
        """Generate and improve a solution based on feedback."""
        full_prompt = f"{prompt}\n{context}\nTask: {task}" if context else f"{prompt}\nTask: {task}"
        response = llm_call(full_prompt)
        thoughts = extract_xml(response, "thoughts")
        result = extract_xml(response, "response")

        print("\n=== GENERATION START ===")
        print(f"Thoughts:\n{thoughts}\n")
        print(f"Generated:\n{result}")
        print("=== GENERATION END ===\n")

        return thoughts, result
    
    def evaluate(prompt: str, content: str, task: str) -> Tuple[str, str]:
        """Evaluate if a solution meets requirements."""
        full_prompt = f"{prompt}\nOriginal task: {task}\nContent to evaluate:{content}"
        response = llm_call(full_prompt)
        evaluation = extract_xml(response, "evaluation")
        feedback = extract_xml(response, "feedback")

        print("=== EVALUATION START ===")
        print(f"Status: {evaluation}")
        print(f"Feedback: {feedback}")
        print("=== EVALUATION END ===\n")

        return evaluation, feedback
    
    def eval_optimizer(task: str, evaluator_prompt: str, generator_prompt: str) -> Tuple[str, List[Dict[str, str]]]:
        """Keep generating and evaluating until requirements are met."""
        memory = []
        chain_of_thought = []
        thoughts, result = generate(generator_prompt, task)
        memory.append(result)
        chain_of_thought.append({"thoughts": thoughts, "result": result})
        improvement_count = 0

        while True:
            evaluation, feedback = evaluate(evaluator_prompt, result, task)
            if evaluation == "PASS":
                return result, chain_of_thought
            else:
                improvement_count += 1
                
            if improvement_count >= 2:
                print("Too many improvements needed. Stopping the process.")
                return result, chain_of_thought
            
            context = "\n".join([
                "Previous attempts:"
                , 
                *[f"- {m}" for m in memory]
                ,
                f"\nFeedback: {feedback}"
            ])

            thoughts, result = generate(generator_prompt, task, context)
            memory.append(result)
            chain_of_thought.append({"thoughts": thoughts, "result": result})

    evaluator_prompt = """
    Evaluate this following code implementation for:
    1.time complexity
    2.software engineering best practices
    You should be evaluating only and not attemping to solve the task.
    
    You need to output 'evaluation' and 'feedback':
    - evaluation: 
    -- PASS: if all criteria are met and you have no furthersuggestions for improvements
    -- NEEDS_IMPROVEMENT: if you believe there can be made significant improvements. Write the improvements in 'feedback'
    -- FAIL: if evaluation of the input does not make sense
    - feedback:
    Write a thorough, consistent and short feedback for what can be improved

    Output your evaluation concisely in the following xml format using the xml tags <evaluation> and <feedback>
    Remember to use both start and end tags
        <evaluation>
        One of PASS, NEEDS_IMPROVEMENT, or FAIL
        </evaluation>
        <feedback>
        What needs improvement and why.
        </feedback>

    XML example:
        <evaluation>
        PASS
        </evaluation>
        <feedback>
        The code does not use Python's built-in functions, such as XXX, which could significantly improve the computation time.
        </feedback>
    """

    generator_prompt = """
        Your goal is to complete the task based on <user input>.
        If there are feedback
        from your previous generations, you should
        reflect on them to improve your solution
        Output your answer concisely in the following xml format, using the tags <thoughts> and <response>
        Remember rto use both start and end tags.

        <thoughts>••••••••••••••••••
        [Your understanding of the task and feedback and
        how you plan to improve]
        </thoughts>
        <response>
        [Your code implementation here]
        </response>

    """

    task = """
        <user input>
        Suppose you have a dataset with n rows (samples) and p columns (features).
        You want to compute the full covariance (or correlation) matrix of these p features.
        Write me Python code this matrix and state the time
        complexity in Big-O notation with respect to n and p. You can not use any external
        libraries for this task.
        </user input>
    """

    eval_optimizer(task, evaluator_prompt, generator_prompt)

ai_agent = False
if ai_agent:
    class ContentAgent:
        def __init__(self, client: Groq):
            self.client = client
            
        def generate_content(self, prompt: str, previous_content: Optional[str] = None, feedback: Optional[str] = None) -> str:
            if previous_content and feedback:
                system_prompt = """You are a helpful AI assistant. Review the previous content and user feedback, 
                then generate an improved version that addresses the feedback while maintaining quality and coherence."""
                
                full_prompt = f"""Previous content: {previous_content}
                User feedback: {feedback}
                Please provide an improved version that addresses this feedback."""
            else:
                system_prompt = "You are a helpful AI assistant. Generate high-quality content based on the user's request."
                full_prompt = prompt
                
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                model="mixtral-8x7b-32768",
                temperature=0.7,
            )
            
            return response.choices[0].message.content

    def run_agent():
        agent = ContentAgent(client)
        
        # Get initial user request
        initial_prompt = input("What would you like me to write about? ")
        
        current_content = agent.generate_content(initial_prompt)
        
        while True:
            # Display current content
            print("\nCurrent content:")
            print("-" * 80)
            print(current_content)
            print("-" * 80)
            
            # Get user feedback
            print("\nIs this content good enough? Enter:")
            print("1. 'yes' if you're satisfied")
            print("2. 'no' and provide feedback for improvements")
            response = input("Your response: ").lower()
            
            if response == 'yes':
                print("\nGreat! Final content has been generated.")
                break
            elif response == 'no':
                feedback = input("Please provide specific feedback for improvements: ")
                current_content = agent.generate_content(initial_prompt, current_content, feedback)
            else:
                print("Invalid response. Please enter 'yes' or 'no'.")

    run_agent()