
# Install required packages
!pip install -qU langchain huggingface_hub python-dotenv

# Import all necessary components
import os
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain_community.llms import HuggingFaceHub

# Set your HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize the HuggingFace model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",  # Free model that works well
    model_kwargs={
        "temperature": 0.7,       # Controls randomness
        "max_length": 512,        # Maximum response length
        "repetition_penalty": 1.1 # Prevents repetitive responses
    }
)

# ===== TASK 1: LLM INTEGRATION =====
def demonstrate_llm_integration():
    """Basic LLM interaction demonstration"""
    print("\n=== TASK 1: LLM INTEGRATION DEMO ===")
    response = llm.invoke("Explain neural networks to a beginner")
    print("LLM Response:", response)

# ===== TASK 2: MEMORY MANAGEMENT =====
def compare_memory_systems():
    """Compare different memory implementations"""
    print("\n=== TASK 2: MEMORY TYPE COMPARISON ===")
    
    test_conversation = [
        ("Hi!", "Hello! How can I help you today?"),
        ("What's AI?", "AI is artificial intelligence - machines that can perform tasks requiring human intelligence."),
        ("What are some examples?", "Examples include chatbots, recommendation systems, and self-driving cars."),
        ("Thanks!", "You're welcome! Let me know if you have other questions.")
    ]
    
    memory_types = {
        "Buffer": ConversationBufferMemory(),
        "Window (k=2)": ConversationBufferWindowMemory(k=2),
        "Summary": ConversationSummaryMemory(llm=llm),
        "Summary Buffer": ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    }
    
    for name, memory in memory_types.items():
        print(f"\n{name} Memory Results:")
        chain = ConversationChain(llm=llm, memory=memory)
        
        for human, ai in test_conversation:
            chain.predict(input=human)
            memory.save_context({"input": human}, {"output": ai})
        
        # Display memory contents
        if hasattr(memory, 'buffer'):
            print(memory.buffer)
        else:
            print(memory.load_memory_variables({})['history'])

# ===== TASK 3: PERSONA DEVELOPMENT =====
def create_persona_chatbot():
    """Create a chatbot with specific personality"""
    print("\n=== TASK 3: PERSONA CHATBOT ===")
    
    # Define our persona
    persona_description = """
    You are CodeCaptain, a senior software engineer with 15 years experience.
    You explain technical concepts clearly with real-world analogies.
    You're enthusiastic about teaching and make learning fun.
    You use occasional programming humor but stay professional.
    """
    
    # Create our prompt template
    prompt_template = f"""{persona_description}
    
    Conversation History:
    {{chat_history}}
    
    Human: {{input}}
    CodeCaptain:"""
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template=prompt_template
    )
    
    # Set up memory
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # Create our chain
    persona_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    # Test our persona
    print(persona_chain.run(input="Explain APIs to a beginner"))
    print(persona_chain.run(input="Give me a funny example about databases"))

# ===== INTERACTIVE DEMO =====
def interactive_demo():
    """Interactive chat with configurable memory"""
    print("\n=== INTERACTIVE CHAT DEMO ===")
    
    memory_options = {
        "1": ("Basic Memory (remembers everything)", ConversationBufferMemory),
        "2": ("Window Memory (last 3 messages)", lambda: ConversationBufferWindowMemory(k=3)),
        "3": ("Summary Memory (condensed)", lambda: ConversationSummaryMemory(llm=llm)),
        "4": ("Smart Memory (summary + recent)", lambda: ConversationSummaryBufferMemory(llm=llm, max_token_limit=100))
    }
    
    print("Choose your memory type:")
    for key, (desc, _) in memory_options.items():
        print(f"{key}. {desc}")
    
    choice = input("Select (1-4): ")
    if choice not in memory_options:
        print("Invalid choice, using basic memory")
        choice = "1"
    
    _, memory_creator = memory_options[choice]
    memory = memory_creator()
    
    # Set up our chat prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful technical assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Create our chain
    chat_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=True
    )
    
    print("\nChat started! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        
        response = chat_chain.run(input=user_input)
        print(f"\nAssistant: {response}")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Run all lab components
    demonstrate_llm_integration()
    compare_memory_systems()
    create_persona_chatbot()
    
    # Start interactive demo
    interactive_demo()