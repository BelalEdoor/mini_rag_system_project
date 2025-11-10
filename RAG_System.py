import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

print("✅ Libraries imported successfully!")

# 1. Load our embedding model
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Create a simple knowledge base
knowledge_base = [
    "The capital of France is Paris, a city famous for the Eiffel Tower and the Louvre museum.",
    "The Amazon rainforest is the world's largest tropical rainforest, known for its incredible biodiversity.",
    "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
    "The Great Wall of China is a series of fortifications stretching over 13,000 miles.",
    "Photosynthesis is the process used by plants to convert light energy into chemical energy."
]

# 3. Encode our knowledge base into embeddings
knowledge_embeddings = retriever_model.encode(knowledge_base, convert_to_tensor=True)

print(f"✅ Retriever model loaded and knowledge base encoded with {len(knowledge_base)} documents.")

# Load our question-answering (generator) model
generator = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

print("✅ Generator (QA) model loaded.")

def run_rag_assessment():
    """Runs a self-assessment of the RAG pipeline with multiple questions."""

    # Define our questions, expected context keywords, and expected answers
    test_questions = [
        {
            "question": "What is the highest mountain?",
            "expected_keyword": "Everest",
            "expected_answer": "Mount Everest"
        },
        {
            "question": "Which city is home to the Louvre museum?",
            "expected_keyword": "France",
            "expected_answer": "Paris"
        },
        {
            "question": "What process do plants use for energy?",
            "expected_keyword": "Photosynthesis",
            "expected_answer": "Photosynthesis"
        }
    ]

    score = 0
    total = len(test_questions) * 2 # 2 points per question (1 for retrieval, 1 for generation)

    print("--- 🚀 Starting RAG System Assessment ---\n")

    for i, test in enumerate(test_questions):
        question = test["question"]
        print(f"\n--- Question {i+1}: '{question}' ---")

        # --- 1. Retrieval Step ---
        question_embedding = retriever_model.encode(question, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(question_embedding, knowledge_embeddings)[0]
        top_result_index = torch.argmax(cos_scores)
        retrieved_context = knowledge_base[top_result_index]

        print(f"🔎  Retrieved Context: '{retrieved_context}'")

        # Check if the retrieval was correct
        if test["expected_keyword"] in retrieved_context:
            print("✅  Retrieval Correct!")
            score += 1
        else:
            print(f"❌  Retrieval Failed. Expected context with keyword: '{test['expected_keyword']}'")

        # --- 2. Generation Step ---
        qa_result = generator(question=question, context=retrieved_context)
        generated_answer = qa_result['answer']

        print(f"✍️  Generated Answer: '{generated_answer}'")

        # Check if the generation was correct
        if test["expected_answer"].lower() in generated_answer.lower():
            print("✅  Generation Correct!")
            score += 1
        else:
            print(f"❌  Generation Failed. Expected answer: '{test['expected_answer']}'")

    # --- Final Score ---
    print(f"\n--- 🏁 Assessment Complete ---")
    print(f"🎯 Final Score: {score} / {total}")
    if score == total:
        print("🎉🎉🎉 Perfect! Your RAG system is working as expected!")
    elif score >= total / 2:
        print("👍 Good job! The system is mostly correct.")
    else:
        print("🔧 The system ran into some issues. Review the steps and check the logic.")

# Run the assessment!
run_rag_assessment()

# ------------------------------
# 🧠 Task 2: Add new test questions
# ------------------------------
def run_rag_assessment_task_2():
    test_questions = [
        {
            "question": "What is the capital of France?",
            "expected_keyword": "Paris",
            "expected_answer": "Paris"
        },
        {
            "question": "Where is Mount Everest located?",
            "expected_keyword": "Himalayas",
            "expected_answer": "Himalayas"
        },
        {
            "question": "What process allows plants to convert light into energy?",
            "expected_keyword": "Photosynthesis",
            "expected_answer": "Photosynthesis"
        },
        {
            "question": "Which rainforest is the largest in the world?",
            "expected_keyword": "Amazon",
            "expected_answer": "Amazon rainforest"
        },
        # ✅ New Question (Challenge)
        {
            "question": "How long is the Great Wall of China?",
            "expected_keyword": "Great Wall",
            "expected_answer": "13,000 miles"
        }
    ]

    score = 0
    total = len(test_questions) * 2

    print("\n--- 🚀 Starting RAG System Assessment (Task 2) ---\n")

    for i, test in enumerate(test_questions):
        question = test["question"]
        print(f"\n--- Question {i+1}: '{question}' ---")
        question_embedding = retriever_model.encode(question, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(question_embedding, knowledge_embeddings)[0]
        top_result_index = torch.argmax(cos_scores)
        retrieved_context = knowledge_base[top_result_index]
        print(f"🔎  Retrieved Context: '{retrieved_context}'")

        # Retrieval check
        if test["expected_keyword"].lower() in retrieved_context.lower():
            print("✅  Retrieval Correct!")
            score += 1
        else:
            print(f"❌  Retrieval Failed. Expected context with keyword: '{test['expected_keyword']}'")

        # Generation step
        qa_result = generator(question=question, context=retrieved_context)
        generated_answer = qa_result['answer']
        print(f"✍️  Generated Answer: '{generated_answer}'")

        if test["expected_answer"].lower() in generated_answer.lower():
            print("✅  Generation Correct!")
            score += 1
        else:
            print(f"❌  Generation Failed. Expected answer: '{test['expected_answer']}'")

    print(f"\n--- 🏁 Assessment Complete ---")
    print(f"🎯 Final Score: {score} / {total}")
    if score == total:
        print("🎉🎉🎉 Perfect! Your RAG system handled the new question!")


# ✅ Run the updated assessment
run_rag_assessment_task_2()


# Task 3, Step 1: Add a new sentence to the knowledge base

knowledge_base_task_3 = [
    "The Moon orbits the Earth approximately every 27.3 days and influences the ocean tides through its gravitational pull."
]

# Re-encode the updated knowledge base
knowledge_embeddings_task_3 = retriever_model.encode(knowledge_base_task_3, convert_to_tensor=True)

print(f"✅ Knowledge base updated and re-encoded with {len(knowledge_base_task_3)} documents.")


# Task 3, Step 2: Test your newly added knowledge

def run_rag_assessment_task_3():
    test_questions = [
        # --- NEW QUESTION FOR YOUR NEW KNOWLEDGE ---
        {
            "question": "How long does it take for the Moon to orbit the Earth?",
            "expected_keyword": "Moon",
            "expected_answer": "27.3 days"
        }
    ]

    score = 0
    total = len(test_questions) * 2

    print("--- 🚀 Starting RAG System Assessment (Task 3) ---\n")

    for i, test in enumerate(test_questions):
        question = test["question"]
        print(f"\n--- Question {i+1}: '{question}' ---")
        # Use the updated embeddings from Task 3
        question_embedding = retriever_model.encode(question, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(question_embedding, knowledge_embeddings_task_3)[0]
        # Use the updated knowledge base from Task 3
        top_result_index = torch.argmax(cos_scores)
        retrieved_context = knowledge_base_task_3[top_result_index]
        print(f"🔎  Retrieved Context: '{retrieved_context}'")

        # Check retrieval
        if test["expected_keyword"].lower() in retrieved_context.lower():
            print("✅  Retrieval Correct!")
            score += 1
        else:
            print(f"❌  Retrieval Failed. Expected context with keyword: '{test['expected_keyword']}'")

        # Generate answer
        qa_result = generator(question=question, context=retrieved_context)
        generated_answer = qa_result['answer']
        print(f"✍️  Generated Answer: '{generated_answer}'")

        # Check answer correctness
        if test["expected_answer"].lower() in generated_answer.lower():
            print("✅  Generation Correct!")
            score += 1
        else:
            print(f"❌  Generation Failed. Expected answer: '{test['expected_answer']}'")

    print(f"\n--- 🏁 Assessment Complete ---")
    print(f"🎯 Final Score: {score} / {total}")
    if score == total:
        print("🏆🏆🏆 Success! You have successfully extended the knowledge of your RAG system!")

# Run the final assessment
run_rag_assessment_task_3()
