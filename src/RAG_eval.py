import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import LangSmith essentials
from langsmith import Client, evaluate

# Import your purely in-memory pipeline
from rag_engine import initialize_medical_rag_pipeline

load_dotenv()

# Initialize LangSmith Client
client = Client()

def setup_evaluation_dataset(dataset_name: str):
    if client.has_dataset(dataset_name=dataset_name):
        print(f"📊 Dataset '{dataset_name}' already exists.")
        return client.read_dataset(dataset_name=dataset_name)

    print(f"🛠️ Creating new dataset: '{dataset_name}'...")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation dataset for the Medical RAG Chatbot."
    )
    
    examples = [
        ("What is the primary purpose of abdominal ultrasound in emergency medicine?", "Abdominal ultrasound is used in emergency settings to rapidly assess internal bleeding (hemorrhage), locate foreign objects like bullets, and perform a preliminary survey of damage. Its portability and versatility have made it common in emergency rooms and even limited ambulance service."),
        ("What are the four modes of ultrasound used in medical imaging?", "The four modes are: (1) A-mode – a single transducer scanning a line, used to measure distances and organ sizes; (2) B-mode – a linear array producing 2D images, the most commonly used type; (3) M-mode – rapid B-mode sequences to capture motion, especially heart movement; and (4) Doppler mode – measures velocities of moving material (like blood), used to investigate valve defects and arteriosclerosis."),
        ("Why must a patient fast before an abdominal ultrasound?", "Fasting for at least eight hours ensures the stomach is empty and small, and that the intestines and bowels are relatively inactive. It also allows the gallbladder to be visualized, since it contracts after eating and may not be seen on a full stomach."),
        ("What are abdominal wall defects and how are they treated?", "Abdominal wall defects are congenital (birth) defects where the stomach or intestines protrude through the abdominal wall, often because the umbilical opening is too large or developed improperly. Treatment is surgical repair; the organs are fully functional but misplaced. Prognosis is generally good if no other anomalies are present."),
        ("What is partial birth abortion (D&X) and what are its perceived advantages over other late-term methods?", "Partial birth abortion, formally called intact dilation and extraction (D&X), terminates a pregnancy in the late second or third trimester by delivering the fetus feet-first until only the head remains inside, then collapsing the skull. Perceived advantages include: (1) intact fetal removal for better autopsy/evaluation of anomalies; (2) potentially lower risk of uterine puncture or cervical damage; (3) avoidance of labor, which may be less emotionally traumatic; and (4) lower cost and shorter procedure time."),
        ("What is the maximum recommended daily dose of acetaminophen for adults, and what is the major risk of exceeding it?", "The maximum recommended dose for adults and children 12 and over is 4 grams (4,000 mg) per 24 hours. Exceeding this dose, especially when combined with alcohol, carries a serious risk of liver damage."),
        ("How does acetaminophen differ from aspirin in treating arthritis?", "Both drugs relieve pain and reduce fever with similar effectiveness. However, acetaminophen is less likely than aspirin to irritate the stomach. Crucially, unlike aspirin, acetaminophen does not reduce the redness, stiffness, or swelling that accompany arthritis."),
        ("What is achalasia and what is its most common first-line treatment?", "Achalasia is an esophageal disorder where the lower esophageal sphincter fails to relax, preventing normal swallowing. It is caused by degeneration of the nerve cells that normally signal the sphincter to open. The first-line treatment is balloon dilation — an inflatable balloon is passed down the esophagus and inflated to force the sphincter open — which is effective in about 70% of patients."),
        ("What are the three alternative treatments for achalasia when balloon dilation is not suitable?", "The three alternatives are: (1) Botulinum toxin injection into the sphincter to paralyze and relax it (symptoms typically return within 1–2 years); (2) Esophagomyotomy — surgical cutting of the sphincter muscle; and (3) Drug therapy with nifedipine (a calcium-channel blocker), which provides relief for about two-thirds of patients for up to two years."),
        ("What causes achondroplasia and what is its inheritance pattern?", "Achondroplasia is caused by a genetic defect that disrupts the normal development of cartilage into bone, particularly in the limbs. It is a dominant trait, meaning anyone carrying the defect displays all symptoms. A parent with achondroplasia has a 50% chance of passing it to each child. It occurs in approximately 1 in every 10,000 births and is the most common cause of dwarfism.")
    ]
    
    for question, expected_answer in examples:
        client.create_example(
            inputs={"question": question},
            outputs={"expected_answer": expected_answer},
            dataset_id=dataset.id
        )
        
    return dataset

def main():
    print("🚀 Starting LangSmith Evaluation Process...")
    
    try:
        # --- FIXED: Removed arguments so it matches your engine ---
        rag_app = initialize_medical_rag_pipeline()
    except FileNotFoundError as e:
        print(f"\n⚠️ Critical Error: {e}")
        return

    eval_llm = ChatGroq(model="llama-3.3-70b-Versatile", temperature=0)

    dataset_name = "Medical_RAG_Test_Suite_v1"
    setup_evaluation_dataset(dataset_name)

    print("\n⚖️ Running Evaluation... (This will process all 10 questions sequentially)")
    
    # 1. Target Task Function (Generates the answer)
    def predict_rag_answer(inputs: dict) -> dict:
        question = inputs["question"]
        state = {
            "original_question": question,
            "chat_history": [],
            "session_id": "eval_session" 
        }
        result = rag_app.invoke(state)
        return {"actual_answer": result["generation"]}

    # 2. Custom Evaluator (Grades the answer)
    def custom_accuracy_evaluator(run, example) -> dict:
        question = example.inputs["question"]
        expected = example.outputs["expected_answer"]
        actual = run.outputs["actual_answer"]
        
        eval_prompt = PromptTemplate.from_template(
            """You are a strict medical grading assistant.
            Compare the Actual Answer to the Expected Answer for the given Question.
            If the Actual Answer contains the core factual information of the Expected Answer, it is Correct.
            Respond strictly with the number 1.0 for Correct or 0.0 for Incorrect. No other text.

            Question: {question}
            Expected Answer: {expected}
            Actual Answer: {actual}
            
            Score (1.0 or 0.0):"""
        )
        
        chain = eval_prompt | eval_llm | StrOutputParser()
        result = chain.invoke({"question": question, "expected": expected, "actual": actual})
        
        try:
            score = float(result.strip())
        except ValueError:
            score = 0.0
            
        print(f"↳ Score for '{question[:30]}...': {'✅ Correct (1.0)' if score == 1.0 else '❌ Incorrect (0.0)'}")
        
        return {"key": "accuracy", "score": score}

    # 3. Run the LangSmith Experiment
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[custom_accuracy_evaluator],
        experiment_prefix="medical-rag-eval",
        metadata={"version": "1.0"},
        max_concurrency=1  # <--- Prevents hitting Groq API Rate Limits
    )
        
    print("\n✅ Evaluation Complete! Check the 'Datasets & Testing' tab in your LangSmith dashboard.")

if __name__ == "__main__":
    main()