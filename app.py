import os
import fitz
import docx
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from crewai import Agent, Task, Crew
from langchain_community.chat_models import ChatLiteLLM

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LLM Setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyAiLDDqU4lgBYY5SmJkYy6-6hg3D_37Omk"
llm = ChatLiteLLM(
    model="gemini/gemini-1.5-flash",
    temperature=0,
    api_key=os.environ["GOOGLE_API_KEY"]
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""

    if request.method == "POST":
        file = request.files.get("resume")
        evaluator_goal = request.form.get("evaluator_goal")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text
            if filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(filepath)
            elif filename.endswith(".docx"):
                resume_text = extract_text_from_docx(filepath)
            else:
                return "Unsupported file format."

            # --- Setup agents ---
            evaluator = Agent(
                role="Resume Evaluator",
                goal=evaluator_goal,
                backstory="Expert in talent assessment and AI hiring standards.",
                llm=llm,
                verbose=True
            )

            writer = Agent(
                role="Reply Writer",
                goal="Craft professional messages to candidates based on fit",
                backstory="HR professional who sends offer or rejection letters.",
                llm=llm,
                verbose=True
            )

            # --- First Task: Evaluation ---
            evaluation_task = Task(
                description=f"""Evaluate the following resume based on the goal: "{evaluator_goal}".
Respond with exactly one word YES or NO followed by a short reason.\n\nResume:\n{resume_text}""",
                agent=evaluator,
                expected_output="YES or NO with short reason"
            )

            # Create crew just for evaluation
            eval_crew = Crew(
                agents=[evaluator],
                tasks=[evaluation_task],
                verbose=False
            )

            eval_output = str(eval_crew.kickoff()).strip()

            # Determine response task based on evaluation
            if eval_output.startswith("YES"):
                reply_instruction = "Write a professional acceptance message to the candidate."
            elif eval_output.startswith("NO"):
                reply_instruction = "Write a polite rejection message to the candidate."
            else:
                reply_instruction = "Respond with an appropriate message due to unclear evaluation."

            # --- Second Task: Message Generation ---
            reply_task = Task(
                description=f"{reply_instruction}\nEvaluation: {eval_output}",
                agent=writer,
                expected_output="A professional message"
            )

            message_crew = Crew(
                agents=[writer],
                tasks=[reply_task],
                verbose=False
            )

            final_message = str(message_crew.kickoff()).strip()

            result = f"{eval_output}\n\n{final_message}"


    return render_template("index.html", result=result)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)