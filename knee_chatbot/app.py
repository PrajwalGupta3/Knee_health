from flask import Flask, render_template, request, redirect, session
from dotenv import load_dotenv
import os, joblib, numpy as np
from groq import Groq

# Load environment
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Initialize Groq client (reads GROQ_API_KEY automatically)
client = Groq()

def generate_content(prompt: str) -> str:
    """
    Send a prompt to the Groq chat model and return the assistant's reply.
    """
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # replace with your exact model ID
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

# Load knee classification model
clf = joblib.load('knee_classifier.pkl')
age_map = {'Young':0,'Young Adult':1,'Adult':2,'Older Adult':3,'Senior':4}
gender_map = {'Male':1,'Female':0,'Non-binary':2}

def diagnose_knee(data: dict) -> list[str]:
    insights = []
    if data['ageGroup'] in ['Older Adult','Senior'] and data['pressure'] > 700:
        insights.append('Osteoarthritis: cartilage degradation due to high pressure.')
    if data['temperature'] > 37.5:
        insights.append('Infection/Inflammation: elevated temperature suggests bursitis or septic arthritis.')
    if abs(data['gyroX']) > 0.5 or abs(data['gyroY']) > 0.5:
        insights.append('Meniscus Tear: abnormal rotational motion.')
    if abs(data['accelZ'] - 9.8) > 1.5:
        insights.append('Ligament Strain/Tear: vertical instability detected.')
    return insights

# Flask routes
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f = request.form
        data = {
            'ageGroup': f['ageGroup'],
            'gender': f['gender'],
            **{k: float(f[k]) for k in [
                'height','weight','temperature','pressure',
                'gyroX','gyroY','gyroZ','accelX','accelY','accelZ'
            ]}
        }
        vec = np.array([[
            age_map[data['ageGroup']],
            data['height'], data['weight'],
            gender_map[data['gender']],
            data['temperature'], data['pressure'],
            data['gyroX'], data['gyroY'], data['gyroZ'],
            data['accelX'], data['accelY'], data['accelZ']
        ]])
        pred = clf.predict(vec)[0]
        session['prediction'] = 'Healthy Knee' if pred == 0 else 'Potentially Unhealthy Knee'
        session['diagnosis'] = diagnose_knee(data) if pred == 1 else []
        return redirect('/chat')
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    prediction = session.get('prediction', '')
    diagnosis = session.get('diagnosis', [])
    bot_reply = None
    if request.method == 'POST':
        user_q = request.form['question']
        prompt = (
            "You are a friendly knee-condition specialist.\n"
            "Speak in a friendly and conversational tone, like you're chatting with a friend.\n"
            "Explain the diagnosis in clear, everyday language. Avoid medical jargon.\n"
            "Use this structure:\n"
            "- Start with a warm greeting like: 'Hey there! Let's go over what you might be dealing with.'\n"
            "- For each diagnosis, use UPPERCASE TITLES for clarity (like MENISCUS TEAR)\n"
            "- Use '->' for bullet points.\n"
            "- For each diagnosis, explain:\n"
            "  -> What it is\n"
            "  -> What causes it\n"
            "  -> How to treat it\n"
            "  -> How to prevent it\n"
            f"Diagnoses: {'; '.join(diagnosis)}\n"
            f"User question: {user_q}"
        )
        response_text = generate_content(prompt)
        formatted_output = response_text.replace('\n', '<br>')
        bot_reply = formatted_output
    return render_template('chat.html', prediction=prediction, diagnosis=diagnosis, response=bot_reply)



if __name__ == '__main__':
    app.run(debug=True)
