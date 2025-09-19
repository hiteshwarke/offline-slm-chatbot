# 🤖 Offline SLM Chatbot

A **fully offline chatbot** built using a **Small Language Model (SLM)** in Python.  
This project demonstrates how you can run AI chat locally without any API, using **Streamlit** and **Hugging Face transformers**.

✅ **SEO Keywords:** Small Language Model, SLM, Offline Chatbot, Local AI Chat, Python AI, Edge AI, Streamlit Chatbot, Offline LLM

## 📌 Features
- Runs completely **offline** on your machine.
- Uses **Small Language Model (SLM)** for fast responses.
- Simple **Streamlit UI** for interactive chat.
- Customizable generation settings: max tokens, temperature, top-p sampling.
- Maintains **chat history** during session.
- CPU friendly, works with CUDA if available.

## 🛠 Installation & Setup
1. Clone the repo:
```bash
git clone https://github.com/hiteshwarke/offline-slm-chatbot.git
cd offline-slm-chatbot
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```

## 📖 How It Works
1. Streamlit provides the frontend for user input.
2. The SLM model (like `microsoft/phi-2`) loads locally using `transformers` library.
3. User input is tokenized and passed to the model to generate a response.
4. Responses are displayed and stored in **session_state** to maintain chat history.
5. Sidebar options allow tweaking generation parameters for experimentation.

## 👨‍💻 About the Author
Hi, I'm **Hitesh Warke**, a Full Stack Python Developer passionate about **AI, Machine Learning, and Web Development**.  
I build practical AI applications like chatbots, document Q&A systems, and API-driven solutions.

- 💼 Skills: Python, Django, FastAPI, React, Node.js, PHP, Laravel  
- 🤖 Interests: LLMs, SLMs, NLP, Generative AI, DSA  
- 🌐 Portfolio: [GitHub Profile](https://github.com/hiteshwarke)  
- 📬 Contact: hwarke2@gmail.com
- 🌐 Linkedin: https://www.linkedin.com/in/hitesh-warke-350775b9/

This project is part of my **personal portfolio** to showcase SLM applications in offline AI solutions.

## 🏷 License
MIT License