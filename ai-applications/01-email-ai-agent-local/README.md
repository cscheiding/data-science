# Email AI Agent (Local) - Intelligent Email Triage & Automation

**Email AI Agent (Local)** is a Streamlit-powered desktop web application that integrates the **Gmail API** and **Google Gemini API** (`google-genai` SDK) to automate email diagnosis, classification, executive summarization and response generation.

The application follows a clean Object-Oriented Architecture (OOP) separating state management, user interface rendering and external API service integration.

---

## 🌟 Key Features

- **Diagnostic Onboarding**: Tracks and estimates weekly time spent managing emails.
- **Gmail OAuth2 Integration**: Securely authenticates and pulls unread emails directly from Gmail.
- **AI Email Analysis**:
  - **Urgency Scoring**: Classifies emails into `Baja` (Low), `Media` (Medium) or `Alta` (High).
  - **Executive Abstract**: Generates a 1–2 sentence summary of the email content.
  - **Context-Aware Drafts**: Proposes professional responses based on required actions.
- **Interactive Triage**: Edit proposed response drafts and send threaded replies directly from the app or mark low-priority emails as read.
- **Performance Caching**: Caches AI responses in session state to eliminate duplicate API requests for previously scanned emails.

---

## 🛠️ Tech Stack

- **Language**: Python 3.14+
- **Frontend / UI Framework**: Streamlit
- **LLM / AI Integration**: Google GenAI SDK (`google-genai` and Gemini 3.5)
- **Email API**: Google API Client (`googleapiclient` and `google-auth-oauthlib`)
- **Architecture**: Modular Object-Oriented Design (Model-View-Presenter variant)

---

## 🏗️ Architecture & Design

The project strictly separates presentation, state and business logic into modular components (`main.py`, `frontend/frontend.py` and `backend/backend.py`) to ensure clean maintenance, type safety and decoupling.

---

## 📂 Project Structure

```text
.
├── main.py
├── parameters.py
├── backend/
│   └── backend.py
├── frontend/
│   └── frontend.py
└── credentials/
    ├── gemini_api_key.txt
    ├── gmail_api_credentials.json
    └── gmail_api_token.json
```

- **`main.py`**  
  Application entry point. Coordinates user flows, links UI actions with backend services and handles Streamlit execution loops.

- **`parameters.py`**  
  Centralized configuration file for API scopes, file paths and model versions.

- **`backend/backend.py`**  
  Encapsulates external service interactions:
  - `GmailService`: Handles OAuth2 flow, token management, pulling unread messages, sending threaded replies and marking messages as read.
  - `GeminiService`: Formats structured prompts and calls Google Gemini models with JSON mode outputs.

- **`frontend/frontend.py`**  
  Contains presentation and state logic:
  - `AppState`: Class wrapping `st.session_state` using properties (getters/setters) for typed state access.
  - `UI`: Static/View class handling all Streamlit component rendering (survey, header, email card and navigation).

- **`credentials/`**  
  Directory storing sensitive key files and local OAuth tokens (ignored in git).

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.14 or higher
- A Google Cloud Project with the **Gmail API** enabled
- OAuth 2.0 Client credentials from Google Cloud Console
- A Google Gemini API Key

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/cscheiding/email-ai-agent-local.git](https://github.com/cscheiding/email-ai-agent-local.git)
cd email-ai-agent-local
pip install -r requirements.txt
```

### 3. Configuration

Place your API keys and credentials inside the `credentials/` folder:

1. Save your Gmail OAuth credentials as `credentials/gmail_api_credentials.json`.
2. Save your Gemini API Key in `credentials/gemini_api_key.txt`.
3. Note: `credentials/gmail_api_token.json` will be automatically generated upon your first successful login.

### 4. Running the App

Launch the Streamlit web server:

```bash
streamlit run main.py
```

---

## 🗺️ Roadmap & Future Developments

- [x] **Local Edition**: Single-user desktop app with local OAuth token handling and session-based caching.
- [ ] **Internationalization (i18n)**: Add multi-language UI support with an English and Spanish toggle.
- [ ] **Cloud / SaaS Edition**: Multi-tenant architecture with database token storage, background worker queues and multi-provider support.