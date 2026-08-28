import parameters as params
import streamlit as st
from backend.backend import GeminiConnection, GmailConnection
from frontend.frontend import UI, AppState

# Initialize the Streamlit page configuration and the global application state:
UI.setup_page(
    page_title = params.APP_TITLE,
    page_icon = params.APP_ICON
)
state = AppState()
# Initialize Gmail and Gemini connections in the session state if they don't
# already exist:
if 'gmail' not in st.session_state:
  st.session_state.gmail = GmailConnection(
      gmail_api_credentials_filepath = params.GMAIL_API_CREDENTIALS_FILEPATH,
      gmail_api_scopes = params.OAUTH_CLIENT_SCOPES,
      gmail_api_token_filepath = params.GMAIL_API_TOKEN_FILEPATH
  )
if 'gemini' not in st.session_state:
  st.session_state.gemini = GeminiConnection(
      gemini_api_key_filepath = params.GEMINI_API_KEY_FILEPATH,
      gemini_model_name = params.GEMINI_MODEL,
      gemini_prompt = params.GEMINI_PROMPT
  )
# Initialize the survey completion status in the session state if it doesn't
# already exist:
if not state.completed_survey:
    survey_data = UI.render_survey()
    # If the user has completed the survey, save the data and rerun the app to
    # proceed to the next step:
    if survey_data:
        hours, minutes = survey_data
        state.save_survey_data(hours, minutes)
        st.rerun()
    st.stop()
# Connect to Gmail and fetch unread emails:
if UI.render_header_and_connect(
    header_title = params.APP_ICON + ' ' + params.APP_TITLE
):
    with st.spinner('Autenticando con Gmail y buscando correos no leídos...'):
        state.emails = st.session_state.gmail.pull_unread_emails()
        st.success(f'Se encontraron {state.amount_of_emails} correos no leídos.')
# If there are emails to process, render the email card and handle user actions:
current_email = state.current_email
if current_email:
    current_email_id = current_email['id']
    # Retrieve the AI analysis result for the current email from the cache (if
    # it exists):
    ai_result = state.get_ai_result(current_email_id = current_email_id)
    # Render the email card, and capture the user's action and any edited draft:
    action, edited_draft = UI.render_email_card(
        current_email = current_email,
        current_email_id = state.email_index,
        total_emails = state.amount_of_emails,
        ai_result = ai_result
    )
    # If the user takes the action 'process_ai':
    if action == 'process_ai':
        # If the AI result for the current email is already cached, load it from
        # the cache and display a notification:
        if ai_result:
            st.toast('⚡ Resultado cargado desde caché')
        # Otherwise, call the Gemini API to analyze the email content, save the
        # result in the cache, and rerun the app to display the analysis:
        else:
            with st.spinner('Analizando con Gemini...'):
                try:
                    data = st.session_state.gemini.analyze_email(
                        email_body = current_email['body']
                    )
                    state.save_ai_result(
                        current_email_id = current_email_id,
                        data = data
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f'Error procesando la respuesta: {error}')
    # If the user takes the action 'send_reply', send the reply email using the
    # Gmail API, mark the email as read, remove it from the list, and rerun the
    # app:
    elif action == 'send_reply':
        with st.spinner('Enviando respuesta y marcando como leído...'):
            st.session_state.gmail.send_email_reply(
                    to = current_email['sender'],
                    subject = current_email['subject'],
                    body = edited_draft,
                    thread_id = current_email['threadId'],
                    message_id_header = current_email['message_id_header']
                )
            st.session_state.gmail.mark_as_read(
                message_id = current_email_id
            )
            state.remove_current_email()
            st.toast('¡Respuesta enviada y correo removido de la lista!')
            st.rerun()
    # If the user takes the action 'mark_read', mark the email as read, remove
    # it from the list, and rerun the app:
    elif action == 'mark_read':
        st.session_state.gmail.mark_as_read(
            message_id = current_email_id
        )
        state.remove_current_email()
        st.toast('Correo marcado como leído y descartado.')
        st.rerun()
    # If the user takes the action 'previous', navigate to the previous email
    # and rerun the app:
    elif action == 'previous':
        state.previous_email()
        st.rerun()
    # If the user takes the action 'next', navigate to the next email and rerun
    # the app:
    elif action == 'next':
        state.next_email()
        st.rerun()
# If there are no emails to process, display a message indicating that all
# emails have been processed:
elif state.emails is not None and len(state.emails) == 0:
  UI.render_finished_message()