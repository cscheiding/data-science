from os.path import join as pjoin

# Gmail API:
GMAIL_API_CREDENTIALS_FILEPATH = pjoin(
    'credentials',
    'gmail_api_credentials.json'
)
GMAIL_API_TOKEN_FILEPATH = pjoin(
    'credentials',
    'gmail_api_token.json'
)
OAUTH_CLIENT_SCOPES = ('https://www.googleapis.com/auth/gmail.modify',)

# Gemini API:
GEMINI_API_KEY_FILEPATH = pjoin('credentials', 'gemini_api_key.txt')
GEMINI_MODEL = 'gemini-3.5-flash-lite'
GEMINI_PROMPT = """
    Analiza el siguiente correo y responde con la siguiente estructura de datos:
    - "urgency": una de estas tres opciones ('Alta', 'Media', 'Baja')
    - "abstract": resumen ejecutivo de 1 a 2 líneas
    - "draft": propuesta de respuesta profesional

    Comentarios de reglas:
    - 'Baja': el correo no requiere respuesta (automáticos/no-reply).
    Borrador debe ser null.
    - 'Media': solicita respuesta (explícita o implícitamente) y hay suficiente
    información. Generar borrador.
    - 'Alta': solicita respuesta (explícita o implícitamente) pero falta
    información. Generar borrador pidiendo aclaración.
    - Siempre debe generarse un resumen.

    Correo:
    {email_body}
    """

# UI:
APP_TITLE = 'AI Email Agent App'
APP_ICON = '📬'