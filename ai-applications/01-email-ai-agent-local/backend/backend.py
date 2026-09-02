import base64
import os
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from operator import itemgetter
import json
from google import genai

class GmailConnection:
    """
    Manages the connection with the Gmail API: pull and push emails.
    """

    def __init__(
        self,
        gmail_api_credentials_filepath: str,
        gmail_api_scopes: tuple[str],
        gmail_api_token_filepath: str
        ) -> None:
        """
        Parameters
        ----------
        - GMAIL_API_CREDENTIALS_FILEPATH: path at which the application's Gmail
        API credentials JSON file (created in Google Cloud Console) is located.
        - GMAIL_SCOPES: Gmail data access permissions that the end users must
        authorize for the app to function.
        - GMAIL_API_TOKEN_FILEPATH: path at which the current user session's
        Gmail API credentials JSON file is located. If the file does not exist,
        it will be generated.
        """
        self.gmail_api_credentials_filepath = gmail_api_credentials_filepath
        self.gmail_api_scopes = gmail_api_scopes
        self.gmail_api_token_filepath = gmail_api_token_filepath
        # Load or create the Gmail API token and store it:
        self._load_gmail_api_token()
        # Create the Gmail API Client and store it:
        self._create_gmail_api_client()

    # def authenticate_gmail(
    def _load_gmail_api_token(
        self
    ) -> None:
        """
        - Manages the OAuth authentication with Gmail.
        
        - Creates a GMAIL_API_TOKEN_FILEPATH file that stores the end user
        session's credentials, thus avoiding the need for the end user to give
        authorization every time a Gmail request is performed.
        """
        # Current user session's Gmail API credentials (a.k.a. token):
        self.gmail_api_token = None
        # If GMAIL_API_TOKEN_FILEPATH already exists, read the token:
        if os.path.exists(self.gmail_api_token_filepath):
            self.gmail_api_token = Credentials.from_authorized_user_file(
                self.gmail_api_token_filepath, self.gmail_api_scopes
            )
        # If the token does not exist or is not valid:
        if not self.gmail_api_token or not self.gmail_api_token.valid:
            # If the token expired and contains a Refresh Token (i.e. contains
            # the key to request a new valid token):
            if (
                self.gmail_api_token
                and self.gmail_api_token.expired
                and self.gmail_api_token.refresh_token
            ):
                # Refresh the token, i.e. request a new valid token:
                self.gmail_api_token.refresh(Request())
            # If the token does not contain a Refresh Token:
            else:
                # If GMAIL_API_TOKEN_CREDENTIALS_FILEPATH does not exist, there
                # is no way to generate a new GMAIL_API_TOKEN. Therefore, raise
                # an error:
                if not os.path.exists(self.gmail_api_credentials_filepath):
                    error = (
                        'No se encontró el archivo',
                        f"'{self.gmail_api_credentials_filepath}'."
                    )
                    raise FileNotFoundError(' '.join(error))
                # If GMAIL_API_TOKEN_CREDENTIALS_FILEPATH exists, generate a new
                # token that contains the permissions specified in
                # GMAIL_API_SCOPES:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.gmail_api_credentials_filepath, self.gmail_api_scopes
                )
                # Run a local HTTP server, open the default web browser, let
                # the end user authorize Google permissions, and store the
                # resulting token as the instance attribute GMAIL_API_TOKEN:
                self.gmail_api_token = flow.run_local_server(port = 0)
            # Update GMAIL_API_TOKEN_FILEPATH with the new token:
            with open(self.gmail_api_token_filepath, 'w') as token:
                token.write(self.gmail_api_token.to_json())

    def _create_gmail_api_client(self) -> None:
        """
        - Creates and stores (as an instance attribute) a Gmail API
        Client instance.
        """
        # Create Gmail API Client instance and store it:
        self.gmail_api_client: Resource = build(
            'gmail', 'v1', credentials = self.gmail_api_token
        )

    def _extract_message_body(self, message_payload: dict) -> str:
        """
        Extracts and decodes the Gmail message's plain text body contained in
        MESSAGE_PAYLOAD.

        Parameters
        ----------
        - MESSAGE_PAYLOAD: JSON message data provided by the Gmail API.

        Returns
        -------
        - MESSAGE_BODY: plain text body of the Gmail message.
        """
        # If MESSAGE_PAYLOAD contains multiple parts, analyze them recursively:
        if 'parts' in message_payload:
            for part in message_payload['parts']:
                message_body = self._extract_message_body(part)
                # If the MESSAGE_BODY contains data (it is a non
                # empty string), return it:
                if message_body:
                    return message_body
        # If MESSAGE_PAYLOAD is written in plain text and contains data:
        elif (
            message_payload.get('mimeType') == 'text/plain'
            and 'data' in message_payload.get('body', {})
        ):
            # Plain text data contained in MESSAGE_PAYLOAD:
            data = message_payload['body']['data']
            # Decode the DATA: Base64url -> bytes -> str (UTF-8)
            message_body = base64.urlsafe_b64decode(data).decode(
                'utf-8', errors = 'ignore'
                )
            return message_body
        # If MESSAGE_BODY is written in plain text, but does not contain data,
        # return an empty string:
        return ""

    def _extract_header(
        self,
        message_payload: dict,
        header_name: str,
        header_replacement_label: str | None
    ) -> str | None:
        """
        Extract and return the value of the header with name HEADER_NAME, from
        the payload MESSAGE_PAYLOAD. If there is not header with name
        HEADER_NAME, instead return HEADER_REPLACEMENT_LABEL.

        Parameters
        ----------
        - MESSAGE_PAYLOAD: the Gmail message's payload.
        - HEADER_NAME: name of the header whose value will be returned (if it
        exists).
        - HEADER_REPLACEMENT_LABEL: value that will be returned if there is no
        header with name HEADER_NAME inside MESSAGE_PAYLOAD.

        Returns
        -------
        - HEADER: the value of the header with name HEADER_NAME, or instead
        HEADER_REPLACEMENT_LABEL if there is no header with name HEADER_NAME.
        """
        # Extract all the headers from MESSAGE_PAYLOAD. If there are no headers,
        # store an empty list:
        headers = message_payload.get('headers', [])
        # Extract the header with name HEADER_NAME. If it does not exist, instead
        # store the value HEADER_REPLACEMENT_LABEL:
        header = next(
                    (
                    h['value'] for h in headers if h['name'].lower() == header_name
                    ),
                    header_replacement_label
                )
        return header

    def pull_unread_emails(
        self,
        max_results = 100,
        subject_replacement_label: str = 'Sin asunto',
        sender_replacement_label: str = 'Desconocido',
        global_message_id_replacement_label: str | None = None
    ) -> list[dict]:
        """
        Pull the latest MAX_RESULTS unread mails.

        Parameters
        ----------
        - MAX_RESULTS: maximum amount of unread mails to pull.
        - SUBJECT_REPLACEMENT_LABEL: label to use for mails that do not contain
        subject metadata.
        - SENDER_REPLACEMENT_LABEL: label to use for mails that do not contain
        sender metadata.
        - GLOBAL_MESSAGE_ID_REPLACEMENT_LABEL: label to use for mails that do
        not contain Message-ID metadata. Message-ID is a globally unique
        identifier used for email messages.

        Returns
        -------
        - MESSAGES_DATA_LIST: list of dictionaries, each dictionary containing
        an email's metadata and content.
        """
        # Request the ID (and threadID) of every unread message (up to
        # MAX_RESULTS messages) in Gmail's inbox. This is done by using the
        # GMAIL_API_CLIENT corresponding to the end user:
        results = (
            self.gmail_api_client.users()
            .messages()
            .list(
                userId = 'me',
                q = 'is:unread in:inbox',
                maxResults = max_results
                )
            .execute()
        ) # Dictionary
        # Get the unread messages' ID and threadID in a dictionary with the format
        # {'id': id_value_string, 'threadId': thread_id_value_string}.
        # If there are no unread messages, just store an empty list:
        messages_ids = results.get('messages', [])
        # List where the unread messages' metadata and content will be stored:
        messages_data_list = []
        # Iterate over all the messages:
        for message_id, thread_id in map(
            itemgetter('id', 'threadId'), messages_ids
            ):
            # Request the current message's data through the GMAIL_API_CLIENT:
            message_data = (
                self.gmail_api_client.users()
                .messages()
                .get(userId = 'me', id = message_id, format = 'full')
                .execute()
            )
            # Message's content (a.k.a. payload):
            message_payload = message_data.get('payload', {})
            # Message's headers:
            subject = self._extract_header(
                message_payload = message_payload,
                header_name = 'subject',
                header_replacement_label = subject_replacement_label
            )
            sender = self._extract_header(
                message_payload = message_payload,
                header_name = 'from',
                header_replacement_label = sender_replacement_label
            )
            global_message_id = self._extract_header(
                message_payload = message_payload,
                header_name = 'message-id',
                header_replacement_label = global_message_id_replacement_label
            )
            # Extract the message's BODY:
            body = self._extract_message_body(message_payload)
            # If the message's body does not plain text characters (e.g. it only
            # contains HTML or images), BODY will be an empty string or contain
            # just blank spaces. In that case, replace BODY with the
            # Gmail-generated plain text snippet/preview:
            if not body.strip():
                body = message_data.get('snippet', '')
            # Message's direct URL in the Gmail website:
            gmail_url = f'https://mail.google.com/mail/u/0/#inbox/{thread_id}'
            # Store the message's metadata and content:
            messages_data_list.append({
                'id': message_id,
                'threadId': thread_id,
                'message_id_header': global_message_id,
                'sender': sender,
                'subject': subject,
                'body': body,
                'url': gmail_url,
            })
        # Return a list containing as many dictionaries as unread emails. Each
        # dictionary provides an email's metada and content:
        return messages_data_list

    def mark_as_read(
        self,
        message_id: str
    ) -> None:
        """
        Marks a Gmail message as 'READ'.

        Parameters
        ----------
        - MESSAGE_ID: Gmail ID of the message that will be marked as 'READ'.
        """
        if self.gmail_api_client:
            # Call the GMAIL_API_CLIENT to mark the message with ID MESSAGE_ID
            # as 'READ', i.e. remove the 'UNREAD' label.
            self.gmail_api_client.users().messages().modify(
                userId = 'me',
                id = message_id,
                body = {'removeLabelIds': ['UNREAD']}
            ).execute()
        else:
            raise('Cannot mark as read, as there is no Gmail API Client.')

    def send_email_reply(
        self,
        to: str,
        subject: str,
        body: str,
        thread_id: str,
        message_id_header: str | None = None,
    ) -> dict:
        """
        Send a Gmail response inside the current conversation thread.

        Parameters
        ----------
        - TO: recipient of the reply message, i.e. sender of the original email
        that is being replied to.
        - SUBJECT: subject of the original email that is being replied to.
        - BODY: reply message's plain text body.
        - THREAD_ID: thread ID in Gmail API.
        - GLOBAL_MESSAGE_ID: Message-ID of the original email that is being
        replied to. Message-ID is a globally unique identifier used for email
        messages.

        Returns
        -------
        - RESPONSE: JSON response received from the Gmail API after sending the
        reply.
        """
        # Create an EmailMessage instance:
        message = EmailMessage()
        message.set_content(body) # Add the body contents of the reply
        message['To'] = to # Add the recipient
        # Add the SUBJECT. If SUBJECT does not already include 'Re:', add it:
        if not subject.lower().startswith('re:'):
            message['Subject'] = f'Re: {subject}'
        else:
            message['Subject'] = subject
        # Link the reply to the Message-ID of the previous email
        # (RFC 5322 standard):
        if message_id_header:
            message['In-Reply-To'] = message_id_header
            message['References'] = message_id_header
        # Base64 URL Safe encoding necessary for the Gmail API:
        encoded_message = base64.urlsafe_b64encode(
            message.as_bytes()
            ).decode('utf-8')
        # Reply message's encoded data and threadId:
        reply_message = {'raw': encoded_message, 'threadId': thread_id}
        # Send the REPLY_MESSAGE and store Gmail API's response:
        response = self.gmail_api_client.users().messages().send(
            userId = 'me', body = reply_message
            ).execute()
        return response


class GeminiConnection:
    """
    Manages the connection with the Gemini API: send a prompt to analyze an
    email, then receive the requested information.
    """
    
    def __init__(
            self,
            gemini_api_key_filepath: str,
            gemini_model_name: str,
            gemini_prompt: str
            ) -> None:
        """
        Parameters
        ----------
        - GEMINI_API_KEY_FILEPATH: path of the text file that stores the Gemini
        API key (created in Google AI Studio).
        - GEMINI_MODEL_NAME: Gemini model to use.
        - GEMINI_PROMPT: prompt that will be sent to Gemini. Must be a string
        template containing the named placeholder 'email_body'.
        """
        self.gemini_api_key_filepath = gemini_api_key_filepath
        self.gemini_model_name = gemini_model_name
        self.gemini_prompt = gemini_prompt
        # Load the Gemini API key and store it:
        self._load_gemini_api_key()
        # Create a Gemini API Client and store it:
        self._create_gemini_api_client()

    def _load_gemini_api_key(self) -> None:
        """
        Store (as an instance attribute) the Gemini API key located inside a text
        file at SELF.GEMINI_API_KEY_FILEPATH.
        """
        # Verify if the text file at GEMINI_API_KEY_FILEPATH exists. If it does
        # not, raise an ERROR:
        if not os.path.exists(self.gemini_api_key_filepath):
            error = (
                f"No se encontró el archivo '{self.gemini_api_key_filepath}'.",
                'Créalo y pega tu API key dentro.',
            )
            raise FileNotFoundError(' '.join(error))
        # If the text file exists, open it and read the Gemini API key:
        with open(
            self.gemini_api_key_filepath, 'r', encoding = 'utf-8'
        ) as file:
            self.gemini_api_key = file.read()#.strip()

    def _create_gemini_api_client(self) -> None:
        """
        Creates and returns a Gemini API Client by using the provided
        GEMINI_API_KEY.

        Parameters
        ----------
        - GEMINI_API_KEY: Gemini API key (created in Google AI Studio).

        Returns
        -------
        - GEMINI_API_CLIENT
        """
        self.gemini_api_client = genai.Client(api_key = self.gemini_api_key)

    def analyze_email(
        self,
        email_body: str
    ) -> dict:
        """
        Send the EMAIL_BODY to Gemini. Return Gemini's response.

        Parameters
        ----------
        - EMAIL_BODY: the plain text body of the email that will be analyzed.
        """
        # Add the EMAIL_BODY to the GEMINI_PROMPT:
        gemini_prompt = self.gemini_prompt.format(email_body = email_body)
        # Request and store a JSON response from the GEMINI_API_CLIENT
        # (specifying both the GEMINI_MODEL_NAME and GEMINI_PROMPT):
        response_json = self.gemini_api_client.models.generate_content(
            model = self.gemini_model_name,
            contents = gemini_prompt,
            config = {'response_mime_type': 'application/json'}
        )
        # Load the JSON response into a dictionary:
        response_data = json.loads(response_json.text)
        return response_data