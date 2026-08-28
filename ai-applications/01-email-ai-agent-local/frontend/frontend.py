import streamlit as st

class AppState:
    """
    Manages and encapsulates the global state of the application
    (st.session_state) using getters, setters, and auxiliary methods.
    """

    def __init__(self) -> None:
        self._state = st.session_state
        self._init_defaults()

    def _init_defaults(self) -> None:
        # Attribute that indicates whether the initial survey was completed:
        if "completed_survey" not in self._state:
            self._state.completed_survey = False
        # Attribute that stores the AI analyses of emails:
        if "ai_cache" not in self._state:
            self._state.ai_cache = {}
        # Attribute that stores the internal index (out of the total number of
        # unread emails loaded) of the currently displayed email:
        if "email_index" not in self._state:
            self._state.email_index = 0

    # Survey: getters and setters
    @property
    def completed_survey(self) -> bool:
        return self._state.completed_survey

    @completed_survey.setter
    def completed_survey(self, value: bool) -> None:
        self._state.completed_survey = value

    def save_survey_data(self, hrs: int, mins: int) -> None:
        self._state.weekly_time_string = f"{hrs}h {mins}m"
        self._state.total_weekly_minutes = (hrs * 60) + mins
        self._state.completed_survey = True

    # Emails: getters and setters
    @property
    def emails(self) -> list[dict] | None:
        return self._state.get("emails", None)

    @emails.setter
    def emails(self, email_list: list[dict]) -> None:
        self._state.emails = email_list
        self._state.email_index = 0

    @property
    def email_index(self) -> int:
        return self._state.email_index

    @email_index.setter
    def email_index(self, index: int) -> None:
        self._state.email_index = index

    @property
    def current_email(self) -> dict | None:
        if not self.emails:
            return None
        self.adjust_email_index()
        return self.emails[self.email_index]

    @property
    def amount_of_emails(self) -> int:
        return len(self.emails) if self.emails else 0

    def remove_current_email(self) -> None:
        """
        Removes the current email from the list and adjusts the index.
        """
        if self.emails and 0 <= self.email_index < len(self.emails):
            self.emails.pop(self.email_index)
            self.adjust_email_index()

    def adjust_email_index(self) -> None:
        """
        Adjusts the email index if it exceeds the number of emails.
        """
        if self.emails and self.email_index >= len(self.emails):
            self.email_index = max(0, len(self.emails) - 1)

    # AI results cache
    def get_ai_result(self, current_email_id: str) -> dict | None:
        return self._state.ai_cache.get(current_email_id)

    def save_ai_result(self, current_email_id: str, data: dict) -> None:
        self._state.ai_cache[current_email_id] = data

    # Navigation methods
    def next_email(self) -> None:
        if self.email_index < self.amount_of_emails - 1:
            self.email_index += 1

    def previous_email(self) -> None:
        if self.email_index > 0:
            self.email_index -= 1


class UI:
    """
    Class responsible exclusively for rendering visual components in Streamlit.
    """

    @staticmethod
    def setup_page(
        page_title: str = 'MailReader AI',
        page_icon: str = "📬"
        ) -> None:
        """
        Configure the Streamlit page with the given title and icon.
        """
        st.set_page_config(
            page_title = page_title,
            page_icon = page_icon,
            layout = "centered"
        )

    @staticmethod
    def render_survey() -> tuple[int, int] | None:
        """
        Draws the initial survey and returns the entered hours and minutes as a
        tuple.
        
        Returns:
        --------
        - (HOURS, MINUTES): A tuple containing the entered hours and minutes,
        """
        # Draw the survey in a bordered container:
        with st.container(border = True):
            st.subheader("📋 Encuesta de Diagnóstico")
            st.write(
                "¿Últimamente, cuánto tiempo semanal estimas que dedicas en" \
                "promedio a responder correos?"
            )
        # Draw two columns for hours and minutes input:
        column_hours, column_minutes = st.columns(2)
        # Draw the hours input in the first column:
        with column_hours:
            input_hours = st.number_input(
                label = "Horas", min_value = 0, step = 1, value = 0
            )
        # Draw the minutes input in the second column:
        with column_minutes:
            input_minutes = st.number_input(
                label = "Minutos", min_value = 0, max_value = 59,
                step = 1, value = 0
            )
        # Draw the submit button and handle its click event:
        if st.button(
            "Comenzar a procesar correos 🚀",
            use_container_width = True
        ):
            hours = int(input_hours) if input_hours else 0
            minutes = int(input_minutes) if input_minutes else 0
            # Validate that at least one of the values is greater than 0:
            if hours == 0 and minutes == 0:
                    st.warning(
                        "⚠️ Debes ingresar al menos un valor mayor a 0 en" \
                        "horas o minutos."
                    )
                    return None
            return hours, minutes
        # If the button was not clicked, return None:
        return None

    @staticmethod
    def render_header_and_connect(
        header_title: str = "📬 MailReader AI"
    ) -> bool:
        """
        Draws the page title and returns True if the user requests to connect to
        Gmail.
        """
        st.title(header_title)
        return st.button("🔗 Conectar a mi Gmail y leer correos")

    @staticmethod
    def render_email_card(
        current_email: dict,
        current_email_id: int,
        total_emails: int,
        ai_result: dict | None
    ) -> tuple[str | None, str | None]:
        """
        Renders the active email, AI analysis, and interaction buttons.

        Parameters
        ----------
        - CURRENT_EMAIL: a dictionary containing the email data.
        - CURRENT_EMAIL_ID: the index of the currently displayed email.
        - TOTAL_EMAILS: the total number of unread emails loaded.
        - AI_RESULT: a dictionary containing the AI analysis result for the
        email, or None if not available.

        Returns
        -------
        - ACTION: the action chosen by the user, which can be one of the
        following:
            - 'process_ai': process the email with AI.
            - 'send_reply': send a reply to the email.
            - 'mark_read': mark the email as read.
            - 'previous': navigate to the previous email.
            - 'next': navigate to the next email.
        - EDITED_DRAFT: the edited draft of the email reply, if applicable.
        """
        # Variables to store the selected action and the edited draft (if any):
        action, edited_draft = None, None
        # Draw a horizontal line and display the current email index:
        st.markdown("---")
        st.caption(
            f"Mostrando correo **{current_email_id + 1}** de**{total_emails}**"
        )
        # Draw the email content in a bordered container:
        with st.container(border = True):
            # Display the email sender, subject and link button in a bordered
            # container:
            with st.container(border = True):
                # Draw two columns: COLUMN_INFO for the sender and subject, and
                # COLUMN_LINK for the link button:
                column_info, column_link = st.columns([3, 1])
                with column_info:
                    st.subheader(f"De: {current_email['sender']}")
                    st.caption(f"Asunto: {current_email['subject']}")
                with column_link:
                    st.link_button("🔗 Abrir en Gmail", current_email["url"])
            # Draw the email body in a disabled text area:
            st.text_area(
                "Contenido del correo:",
                value = current_email["body"],
                height = 200,
                disabled = True,
            )
            # Draw the "Process with AI" button and handle its click event:
            if st.button("⚡ Procesar este correo con IA"):
                action = "process_ai"
            # If there is an AI result, display the urgency, abstract, and
            # draft:
            if ai_result:
                urgency = ai_result.get("urgency")
                abstract = ai_result.get("abstract")
                draft = ai_result.get("draft")
                if urgency == "Baja":
                    st.success(
                        f"🟢 **Urgencia Baja**\n\n**Resumen:** {abstract}"
                    )
                elif urgency == "Media":
                    st.warning(
                        f"🟡 **Urgencia Media**\n\n**Resumen:** {abstract}"
                    )
                else:
                    st.error(
                        f"🔴 **Urgencia Alta**\n\n**Resumen:** {abstract}"
                    )
                # If the urgency is Medium or High, display the editable draft:
                if urgency in ("Media", "Alta") and draft:
                    edited_draft = st.text_area(
                        "Borrador sugerido (editable):",
                        value = draft,
                        height = 120
                    )
                    # If the "Send Reply" button is clicked, set the action to
                    # "send_reply":
                    if st.button("🚀 Responder en el mismo hilo"):
                        action = "send_reply"
                # If the urgency is Low, display the "Mark as Read" button:
                elif urgency == "Baja":
                    if st.button("👁️ Marcar como leído"):
                        action = "mark_read"
            # Draw a horizontal line:
            st.markdown("---")
            # Draw the navigation buttons for previous and next emails:
            column_previous, column_next = st.columns(2)
            # Draw the "Previous Email" button in the left column:
            with column_previous:
                if st.button(
                    "⬅️ Correo Anterior",
                    use_container_width = True,
                    disabled = (current_email_id == 0),
                ):
                    # Set the action to "previous" if the button is clicked:
                    action = "previous"
            # Draw the "Next Email" button in the right column:
            with column_next:
                if st.button(
                    "➡️ Siguiente Correo",
                    use_container_width = True,
                    disabled = (current_email_id >= total_emails - 1),
                ):
                    # Set the action to "next" if the button is clicked:
                    action = "next"
        # Return the selected action and the edited draft (if any):
        return action, edited_draft

    @staticmethod
    def render_finished_message() -> None:
        """
        Displays a congratulatory message when all unread emails have been
        processed.
        """
        st.success(
            "🎉 ¡Felicidades! Has procesado todos los correos no leídos."
        )