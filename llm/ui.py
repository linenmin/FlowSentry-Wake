# Copyright Axelera AI, 2025
"""
ui.py
Reusable Gradio UI builder for LLM chat, with a Axelera-Customized and a native UI.
"""

import pathlib

from llm.conversation import stream_response
import numpy as np

from axelera.app import config, logging_utils

LOG = logging_utils.getLogger(__name__)

ASSET_DIR = config.env.llm_root / "assets/file"
CSS_PATH = config.env.llm_root / "assets/styles.css"


def load_css():
    if not CSS_PATH.exists():
        LOG.error(f"CSS file not found: {CSS_PATH}")
        raise FileNotFoundError(f"CSS file not found: {CSS_PATH}")
    with open(CSS_PATH, 'r') as f:
        return f.read()


def format_system_metrics(tracers):
    """
    Format system metrics from tracers into a string for display.

    Args:
        tracers: List of tracer objects

    Returns:
        String representation of metrics or empty string if no tracers
    """
    if not tracers:
        return ""

    system_metrics = []
    for tracer in tracers:
        for metric in tracer.get_metrics():
            system_metrics.append(f"{metric.title}: {metric.value:.1f}{metric.unit}")

    if system_metrics:
        return " | " + " | ".join(system_metrics)
    return ""


def create_header_html():
    import gradio as gr

    # Use the merged logo if you demo on Arduinio platform
    # logo_path = str(ASSET_DIR / "merged_logo.png")
    logo_path = str(ASSET_DIR / "Axelera-AI-logo-White.png")
    icon_path = str(ASSET_DIR / "AX-TOP-Icon-.png")
    with gr.Row(elem_classes="header-container") as header:
        with gr.Column(elem_classes="logo-wrapper"):
            gr.Image(
                logo_path,
                container=False,
                show_label=False,
                elem_classes=["logo"],
                interactive=False,
                show_download_button=False,
                show_fullscreen_button=False,
            )
        with gr.Row(elem_classes="title-container"):
            gr.Image(
                icon_path,
                container=False,
                show_label=False,
                elem_classes=["icon"],
                interactive=False,
                show_download_button=False,
                show_fullscreen_button=False,
            )
            gr.Markdown(
                """
                <div class="title-text">
                    <p class="axelera">Axelera AI</p>
                    <p class="slm-demo">SLM Demo</p>
                </div>
                """
            )
    return header


def get_short_model_name(model_name):
    """Return the short model name for display, handling both with and without '/' in the name."""
    if model_name is not None:
        if "/" in model_name:
            return model_name.split("/")[-1]
        else:
            return model_name
    return "Unknown Model"


def build_llm_ui(
    model_instance,
    chat_encoder,
    tokenizer,
    max_tokens,
    system_prompt,
    temperature,
    model_name,
    no_history=False,
    tracers=None,
):
    """
    Build a Gradio Blocks UI for LLM chat, matching phi3_demo.py.
    Returns a Gradio Blocks app.
    """
    import gradio as gr

    # Ensure the UI uses the current system prompt from the encoder
    # This allows the UI to properly show the correct prompt even after page refresh
    current_system_prompt = getattr(chat_encoder, 'system_prompt', system_prompt)

    # Update the system prompt only if it's not already set in the encoder
    if not hasattr(chat_encoder, 'system_prompt'):
        chat_encoder.update_system_prompt(system_prompt)

    css = load_css()
    LOG.info("Building LLM Gradio UI...")

    # Extract short model name for display
    short_model_name = get_short_model_name(model_name)

    def chat_fn(message, history):
        import time

        if history is None:
            history = []

        # If no_history is enabled, only keep the visual history in the UI
        # but don't pass it to the encoder - treat each message independently
        chat_history = [] if no_history else [(h[0], h[1]) for h in history if h[0] and h[1]]

        # Show user message immediately
        history = history + [[message, ""]]
        yield history, "", "Ready"

        try:
            input_ids, embedding_features = chat_encoder.encode(message, chat_history)
            response = ""
            # Use stream_response for streaming output
            for new_text, stats in stream_response(
                model_instance,
                chat_encoder,
                tokenizer,
                input_ids,
                embedding_features,
                max_tokens,
                temperature,
                tokenizer.eos_token_id,
                getattr(model_instance, 'end_token_id', None),
            ):
                response = new_text
                history[-1][1] = response
                status_str = f"TTFT: {stats['ttft']:.2f}s | Tokens/sec: {stats['tokens_per_sec']:.2f} | Tokens: {stats['tokens']}"

                if tracers:
                    status_str += format_system_metrics(tracers)

                yield history, "", status_str

            # We need to add the message-response pair to history to make the stateless approach work
            # since we need to properly consume the generated text
            chat_encoder.add_to_history(message, response)
            LOG.info(f"User message processed, response length: {len(response)}")

            # Clear history if no_history flag is set
            if no_history:
                LOG.debug(
                    "--no-history flag enabled, clearing conversation history after response"
                )
                # Clear history but preserve system prompt to avoid unnecessary reprocessing
                chat_encoder.reset(preserve_system_prompt=True)
        except Exception as e:
            # Handle any exceptions that occur during processing
            LOG.error(f"Error processing message: {str(e)}")
            error_message = f"Sorry, an error occurred: {str(e)}"
            history[-1][1] = error_message
            yield history, "", "Error"

    def clear_history():
        # Reset but preserve the system prompt to avoid unnecessary reprocessing
        chat_encoder.reset(preserve_system_prompt=True)
        LOG.info("Chat history cleared.")
        return [], "", "Ready"

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray").set(
            body_background_fill="linear-gradient(180deg, #000000 0%, #0A1130 50%, #152147 100%), radial-gradient(circle at top right, #000000, transparent)",
            body_background_fill_dark="linear-gradient(180deg, #000000 0%, #0A1130 50%, #152147 100%), radial-gradient(circle at top right, #000000, transparent)",
            background_fill_primary="#0D1223",
            background_fill_primary_dark="#0D1223",
            background_fill_secondary="rgba(13, 18, 35, 0.95)",
            background_fill_secondary_dark="rgba(13, 18, 35, 0.95)",
            body_text_color="white",
            body_text_color_dark="white",
            body_text_color_subdued='*secondary_200',
            body_text_color_subdued_dark='*secondary_200',
            border_color_primary="rgba(180, 180, 180, 0.1)",
            border_color_primary_dark="rgba(180, 180, 180, 0.1)",
            panel_background_fill="#0D1223",
            panel_background_fill_dark="#0D1223",
            block_background_fill="#050918",
            block_label_background_fill="#050918",
            input_background_fill="#050918",
            button_secondary_background_fill="#050918",
            input_border_color="rgba(180, 180, 180, 0.2)",
        ),
        css=css,
    ) as demo:
        header = create_header_html()
        gr.Markdown(
            f'<div class="chat-header"><b>Chat with the <span style="color:#FBBE18">{short_model_name}</span> model.<br>Enter your message and see the AI respond in real-time.</b></div>'
        )
        # Add a state to store the current system prompt
        system_prompt_state = gr.State(current_system_prompt)
        with gr.Column(elem_classes="chatbox"):
            chatbot = gr.Chatbot(
                show_label=False,
                elem_classes="chat-display",
                avatar_images=(None, str(ASSET_DIR / "AX-TOP-Icon-.png")),
                resizeable=True,
            )
            with gr.Row(elem_classes="input-container"):
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Ask me something...",
                    container=False,
                    elem_classes="message-input",
                    scale=9,
                    submit_btn='➤',
                )
                clear = gr.ClearButton(value="Clear", scale=1, elem_classes="clear-btn")
        # Modal for system prompt
        with gr.Group(visible=False) as modal_group:
            gr.Markdown("### System Prompt Settings")
            # Bind the textbox value to the state
            system_prompt_input = gr.Textbox(
                label="", value=current_system_prompt, lines=5, elem_classes="system-prompt-input"
            )
            with gr.Row():
                update_prompt_btn = gr.Button("Update", elem_classes="update-prompt-btn")
                cancel_btn = gr.Button("Cancel", elem_classes="cancel-btn")
        # Add settings button with icon for system prompt
        with gr.Row(elem_classes="status-container"):
            with gr.Column(scale=1, min_width=45):
                settings_btn = gr.Button(
                    value="",
                    icon=str(ASSET_DIR / "setting.png"),
                    elem_classes="settings-btn",
                )
            with gr.Column(scale=5):
                pass  # Placeholder for status messages if needed
            with gr.Column(scale=4):
                pass  # Placeholder for system info if needed

        def show_modal(current_prompt):
            # Set the textbox value to the current system prompt
            return gr.Group(visible=True), gr.update(value=current_prompt)

        def hide_modal():
            return gr.Group(visible=False), gr.update()

        def update_prompt(new_prompt):
            chat_encoder.update_system_prompt(new_prompt)
            LOG.info(f"System prompt updated: {new_prompt}")
            return gr.Group(visible=False), new_prompt

        # Add a status bar
        status = gr.Markdown(value="Ready", elem_classes="status-text")
        msg.submit(chat_fn, [msg, chatbot], [chatbot, msg, status], queue=True, api_name="chat")
        clear.click(clear_history, None, [chatbot, msg, status], queue=False)
        update_prompt_btn.click(
            fn=update_prompt,
            inputs=[system_prompt_input],
            outputs=[modal_group, system_prompt_state],
        )
        cancel_btn.click(fn=hide_modal, outputs=[modal_group, system_prompt_input])
        # When opening the modal, set the textbox value to the current system prompt
        settings_btn.click(
            fn=show_modal, inputs=[system_prompt_state], outputs=[modal_group, system_prompt_input]
        )

        # ESC key to clear chat history
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') {
                        document.querySelector('button.clear-btn').click();
                    }
                });
            }""",
        )
    return demo


def build_llm_ui_native(
    model_instance,
    chat_encoder,
    tokenizer,
    max_tokens,
    system_prompt,
    temperature,
    model_name=None,
    no_history=False,
    tracers=None,
):
    """
    Build a modern, native Gradio UI for LLM chat using only Gradio's built-in themes and components.
    Returns a Gradio Blocks app.
    """
    import gradio as gr

    # Ensure the UI uses the current system prompt from the encoder
    # This allows the UI to properly show the correct prompt even after page refresh
    current_system_prompt = getattr(chat_encoder, 'system_prompt', system_prompt)

    # Update the system prompt only if it's not already set in the encoder
    if not hasattr(chat_encoder, 'system_prompt'):
        chat_encoder.update_system_prompt(system_prompt)
    else:
        LOG.info(f"Using existing system prompt from encoder: {current_system_prompt}")

    LOG.info("Building native Gradio LLM UI...")

    # Extract short model name for display
    short_model_name = get_short_model_name(model_name)

    def chat_fn(message, history):
        import time

        if history is None:
            history = []

        # If no_history is enabled, only keep the visual history in the UI
        # but don't pass it to the encoder - treat each message independently
        chat_history = [] if no_history else [(h[0], h[1]) for h in history if h[0] and h[1]]

        # Show user message immediately
        history = history + [[message, ""]]
        yield history, "", "Ready"

        try:
            input_ids, embedding_features = chat_encoder.encode(message, chat_history)
            response = ""
            for new_text, stats in stream_response(
                model_instance,
                chat_encoder,
                tokenizer,
                input_ids,
                embedding_features,
                max_tokens,
                temperature,
                tokenizer.eos_token_id,
                getattr(model_instance, 'end_token_id', None),
            ):
                response = new_text
                history[-1][1] = response
                status_str = f"TTFT: {stats['ttft']:.2f}s | Tokens/sec: {stats['tokens_per_sec']:.2f} | Tokens: {stats['tokens']}"

                if tracers:
                    status_str += format_system_metrics(tracers)

                yield history, "", status_str

            # We need to add the message-response pair to history to make the stateless approach work
            # since we need to properly consume the generated text
            chat_encoder.add_to_history(message, response)
            LOG.info(f"User message processed, response length: {len(response)}")

            # Clear history if no_history flag is set
            if no_history:
                LOG.debug(
                    "--no-history flag enabled, clearing conversation history after response"
                )
                # Clear history but preserve system prompt to avoid unnecessary reprocessing
                chat_encoder.reset(preserve_system_prompt=True)
        except Exception as e:
            # Handle any exceptions that occur during processing
            LOG.error(f"Error processing message: {str(e)}")
            error_message = f"Sorry, an error occurred: {str(e)}"
            history[-1][1] = error_message
            yield history, "", f"<span style='color:red;font-weight:bold;'>Error</span>"

    def clear_history():
        # Reset but preserve the system prompt to avoid unnecessary reprocessing
        chat_encoder.reset(preserve_system_prompt=True)
        LOG.info("Chat history cleared.")
        return [], "", "Ready"

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray")) as demo:
        with gr.Row():
            gr.Image(
                str(ASSET_DIR / "Axelera-AI-logo-White.png"),
                show_label=False,
                height=16,
                elem_id="logo",
            )
        gr.Markdown(
            f"""
        <style>
        @media (prefers-color-scheme: light) {{
            .axelera-title {{ color: #1a237e !important; }}
            .slm-demo {{ color: #0D1223 !important; }}
            .slm-desc {{ color: #444 !important; }}
            .model-name {{ color: #1565c0 !important; font-weight: bold; }}
        }}
        @media (prefers-color-scheme: dark) {{
            .axelera-title {{ color: #FBBE18 !important; }}
            .slm-demo {{ color: #fff !important; }}
            .slm-desc {{ color: #b0b0b0 !important; }}
            .model-name {{ color: #FBBE18 !important; font-weight: bold; }}
        }}
        </style>
        <div style='text-align:center; margin-top: 0.5em; margin-bottom: 0.5em;'>
            <span class='axelera-title' style='font-size: 1.7em; font-weight: bold;'>Axelera AI</span><br>
            <span class='slm-demo' style='font-size: 1.1em;'>SLM Demo</span><br>
            <span class='slm-desc' style='font-size: 1em;'>
                {'Chat with <span class="model-name" style="color:#1565c0;font-weight:bold;">' + short_model_name + '</span> on Metis' if short_model_name != 'Unknown Model' else 'Chat with a Small Language Model on Metis'}
            </span>
        </div>
        """
        )
        gr.Markdown("")  # Spacer
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    chatbot = gr.Chatbot(
                        height=400,
                        bubble_full_width=False,
                        avatar_images=(None, str(ASSET_DIR / "AX-TOP-Icon-.png")),
                    )
        with gr.Row():
            msg = gr.Textbox(placeholder="Type your message...", show_label=False, scale=8)
            clear = gr.ClearButton(value="Clear", scale=1)
        gr.Markdown("")  # Spacer
        with gr.Row():
            status = gr.Markdown(
                "<span style='color:#FBBE18;font-weight:bold;'>Ready</span>", elem_id="status-bar"
            )
        msg.submit(chat_fn, [msg, chatbot], [chatbot, msg, status], queue=True)
        clear.click(clear_history, None, [chatbot, msg, status], queue=False)
        with gr.Accordion("Settings", open=False):
            system_prompt_box = gr.Textbox(
                label="System Prompt", value=current_system_prompt, lines=2
            )
            temp_slider = gr.Slider(
                label="Temperature", minimum=0, maximum=1.5, value=temperature, step=0.05
            )

            def update_settings(new_prompt, new_temp):
                chat_encoder.update_system_prompt(new_prompt)
                nonlocal temperature
                temperature = new_temp
                return new_prompt, new_temp

            system_prompt_box.change(
                update_settings, [system_prompt_box, temp_slider], [system_prompt_box, temp_slider]
            )
            temp_slider.change(
                update_settings, [system_prompt_box, temp_slider], [system_prompt_box, temp_slider]
            )
    return demo
