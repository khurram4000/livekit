from pathlib import Path
from openai import OpenAI

client = OpenAI()
speech_file_path = Path(__file__).parent / "shimmer.mp3"

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="shimmer",
    input="LiveAdmins, where innovation meets excellence in online customer engagement. With our trained live chat agents, cutting-edge technology and AI, we are dedicated to delivering unparalleled, 24/7, personalized experiences for your website visitors.",
    instructions=(
        "Speak with a warm, cheerful, and energetic tone. "
        "Add natural pauses, slight ups and downs in pitch for emphasis, "
        "and vary the speed slightly to match how a passionate person would speak. "
        "Imagine you're genuinely excited and inspiring a visitor. "
        "Sound as human and engaging as possible—like a TED talk speaker connecting emotionally with the audience."
    ),
) as response:
    response.stream_to_file(speech_file_path)