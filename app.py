import gradio as gr
from transformers import pipeline
import random
import nltk

nltk.download('punkt')

# Dictionary that maps the user-friendly model names to their actual names
model_names = {
    "BERT base": "google-bert/bert-base-cased",
    "DistilBERT base": "distilbert/distilbert-base-cased",
    "RoBERTa base": "FacebookAI/roberta-base",
    "BERT finetuned on a dataset for mask filling": "emma7897/bert_one",
    "DistilBERT finetuned on a dataset for mask filling": "emma7897/distilbert_one",
    "BERT finetuned on a dataset of stories for children": "emma7897/bert_two",
    "DistilBERT finetuned on a dataset of stories for children": "emma7897/distilbert_two",
}

sample_paragraphs = [
    "Once upon a time, in a faraway land, there lived a beautiful princess named [MASK]. She was known throughout the kingdom for her [MASK] and immense bravery. One day, while exploring the large forest, she stumbled upon a [MASK] hidden amongst the trees. Curiosity piqued, she ventured inside and discovered a [MASK] filled with treasures beyond imagination. Little did she know, her adventures were just beginning.",
    "In the city of [MASK], where the streets were always very crowded and the skyscrapers reached for the sky, there was a tall detective named Sam. With a keen eye for detail and a knack for solving mysteries, Sam was the best in the business. When horrific crime shook the city to its core, Sam was called to travel to [MASK]. With determination and a trusty [MASK] by his side, Sam set out to uncover the truth.",
    "On a remote island in the middle of the [MASK], there stood a blue lighthouse overlooking the turbulent waters. Inside, a keeper tended to the beacon, guiding [MASK] safely to shore. One stormy night, as the waves crashed against the rocks and the wind howled through the [MASK], a ship appeared on the horizon, its sails tattered and its crew in desperate need of help. With nerves of [MASK] and a steady hand, the lighthouse keeper sprang into action, signaling the way to safety.",
    "In a whimsical village nestled in the [MASK] countryside, there lived an inventor named Zoey. Day and night, Zoey toiled away in her workshop, creating [MASK] that defied imagination. There was no limit to Zoey's creativity. But when a problem threatened to disrupt the peace of the village, Zoey knew it was time to put her [MASK] to the test. With gears whirring and steam hissing, Zoey set out to save the day.",
    "Meet Emma, a spirited young soul with [MASK] dreams. Emma's eyes sparkle with determination as she envisions herself soaring among the stars as an aspiring [MASK]. She spends her days devouring books about [MASK]. When Emma is not gazing at the stars, you can find her drawing pictures of [MASK].",
    "Hello! I would like to introduce you to my best friend, [MASK]."
]

example_models = [
    "BERT base",
    "DistilBERT base",
    "RoBERTa base",
    "BERT finetuned on a dataset for mask filling",
    "DistilBERT finetuned on a dataset for mask filling",
    "BERT finetuned on a dataset of stories for children",
    "DistilBERT finetuned on a dataset of stories for children",
]

# Create a nested list for the examples
examples = [[random.choice(example_models), paragraph] for paragraph in sample_paragraphs]

def textGenerator(model, userInput):
    model_name = model_names[model]
    fill_mask = pipeline("fill-mask", model=model_name)
    sentences = nltk.sent_tokenize(userInput)
    processed_sentences = []
    if model_name != "FacebookAI/roberta-base":
        for sentence in sentences:
            while "[MASK]" in sentence:
                predictions = fill_mask(sentence, top_k=10)
                token_strings = []
                for prediction in predictions:
                    token_strings.append(prediction['token_str'])
                selected_token = random.choice(token_strings)
                sentence = sentence.replace("[MASK]", f"<mark>{selected_token}</mark>", 1)
            processed_sentences.append(sentence)
        processedText = " ".join(processed_sentences)
    if model_name == "FacebookAI/roberta-base":
        for sentence in sentences:
            while "[MASK]" in sentence:
                sentence = sentence.replace("[MASK]", "<mask>", 1)
                predictions = fill_mask(sentence, top_k=10)
                token_strings = []
                for prediction in predictions:
                    token_strings.append(prediction['token_str'])
                selected_token = random.choice(token_strings).strip()
                sentence = sentence.replace("<mask>", f"<mark>{selected_token}</mark>", 1)
            processed_sentences.append(sentence)
        processedText = " ".join(processed_sentences)
    return processedText

screen = gr.Interface(fn=textGenerator, inputs=[
        gr.Radio(list(model_names.keys()), label="LLM", info="Which LLM would you like to use?"),
        gr.Textbox(label = "User Input", info="Please enter a paragraph. Replace words that you want the LLM to fill in with [MASK]. Note: there is a limit of one [MASK] per sentence."),
    ], outputs = gr.HTML(label = "Processed Text"),
    examples = examples,
    )

if __name__ == "__main__":
    screen.launch()