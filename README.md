Trying to create my first Storytelling bot with Q/A, summarizing capabilities using T5 transformer (for Q/A), sshleifer/distilbart-cnn-12-6 (extractive_summarizer), facebook/bart-large-cnn(abstractive_summarizer) and assess shortcoming and care that I need to take while training the models or using them. <br>

This project is an initial draft of my course project in IS 707- Intelligent Technologies at UMBC. Where I was experimenting with how to train models to hugging face and trying to create some interesting applications. <br>

The updated version of the project can be found here: https://[github.com/Ilurusheshasai/multiple-LLM-chatbot](https://github.com/Ilurusheshasai/Chatbot-llamaIndex-gemini-RAG) <br>

I have used multiple LLMs to empower chatbots with human-like capabilities in summarizing (extractive (to retrieve important points from stories) and abstraction(to rewrite a sentence without changing the meaning)), Question answering, speaking out the story (using Python gtts module).

So, I have used 3 models and ran them locally.
Although summarizing the stories is working fairly, I wouldn't say I liked the performance of the QA model, for the following reasons.

It gives correct answers only for a few questions.
On top of that, The correct answers are also not convincing because it looks like they give answers giving the exact sentence from the story.
My argument is if there is an LLM that learned a lot from pre-training before, I do not want to train it again by passing questions, context, and answers to work on my local story base. If I had to train it on question answer and context again, I would choose a simple conditional code with TF-IDF to find similarity and answer it.
So, I explored other options and wanted to try RAG using a vector database and knowledge graphs.
This repo is an implementation of a RAG-based vector store to leverage the best output from LLM (Google Gemini pro model) in this case...
The objective was to learn and explore, I might have trained and fine-tuned these models to improve the performance at the same time I looked for easier and better options which led to implementing RAG using LLamaIndex. which I think is a good exposure. Thats it!! dive into and explore and post suggestions if you can give any....
The updated version of the project can be found here: https://[github.com/Ilurusheshasai/multiple-LLM-chatbot](https://github.com/Ilurusheshasai/Chatbot-llamaIndex-gemini-RAG) <br>

# How to test this?
Just clone this repo, run the final_draft.py with other folders, and just respond to the prompts, you will be able to access all the services of this draft chatbot.

# Key points to remember this is not the final polished chatbot instead it is the starting block. Need to finetune the models for better performance...
# Any suggestions for me to improve chatbot performance are very well appreciated.  
