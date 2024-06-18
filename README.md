# Introduction:
Trying to create my first Storytelling bot with Q/A, summarizing capabilities using T5 transformer (for Q/A), sshleifer/distilbart-cnn-12-6 (extractive_summarizer), facebook/bart-large-cnn(abstractive_summarizer) and assess shortcoming and care that I need to take while training the models or using them. <be>

# What does this repo showcase:
This project is an initial draft of my course project in IS 707- Intelligent Technologies at UMBC. Where I was experimenting with how to use and train models by hugging face and trying to create some interesting applications. <br>

The updated version of the project can be found here: https://[github.com/Ilurusheshasai/multiple-LLM-chatbot](https://github.com/Ilurusheshasai/Chatbot-llamaIndex-gemini-RAG) <be> Ihave used rag here.

# Why I did this?
I wanted to leverage the power of various LLMs and orchestrate them together to make them act like a good storyteller for kids as part of coursework. 
I have used multiple LLMs to summarize and perform QA on a story. On top top this I thought this application could just provide an option to display a story if a person wants to read it and also read out the story/ summary in K words the user asked.
  1) Extractive summarizer to retrieve important points from stories.
  2) Abstraction summarizer to rewrite a sentence without changing the meaning.
  3) Question answering to answer questions from the story.
  4) Just speaking out the story (using Python gtts module).

So, I have used 3 LLM from Hugging face. 
  1) sshleifer/distilbart-cnn-12-6 for extractive summarization.
  2) facebook/bart-large-cnn for abstractive summarization.

Although summarizing the stories is working fairly, I wouldn't say I liked the performance of the QA model, for the following reasons.

  1) It gives correct answers only for a few questions.
  2) On top of that, The correct answers are also not convincing because it looks like they give answers giving the exact sentence from the story.
  3) My reason for not liking the performance of this orchestration is if there is an LLM that learned a lot from pre-training before, I do not want to train it again by passing questions, context, and answers to work on my local story base. If I had to train it on question answer and context again, I would choose a simple code with conditions and use TF-IDF to find similarity and answer it. <br>
So, I explored other options and wanted to try RAG using a vector database and knowledge graphs.<br>

The objective was to learn and explore, I might have trained and fine-tuned these models to improve the performance at the same time I looked for easier and better options which led to implementing RAG using LLamaIndex. which I think is a good exposure. Thats it!! dive into and explore and post suggestions if you can give any....

# How to test this?
Just clone this repo, run the final_draft.py with other folders, and just respond to the prompts, you will be able to access all the services of this draft chatbot.

# Key points to remember this is not the final polished chatbot instead it is the starting block. Need to finetune the models for better performance...
# Any suggestions for me to improve chatbot performance are very well appreciated. 
The updated version of the project can be found here: https://[github.com/Ilurusheshasai/multiple-LLM-chatbot](https://github.com/Ilurusheshasai/Chatbot-llamaIndex-gemini-RAG) <br>
