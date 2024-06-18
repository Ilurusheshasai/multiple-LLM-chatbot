# Introduction:
Trying to create my first Storytelling bot with Q/A, summarizing capabilities using T5 transformer (for Q/A), sshleifer/distilbart-cnn-12-6 (extractive_summarizer), facebook/bart-large-cnn(abstractive_summarizer) and assess shortcoming and care that I need to take while training the models or using them. <be>

# What does this repo showcase:
This project is an initial draft of my course project in IS 707- Intelligent Technologies at UMBC. Where I was experimenting with how to use and train models by hugging face and trying to create some interesting applications. <br>

The updated version of the project can be found here: https://[github.com/Ilurusheshasai/multiple-LLM-chatbot](https://github.com/Ilurusheshasai/Chatbot-llamaIndex-gemini-RAG) <be> Ihave used rag here.

# Why did I do this?
I wanted to leverage the power of various LLMs and orchestrate them together to make them act like a good storyteller for kids as part of coursework. 
I have used multiple LLMs to summarize and perform QA on a story. On top top this I thought this application could just provide an option to display a story if a person wants to read it and also read out the story/ summary in K words the user asked.
  1) Extractive summarizer to retrieve important points from stories.
  2) Abstraction summarizer to rewrite a sentence without changing the meaning.
  3) Question answering to answer questions from the story.
  4) Just speaking out the story (using Python gtts module).

# LLMs Used:
  So, I have used 3 LLM from Hugging face. 
    1) sshleifer/distilbart-cnn-12-6 for extractive summarization.
    2) facebook/bart-large-cnn for abstractive summarization.
    3) T5-base transformer for Question answering tasks.
# About Data:
As this is a story-telling Chatbot I started gathering stories that I came across when I was in my school and asked chatGPT to create a few more stories just for fun!!.
Now, I have gathered 5 stories to start with. However, I made sure that I had stories that looked like summaries(third-person narrative) and conversation types. (Why? just to make it tough to get results and make it tough to pass the course!! lol), actually I wanted to test if LLM performs better on any of the mentioned text formats.

# Flow of code/Implementation:
  1) Several chunks of each story are made and passed to extractive summarize first and then abstractive summarize to summarize the story.
  2) Used parallel processing to make this process faster.
  3) These chunks are used because the summarizers can't take the entire story at a time they can only take a finite input.
  4) I have used Tf-Idf to take queries from users ( like "Tell a story about Alexander") and process this query to provide suggestions of the most relevant stories from the database to the user.
  5) Then the user can select a story and this story is pre-processing and indexed. Hypothetically, this needs to be done only once when a user adds a new story to his library. In this code, this happens every time a user selects a story.
  6) While summarizing users can select the length of the summary.
  7) The user can provide a prompt for QA and get answers. Improvement of answers can also be implemented in the future.
  8) As of now all tasks like reading, summarizing, QA, and printing are performed after the user selection of story.  

# My views on this project:
Although summarizing the stories is working fairly, I wouldn't say I liked the performance of the QA model, for the following reasons.

  1) It gives correct answers only for a few questions.
  2) On top of that, The correct answers are also not convincing because it looks like they give answers giving the exact sentence from the story.
  3) My reason for not liking the performance of this orchestration is if there is an LLM that learned a lot from pre-training before, I do not want to train it again by passing questions, context, and answers to work on my local story base. If I had to train it on question answer and context again, I would choose a simple code with conditions and use TF-IDF to find similarity and answer it. <br>
  4) So I just passed the entire story to LLM along with the query and asked it to answer.
  5) This made me think about the case of the LLM performing poorly if the questions asked are from the end of long stories. As there is a chance that they cross the context that LLM can capture and LLM fails to capture the context to answer the question properly. Well, I need to test that.

So, I explored other options and wanted to try RAG using a vector database and knowledge graphs.<br>

The objective was to learn and explore, I might have trained and fine-tuned these models to improve the performance at the same time I looked for easier and better options which led to implementing RAG using LLamaIndex. which I think is a good exposure. Thats it!! dive into and explore and post suggestions if you can give any....

# How to test this?
Just clone this repo, run the final_draft.py with other folders, and just respond to the prompts, you will be able to access all the services of this draft chatbot.

# Key points to remember this is not the final polished chatbot instead it is the starting block. Need to finetune the models for better performance...

# Future work
    - I will train the QA model with Question, answer, and context and compare performance with and without finetuning.
    - try to use the Question and find context using an extractive summarize, pass them to T5 transformer and synthesise answer by passing Question, context to get answer. <br>
    
# Any suggestions for me to improve chatbot performance are very well appreciated. 
The updated version of the project can be found here: https://[github.com/Ilurusheshasai/multiple-LLM-chatbot](https://github.com/Ilurusheshasai/Chatbot-llamaIndex-gemini-RAG) <br>
