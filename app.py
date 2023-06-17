from typing import Dict, Any

import gradio as gr
import torch
from transformers import pipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)


class ConversationBufferMem(ConversationBufferWindowMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save the context to the buffer.

        Args:
            inputs: The input parameters.
            outputs: The outputs of the model.

        Returns:
            None.
        """
        super(ConversationBufferMem, self).save_context(inputs, {'response': outputs['result']})




generate_text = pipeline(model="databricks/dolly-v2-3b", 
                         torch_dtype=torch.bfloat16, 
                         trust_remote_code=True,
                         device_map="auto",
                         return_full_text=True)

prompt = PromptTemplate(input_variables=["instruction"],template="{instruction}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)

model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)

persist_directory = 'recipedb'
vectordb = Chroma(persist_directory=persist_directory,
                   embedding_function=hf)
vectordb.get()
retriever = vectordb.as_retriever(search_kwargs={'k':3})

memory = ConversationBufferMem(k=3)
qa_chain_with_memory = RetrievalQA.from_chain_type(llm=hf_pipeline,
                                                   chain_type="stuff",
                                                   retriever=retriever,
                                                   return_source_documents=True,
                                                   memory=memory)


template = '''
     You are the assistant to a chef. You have a deep knowledge of cooking techniques and ingredients. You can provide specific details about recipes, using the context given and the user's question. 
     If you don't know the answer, you truthfully say you don't know and don't try to make up an answer.
    ----------------
    {context}

    Question: {question}
    Helpful Answer:'''

qa_chain_with_memory.combine_documents_chain.llm_chain.prompt.template = template


examples = [
        "Instead of making a peanut butter and jelly sandwich, what else could I combine peanut butter with in a sandwich? Give five ideas",
        "How do i prepare egg salaad?",
        "what is the rcepe for fruit salaad",
         "I'm looking for a recipe for a vegan chili"
    ]

def process_example(args):
    for x in generate(args):
        pass
    return x

def clean_data(text):
    cleaned_data = ' '.join(text.split())
    #new line for each step
    recipe_steps = cleaned_data.replace('\n\n', '\n')
    return recipe_steps

def get_response(llm_response):
    # Get the cleaned text
    cleaned_data = clean_data(llm_response['result'])
    return cleaned_data



def generate(instruction):
    response = qa_chain_with_memory(instruction)
    processed_data = get_response(response)
    result = ""
    for data in processed_data.split(" "):
        result += data + " "
        yield result


css = ".generating {visibility: hidden}"

with gr.Blocks(theme=theme, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(
            """ ##  👨‍🍳Dolly Chef
            Dolly Chef is an advanced food assistant cooking recipe bot powered by the Dolly-v2 model. It is designed to provide users with a seamless and interactive cooking experience. Whether you're a novice cook or an experienced chef, Dolly Chef is here to assist you 
            in preparing delicious meals. For more details, please refer to the [model card](https://huggingface.co/databricks/dolly-v2-12b)

            Type in the box below and click the button to generate answers to your most pressing questions!


      """
        )
        gr.HTML(
            "<p>The data use to build this llm chatbot was RecipeNLG (cooking recipes dataset) from kaggle , you can check itout here. : <a style='display:inline-block' href='https://www.kaggle.com/datasets/paultimothymooney/recipenlg'></a> </p>")

        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question", elem_id="q-input")

                with gr.Box():
                    gr.Markdown("**Answer**")
                    output = gr.Markdown(elem_id="q-output")
                submit = gr.Button("Generate", variant="primary")
                gr.Examples(
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )

    submit.click(generate, inputs=[instruction], outputs=[output])
    instruction.submit(generate, inputs=[instruction], outputs=[output])

demo.queue(concurrency_count=16).launch(debug=True)
demo.launch()
