# Content

This is a simple hands-on workbook to play with AI workflows and agents

I have based the definitions of workflows and agents on an [article](https://www.anthropic.com/research/building-effective-agents) from Anthropic 

Most of the examples about workflows is inspired from this [paper](https://media.licdn.com/dms/document/media/v2/D4D1FAQGI0EiS0GWJXg/feedshare-document-pdf-analyzed/B4DZQ_6TywHYAY-/0/1736239046163?e=1739404800&v=beta&t=LbDOMG3-WhmPp4sw-d8mzu5Rflhkr7N1VZ13wJpyQiA) by Peyman Kor. I have also used output from Claude 3.5 Sonnet as a starting point other examples.

All relevant examples are in the notebook `workflows_and_agents.ipynb`

The notebook is using LLMs from Groq. You will need to get an api-key from Groq and save it as an environment variable called `GROQ_API_KEY`
See https://console.groq.com/login , you can log in directly with GitHub or Google, or make an account by email.


## Setup

### With uv

If you are using uv with python, all you need to do is running

``` bash
uv init
```

### Without uv

Create a python virtual environment

``` bash
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```
