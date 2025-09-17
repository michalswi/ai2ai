## ai2ai

![](https://img.shields.io/github/issues/michalswi/ai2ai)
![](https://img.shields.io/github/forks/michalswi/ai2ai)
![](https://img.shields.io/github/stars/michalswi/ai2ai)
![](https://img.shields.io/github/last-commit/michalswi/ai2ai)

GO app to chat with ChatGPT (OpenAI) and Grok (X) simultaneously.  

**Two options to run:**
```
> interactive discussion with AI models
$ go run .

> automatic discussion between AI models
$ go run . -auto
```
There is env variable **AI_TURNS** (default set to 3) to limit the number of responses from the AI models. 

For example `AI_TURNS=2 ./ai2ai` .

![example](./img/example.png)
