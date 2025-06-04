from openai import OpenAI

client = OpenAI(
    # 将这里换成你在便携AI聚合API后台生成的令牌
    api_key="sk-9IAkf9j09twKnscijrcO03ApuOOiLkRHs999CltJZWi0UEEQ",
    # 这里将官方的接口访问地址替换成便携AI聚合API的入口地址
    base_url="https://api.bianxieai.com/v1/"
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Who are you",
        }
    ]
)

print(completion.choices[0].message)