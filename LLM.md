# <center>A²PI²-LLM version2.2</center>

* 包含大语言模型(LLM)开发的软件包.

# 1.langchain

| 版本   | 描述                     | 注意 | 适配Apple芯片 |
| ------ | ------------------------ | ---- | ------------- |
| 0.3.25 | 通过可组合性开发LLM应用. | -    | 是            |

## 1.1.prompts

### 1.1.1.ChatPromptTemplate

#### 1.1.1.1.format_messages()

将聊天模版格式化为最终的消息列表.|`list`

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_template(template='你是一个擅长翻译的AI.\n请帮我翻译: {text}')
prompt = template.format_messages(text='Hello, World!')  # 用于填充此聊天模版的全部模版消息中的模版变量的关键字参数.
```

#### 1.1.1.2.from_template()

从字符串创建聊天提示模版.|`langchain_core.prompts.chat.ChatPromptTemplate`

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_template(template='你是一个擅长翻译的AI.\n请帮我翻译: {text}')  # str|模版字符串.
```

## 1.2.text_splitter

### 1.2.1.RecursiveCharacterTextSplitter()

实例化递归字符文本分割器.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000,  # int(可选)|4000|分块的最大长度.
                                               chunk_overlap=200,  # int(可选)|200|分块之间重叠的字符数.
                                               separators=None,  # list of str(可选)|None|预定义的分隔符.
                                               keep_separator=True)  # bool(可选)|True|是否保留分隔符及其在每个对应分块中的位置.
```

#### 1.2.1.1.split_text()

根据预定义的分隔符将输入文本分割成更小的块.|`list of str`

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10,
                                               chunk_overlap=10,
                                               separators=['\n\n', '\n'],
                                               keep_separator=False)
chunks = text_splitter.split_text(text='Hello, World!\n根据预定义的分隔符将输入文本分割成更小的块.')  # str|待分割的输入文本.
```

# 2.langchain_community

| 版本   | 描述                         | 注意 | 适配Apple芯片 |
| ------ | ---------------------------- | ---- | ------------- |
| 0.3.24 | 社区贡献的LangChain集成组件. | -    | 是            |

## 2.1.utilities

| 版本 | 描述                                    | 注意 |
| ---- | --------------------------------------- | ---- |
| -    | 工具集是与第三方系统及软件包的集成组件. | -    |

### 2.1.1.SQLDatabase

#### 2.1.1.1.from_uri()

从URI构建SQLAlchemy引擎.|`langchain_community.utilities.sql_database.SQLDatabase`

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(database_uri='sqlite:///databasex.db')  # str|数据库的URI.
```

##### 2.1.1.1.1.table_info

数据库中所有表的信息.|`str`

```python
db.table_info
```

# 3.openai

| 版本   | 描述                      | 注意 | 适配Apple芯片 |
| ------ | ------------------------- | ---- | ------------- |
| 1.84.0 | OpenAI API的官方Python库. | -    | 是            |

## 3.1.OpenAI

创建一个新的同步OpenAI客户端实例.

```python
from openai import OpenAI

client = OpenAI(api_key='ollama',  # str|None|API访问令牌.
                base_url='http://localhost:11434/v1/')  # str or httpx.URL or None|None|API基础地址.
```

### 3.1.1.chat

#### 3.1.1.1.completions

##### 3.1.1.1.1.create()

为给定的聊天对话创建模型响应|`openai.types.chat.chat_completion.ChatCompletion`.

```python
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1/')
response = client.chat.completions.create(
    messages=[ChatCompletionUserMessageParam(content='圆周率是多少?', role='user')],  # list or ChatCompletionMessageParam|包含到目前为止的对话的消息列表.
    model='qwen3:0.6b',  # str or ChatModel|生成响应所使用的模型ID.
    extra_body=None  # object|None|向请求中添加的额外的JSON属性.
)
```

## 3.2.types

### 3.2.1.chat

#### 3.2.1.1.ChatCompletionUserMessageParam

聊天对话用户消息参数.|`dict`

```python
from openai.types.chat import ChatCompletionUserMessageParam

message = ChatCompletionUserMessageParam(content='圆周率是多少?',  # str|用户消息的内容.
                                         role='user')  # str|消息作者的角色.
```

