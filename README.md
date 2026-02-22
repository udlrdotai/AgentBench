# AgentBench

面向超级个体的 AI Agent 评测平台。

AgentBench 是一个 AI Agent 工具评测平台（Benchmark），核心关注 **模型与其配合的 Agent 框架/工具链的整体产出效果**，帮助用户快速判断"哪个模型 + 哪个 Agent 工具组合"在实际场景中表现最好。

## 评测领域

围绕"超级个体"的四大高价值领域进行评测：

- **代码开发** — 编码、调试、架构设计、项目搭建等端到端开发任务
- **自媒体** — 文案撰写、内容策划、多平台适配、SEO 优化等内容生产任务
- **期权投资** — 市场分析、策略生成、风险评估、数据解读等投资辅助任务
- **个体健康** — 健康数据解读、运动计划、饮食建议、症状初筛等健康管理任务

## 项目结构

```
agentbench/
├── config.py              # 配置与模型注册表
├── runner.py              # 评测主运行器
├── models/                # 模型适配器
│   ├── base.py            # 模型抽象基类
│   ├── openai_model.py    # OpenAI (GPT) 适配器
│   └── anthropic_model.py # Anthropic (Claude) 适配器
├── tasks/                 # 任务管理
│   ├── base.py            # Task 数据结构
│   └── loader.py          # JSON 任务加载器
└── evaluators/            # 评测打分
    ├── base.py            # 评估器抽象基类
    └── llm_judge.py       # LLM-as-Judge 评估器
benchmarks/
└── text_generation.json   # 文本生成基准测试集
run.py                     # CLI 入口
```

## 快速开始

### 环境要求

- Python 3.10+

### 安装

```bash
git clone https://github.com/aiturnright/AgentBench.git
cd AgentBench
pip install -e .
```

### 配置

复制环境变量模板并填入 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key
```

### 运行评测

```bash
# 使用 OpenAI 模型运行评测（默认）
python run.py

# 同时评测 OpenAI 和 Anthropic 模型
python run.py --models openai,anthropic

# 指定任务集和输出路径
python run.py --models openai,anthropic --tasks benchmarks/text_generation.json --output results/output.json

# 指定 LLM 评审模型
python run.py --models openai --judge gpt-4o-mini
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--models` | `openai` | 逗号分隔的模型名称，可选：`openai`、`anthropic` |
| `--tasks` | `benchmarks/text_generation.json` | 基准测试 JSON 文件路径 |
| `--output` | `results/output.json` | 结果输出 JSON 文件路径 |
| `--judge` | `gpt-4o-mini` | LLM 评审使用的模型 ID |

## 工作原理

1. 从 JSON 文件加载评测任务
2. 初始化 LLM 评审模型（默认使用 GPT-4o-mini）
3. 初始化被评测的模型（OpenAI / Anthropic）
4. 对每个模型、每个任务：
   - 模型根据 prompt 生成输出
   - LLM 评审根据评分标准打分（1-10 分）
5. 输出评测报告并保存结果

## 支持的模型

| 名称 | 模型 | 说明 |
|------|------|------|
| `openai` | gpt-4o-mini | OpenAI Chat Completion API |
| `anthropic` | claude-sonnet-4-20250514 | Anthropic Messages API |

## 基准测试集

当前包含 `text_generation.json`，涵盖 4 个领域共 6 个评测任务：

| 任务 ID | 领域 | 内容 |
|---------|------|------|
| code-doc-1 | 代码开发 | 编写 Python 重试装饰器 API 文档 |
| code-doc-2 | 代码开发 | 编写 Flask 限流中间件技术设计文档 |
| media-post-1 | 自媒体 | 撰写 AI 与效率主题公众号文章 |
| options-report-1 | 期权投资 | 编写期权策略分析报告 |
| health-plan-1 | 个体健康 | 制定 4 周健身计划 |
| health-diet-1 | 个体健康 | 制定力量训练配套饮食方案 |

### 自定义任务

任务文件为 JSON 格式，结构如下：

```json
{
  "tasks": [
    {
      "id": "task-id",
      "domain": "领域名称",
      "capability": "能力维度",
      "prompt": "评测提示词",
      "criteria": "评分标准"
    }
  ]
}
```

## 扩展模型

实现 `BaseModel` 抽象类即可添加新的模型适配器：

```python
from agentbench.models.base import BaseModel

class MyModel(BaseModel):
    @property
    def name(self) -> str:
        return "my-model"

    def generate(self, prompt: str) -> str:
        # 调用模型 API 并返回生成结果
        ...
```

然后在 `agentbench/config.py` 的 `MODEL_REGISTRY` 中注册：

```python
MODEL_REGISTRY = {
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
    "my-model": MyModel,
}
```

## 技术栈

- **Python 3.10+**
- **OpenAI SDK** (`openai>=1.0`) — GPT 模型调用
- **Anthropic SDK** (`anthropic>=0.20`) — Claude 模型调用
- **python-dotenv** (`python-dotenv>=1.0`) — 环境变量管理

## 许可证

本项目为私有项目。
