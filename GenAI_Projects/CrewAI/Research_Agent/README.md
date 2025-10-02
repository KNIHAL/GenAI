# 🔬 Research Agent

A powerful multi-agent research system built with CrewAI that leverages Google's Gemini AI and Serper for comprehensive research and report generation.

## 🌟 Features

- **Multi-Agent System**: Two specialized AI agents working in coordination
- **Real-time Web Search**: Powered by Serper API for current information
- **Advanced AI**: Uses Google's Gemini 2.0 Flash model for intelligent analysis
- **Interactive Mode**: Continuous research sessions with multiple queries
- **Professional Reports**: Generates detailed markdown reports
- **Training & Testing**: Built-in capabilities for model improvement

## 🏗️ Architecture

### Agents
- **🔍 Researcher**: Senior Data Researcher specialized in finding cutting-edge developments
- **📊 Reporting Analyst**: Creates detailed reports from research findings

### Workflow
1. **Research Phase**: Gather information using web search
2. **Analysis Phase**: Process and structure findings
3. **Report Generation**: Create comprehensive markdown reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CrewAI
- Google Gemini API access
- Serper API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/KNIHAL/Research_Agent.git
cd Research_Agent
```

2. **Install dependencies**
```bash
pip install crewai crewai-tools python-dotenv
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

### Getting API Keys

#### Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

#### Serper API Key
1. Visit [Serper.dev](https://serper.dev/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Copy the key to your `.env` file

## 🎯 Usage

### Interactive Mode
Run the research agent interactively:
```bash
python main.py
```

Enter your research topic when prompted:
```
What would you like to research? AI trends in 2024
```

### Training Mode
Train the model with custom iterations:
```bash
python main.py train <iterations> <filename>
```

Example:
```bash
python main.py train 10 training_data.json
```

### Testing Mode
Test the system performance:
```bash
python main.py test <iterations> <eval_model>
```

Example:
```bash
python main.py test 5 gemini-2.0-flash
```

### Replay Mode
Replay a specific task execution:
```bash
python main.py replay <task_id>
```

## 📁 Project Structure

```
research_agent/
├── src/
│   └── research_agent/
│       ├── config/
│       │   ├── agents.yaml      # Agent configurations
│       │   └── tasks.yaml       # Task definitions
│       ├── crew.py             # Main crew implementation
│       └── main.py             # Entry point and CLI
├── reports/
│   └── report.md              # Generated research reports
├── .env                       # Environment variables
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## ⚙️ Configuration

### Agent Configuration (`agents.yaml`)
- **Researcher**: Configured for web search and data gathering
- **Reporting Analyst**: Specialized in report generation and analysis

### Task Configuration (`tasks.yaml`)
- **Research Task**: Generates 10 bullet points of relevant information
- **Reporting Task**: Expands findings into comprehensive report sections

### Crew Configuration (`crew.py`)
- **LLM**: Gemini 2.0 Flash with temperature 0.5
- **Tools**: SerperDevTool for web search
- **Process**: Sequential execution

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | CrewAI | Multi-agent orchestration |
| **LLM** | Google Gemini 2.0 Flash | AI reasoning and analysis |
| **Search** | Serper API | Real-time web search |
| **Language** | Python 3.8+ | Core implementation |
| **Config** | YAML | Agent and task configuration |

## 📊 Output

The system generates detailed research reports in markdown format, saved as `report.md`. Each report includes:

- **Topic Overview**: Introduction and context
- **Key Findings**: 10 main research points
- **Detailed Analysis**: Expanded sections for each finding
- **Current Information**: Up-to-date data based on search results

## 🔧 Customization

### Modifying Agents
Edit `config/agents.yaml` to customize:
- Agent roles and goals
- Backstories and personalities
- Capabilities and specializations

### Adjusting Tasks
Edit `config/tasks.yaml` to modify:
- Task descriptions and requirements
- Expected outputs and formats
- Agent assignments

### Changing LLM Settings
In `crew.py`, modify:
```python
self.llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.5,  # Adjust for creativity vs consistency
)
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `.env` file exists with correct keys
   - Verify API keys are valid and active

2. **Import Errors**
   - Check all dependencies are installed
   - Verify Python version compatibility

3. **Search Failures**
   - Confirm Serper API key is working
   - Check internet connection

4. **Report Generation Issues**
   - Ensure write permissions in project directory
   - Check disk space availability

## 📈 Performance Tips

- **API Limits**: Monitor your Gemini and Serper API usage
- **Temperature Settings**: Lower values (0.1-0.3) for factual research, higher (0.7-0.9) for creative analysis
- **Batch Processing**: Use training mode for multiple similar queries
- **Caching**: Store frequently accessed information locally

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [CrewAI](https://crewai.com/) for the multi-agent framework
- [Google](https://ai.google.dev/) for Gemini AI capabilities
- [Serper](https://serper.dev/) for web search functionality

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Built with ❤️ using CrewAI, Gemini AI, and Serper**
