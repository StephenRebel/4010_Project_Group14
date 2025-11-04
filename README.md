# COMP4010-FinalProject-RLMusicBot
This is a repository containing the code implementation for a reinforcement learning agent to generate simple musical compositions

# Motivation

There are many applications of Reinforcement Learning (RL) to a multitude of domains and music is no different. Many projects focus on the application of RL to refine existing generative neural networks, yet we find there to be limited work addressing the process of training an agent from scratch to generate meaningful and musically desirable compositions. Through this we hope to deepen our knowledge of RL techniques and musical generation, and develop an interesting solution to contribute to existing methodologies.

# Problem Statement

We address the problem of training a Reinforcement Learning Agent to produce short musical compositions from scratch. To accomplish this we will design a Gymnasium environment to train an RL agent by generating musical compositions and returning feedback through a reward computed over the result. We will compare multiple different RL methods (Deep Q Networks, ) and compare the outcomes. This agent will be trained from scratch learning entirely from its experience in our environment and guided by our reward function. Thus it is pivotal to produce a meaningful and useful result to design an effective reward function

# Steps to Run

1. Clone the repository onto your local machine

```bash
git clone https://github.com/StephenRebel/4010_Project_Group14.git
```

2. Create the virtual environment and activate

```bash
py -3.11.9 -m venv .venv

# Windows
.venv\Scripts\activate

# Linux and MacOS
source .venv/bin/activate
```

3. Install pip requirements

```bash
pip install -r requirements
```

4. Running python script

```bash
python *runnable_script*.py
```

5. Final visualization code (optional)

To run the visualization part of the environment and to play the resulting composition you will need to install MuseScore 4.

This webpage gives some details on installing the program: https://musescore.org/en/4.0

Run the following command in a python interactive repl to configure the necessary paths:

```python
import music21
music21.mainConfigure()
```
