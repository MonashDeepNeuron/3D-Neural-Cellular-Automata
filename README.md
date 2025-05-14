# Neural-Cellular-Automata

## Goals

Our goals can be broken down into answering three topics:

1. What are NCA? How is NCA different from other NCA?
2. What can NCA be used for? Does NCA provide an advantage over other similar architectures?
3. How can NCA be improved?

As a result of answering these questions, we aim to produce a research paper.

## Open Source Website!

All of our code for <a href="https://neuralca.org">our website</a> and for generating website content is open source and available here! There are two main parts:

1. **The Training Code:** This produces the weights an biases of our neural network. All implemented in **PyTorch**. There will be multiple versions of this for various experiments.
2. **The Website:** Models are rendered in **WebGPU** and **JavaScript.** Output is what you see in front of you!

## Current Progress

### Website Construction
- [x] Learning WebGPU and creating a platform to share our work
  - [x] Built Conway's game of life
  - [x] Built Life-like CA, capable of taking input changing the behaviour of the CA
  - [x] Built Larger than Life, capable of creating coloured outputs 
  - [x] Built Continuous, implementing convolutions and live-editable WebGPU code
  - [x] Built Growing Neural Cellular Automata
    - [x] Developed and trained model
    - [x] Ported model weights over to the website in readable format
    - [ ] Implemented Growing Neural Cellular Automata on website
- [ ] Built the about page
- [x] Created basic descriptions of Neural Cellular Automata
- [ ] Created advanced resources on Neural Cellular Automata

### Model Development
- [x] Implement our own copy of Growing Neural Cellular Automata in PyTorch
  - [x] Implement model
  - [x] Implement Growing training script
  - [x] Implement Persisting training script
  - [ ] Implement Regenerating training script
- [x] Create system to weights and convert to format usable for website
- [x] Explore the viability of extending NCA for Image Segmentation
  - [x] Implementation of the paper Image Segmentation Neural Cellular Automata
  - [x] Implementation of the paper Med-NCA
- [x] Explore the viability of extending NCA for texture generation
  - [x] Implementation of Self Organising Textures

## Resources

This project is currently using the Notion platform to document project progress and important information. This Notion workspace will be made public at a later point in time.

However, some useful resources for this project include:
- Understanding Cellular Automata (CA)
  - Introduction to [Conway's Game of Life](https://playgameoflife.com/)
  - [Explaining CA](https://natureofcode.com/book/chapter-7-cellular-automata/)
  - What are ["Life-Like" CAs](https://en.m.wikipedia.org/wiki/Life-like_cellular_automaton#cite_note-23)
- Neural Cellular Automata
  - [Neural Patterns](https://neuralpatterns.io)
  - [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- Tutorials for Building CAs
  - [Daniel Shiffman's Tutorial](https://www.youtube.com/watch?app=desktop&v=DKGodqDs9sA)
  - [Building Conway's Game of Life using WebGPU](https://codelabs.developers.google.com/your-first-webgpu-app#0)
  - [Physics Simulation with CA](https://www.youtube.com/watch?v=VLZjd_Y1gJ8&pp=ygUfY2VsbHVsYXIgYXV0b21hdGEgc2FuZCBwYXJ0aWNsZQ%3D%3D)
- Other
  - [Noita](https://en.wikipedia.org/wiki/Noita_(video_game)#cite_note-11) uses CA to make physics simulation
  - ["Rule-String" Notation](https://conwaylife.com/wiki/Rulestring)
  - [CAs and Computational Systems](https://direct.mit.edu/isal/proceedings/isal2021/33/105/102949)
  - [Emergent Gardens](https://www.youtube.com/@EmergentGarden)
- WebGPU Resources
  - [WebGPU Shader Tips](https://toji.dev/webgpu-best-practices/dynamic-shader-construction.html)
  - [Typescript & WebGPU Examples](https://webgpu.github.io/webgpu-samples/samples/helloTriangle)

More resources will be made available in this repo as the project progresses.
