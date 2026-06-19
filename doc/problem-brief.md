# Problem Brief

## Source Documents

- `Brainrot_Image_Gen.pdf` - read. HW6 assignment statement, pages 1-5.
- User instruction in this conversation - read. The user intends to start from scratch.

## Assignment Objective

Train a conditional image generation model from scratch for Brainrot images formed from animal-object pairs. Use the trained model to generate 2,000 `64x64` Brainrot images conditioned on the rows in `generate.csv`.

The generated images are evaluated against a hidden test set using Fréchet Inception Distance (FID) and CLIP-T.

## Required Inputs

- Brainrot Dataset training data, including `train.csv` with columns:

```text
id,animal,object
000001.png,fish,chair
...
```

- Training images corresponding to the IDs in `train.csv`.
- `generate.csv` with columns:

```text
id,animal,object,prompt
000001.png,shark,sneaker,a shark and a sneaker
...
```

- Codabench-provided files, including reference FID statistics `test_mu.npy` and `test_sigma.npy`.
- Animal categories stated in the PDF: shark, crocodile, frog, cat, dog, capybara, elephant, bird, fish, monkey.
- Object categories stated in the PDF: sneaker, airplane, coffee cup, banana, cactus, toilet, pizza, drum, car, chair.

## Required Outputs

- Exactly 2,000 generated PNG images.
- Each output image must be RGB, `64x64`, and named according to `generate.csv`.
- Images must be generated using the corresponding ID and condition from `generate.csv`.
- Codabench upload output: a zip containing the contents of `generated_images/`.
- E3 submission package named `HW6_{student_id}.zip`.

## Constraints

- The main generator must be a conditional image generation model trained from scratch.
- The generator itself must not use pretrained weights, including pretrained UNet, Transformer, diffusion model, or other generative model weights.
- Students must implement the model architecture and training flow themselves, including components such as backbone, scheduler, loss function, training loop, and sampling procedure.
- Diffusers and other high-level generation pipelines or training flows must not be used directly.
- Pretrained VAE or pretrained CLIP may be used only as auxiliary modules, such as for latent representation, feature extraction, or evaluation.
- Auxiliary pretrained modules must not replace the main generator and must not directly generate the final images.
- The assignment allows extra training data beyond Brainrot Dataset, but does not guarantee score improvement.
- The training process and trained model must be reproducible by the TAs.
- Brainrot Dataset is for course use only and must not be repurposed.
- Codabench has a daily upload limit of 3 submissions.

## Evaluation or Grading Criteria

- Evaluation platform: Codabench.
- Metrics:
  - FID: lower is better; worth 50% of the grade.
  - CLIP-T: higher is better; worth 50% of the grade.
- Hidden test set: 3,000 images, 30 images for each animal-object pair.
- `generated_images/` is evaluated against the hidden test set.
- FID uses the same method as the provided `test_mu.npy` and `test_sigma.npy` reference statistics.
- CLIP-T compares generated images to the corresponding `generate.csv` text prompts using OpenAI CLIP ViT-B-32-quickgelu.
- FID thresholds:
  - `<= 49.2545`: 100%
  - `<= 58.0755`: 90%
  - `<= 75.0642`: 80%
  - `<= 90.0142`: 70%
  - otherwise: 0%
- CLIP-T thresholds:
  - `>= 0.2703`: 100%
  - `>= 0.2618`: 90%
  - `>= 0.2536`: 80%
  - `>= 0.2170`: 70%
  - otherwise: 0%
- Penalties:
  - Late submission: one-week late window only, final score multiplied by 0.7.
  - Incorrect file format: minus 10 points.
  - Plagiarism: copier gets 0; copied-from student loses 10 points.
  - Non-reproducible result: 0.
  - Assignment rule violation: 0.
  - Not participating in the competition: 0.

## Required Deliverables

- Codabench submission: zipped `generated_images/` containing 2,000 PNG images.
- E3 submission: `HW6_{student_id}.zip`.
- Example E3 package structure from the PDF:

```text
HW6_{student_id}/
├── generated_images/
├── scripts/
├── model.pth
├── README.md
└── requirements.txt
```

- `generated_images/`: 2,000 `64x64` PNG images.
- `scripts/`: training, generation, and related configuration/code.
- `model.pth`: trained model weights used to generate the images.
- `README.md`: training method, environment setup, and generation commands for TA reproduction.
- `requirements.txt`: Python packages required for training and generation.
- If extra data is used or the model weights exceed E3 upload limits, provide cloud links in the submission materials.

## Relevant Methods From Papers

None found in the provided sources.

## Data, Benchmarks, or Test Cases

- Brainrot Dataset contains 10 animal categories and 10 object categories, for 100 animal-object combinations.
- Training set: 4,799 images.
- Test set: hidden, 3,000 images, 30 per animal-object pair.
- Generation target: 2,000 images, 20 per animal-object pair.
- Example prompts follow the format `a {animal} and a {object}`.
- Baseline models listed in the PDF use approximately 62.68M parameters and RTX 4070 12GB VRAM, but detailed baseline methods are not disclosed before the deadline.

## Implementation Environment

- The PDF does not require a specific programming language or framework.
- The PDF requires a `requirements.txt` in the E3 package.
- Baseline training environment shown in the PDF: RTX 4070 with 12GB VRAM.
- Codabench is the official evaluation platform.
- E3 is the official package submission platform.

## Confirmed Facts

- The assignment is HW6 Brainrot Image Generation.
- The model must be a conditional image generator trained from scratch.
- The final generated output must contain exactly 2,000 `64x64` PNG images.
- The requested conditions come from `generate.csv`.
- Evaluation uses FID and CLIP-T with equal grading weight.
- Pretrained generative weights and high-level generation pipelines are forbidden for the main generator.
- Pretrained CLIP or VAE may be used only as auxiliary modules.
- Reproducibility by the TAs is required.
- The user wants to start the repository implementation from scratch.

## Assumptions

- The repository-local `dataset/` contents match the assignment files described in the PDF.
- The repository-local `hw6_reference/` contents correspond to the Codabench-provided FID reference statistics.
- The student ID will be supplied later for final packaging.
- The final implementation can use Python because the required package includes `requirements.txt`, but the PDF itself does not mandate Python.

## Open Questions

- What is the exact E3 due date? The PDF says to follow the assignment instructions, but the date is not included in the provided text.
- What is the student's ID for naming the final package?
- Will the final solution use only Brainrot Dataset, or also allowed extra data?
- Will any auxiliary pretrained CLIP or VAE be used? If so, the usage must remain within the PDF's restrictions and be documented for reproducibility.
- Are there course-specific Codabench instructions from HW5 that affect file naming, leaderboard display, or upload procedure?
- What hardware and training time budget are available for the scratch model?

## Notes for Proposal Generation

- Start from the confirmed assignment facts and open questions above.
- The proposal may choose an architecture, training loop, sampler, and validation workflow, but those choices are not specified by the PDF.
- Preserve the scratch-training restriction and avoid any proposal that depends on pretrained generative weights or Diffusers-style pipelines.
- Include validation for image count, filenames, PNG format, RGB mode, and `64x64` size because the PDF warns that incorrect quantity or format affects scoring.
- Include reproducibility steps in the proposal because TA reproduction is a grading requirement.
