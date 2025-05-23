# synthetic_fs_generator.py
import ollama
import time
import json
import os
FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..', '..')
DATA_PATH=os.path.join(CODE_PATH, 'data')
PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')


NUM_SAMPLES = 20  # Number of synthetic file systems to generate
MODEL = 'llama3.1:8b'   # Name of your local model (verify with `ollama list`)
SLEEP_TIME = 1    # Delay between requests to avoid overloading local model
persona_prompt = """Imagine a random person and add specifications about how their file system might differ. Keep within 3 sentences."""

tree_prompt_template = """Create a Linux file system structure starting from ~ for this user:
{persona}


Output ONLY:
<START>
File structure with the number of tabs representing it's depth
<END>

Example of the Format:
<START>
~
    University
        CS886
            final_presentation.ptx
        Career
            Resume.pdf
            Portfolio.txt
        Projects
            Impact of AI
                Project 1 - Concept Art
                    Thumbnail.png
                    Sketches
                Final.m4a
            Python Project
                main.py
                requirements.txt
                README.md  
    Music
        Classical
            Sonata in G minor.mp3
<END>
"""

# tree_prompt_template = """Create a Linux file system structure starting from ~ for this user: {persona}.
# Only output the file system in this format:
# ~(.bashrc, .profile, Pictures (Bali, macOS screenshots), Documents (Projects (CS886 (final_presentation.ptx)), Resume.pdf), Music, Videos)
# where '(' and ')' are used to denote directory contents.
# """
def generate_sample():
    # Generate persona
    persona_response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": persona_prompt}]
    )
    persona = persona_response['message']['content'].strip()
    
    # Generate file tree
    tree_response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": tree_prompt_template.format(persona=persona)}]
    )
    tree = tree_response['message']['content'].strip()
    
    return {"persona": persona, "tree": tree}

if __name__ == "__main__":
    # Verify Ollama service is running first (`ollama serve` in another terminal)
    dataset = []
    
    for i in range(NUM_SAMPLES):
        try:
            sample = generate_sample()
            dataset.append(sample)
            print(f"Generated sample {i+1}/{NUM_SAMPLES}")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(f"Error generating sample {i+1}: {str(e)}")
    
    with open(os.path.join(RAW_DATA_PATH,'file_systems_dataset2.json'), 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nDataset saved to file_systems_dataset2.json ({len(dataset)} samples)")

