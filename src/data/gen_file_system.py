# synthetic_fs_generator.py
import ollama
import time
import json
import os
from faker import Faker
FILE_PATH=os.path.dirname(os.path.abspath(__file__))
CODE_PATH= os.path.join(FILE_PATH, '..', '..')
DATA_PATH=os.path.join(CODE_PATH, 'data')
PROCESSED_DATA_PATH=os.path.join(DATA_PATH, 'processed')
RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
FILE_NAME="file_systems_dataset.json"

NUM_SAMPLES = 100  # Number of synthetic file systems to generate
MODEL = 'dolphin3:latest'   # Name of your local model (verify with `ollama list`)
SLEEP_TIME = 1    # Delay between requests to avoid overloading local model
# persona_prompt = """Tell me about {person} and describe how their file system might differ. Keep within 3 sentences."""

TREE_PROMPT_TEMPLATE = """Create the file system structure starting from ~ for {name}, {sex}, born on {date_of_birth} who is a {profession} at {company}.


Output ONLY:
<START>
File structure with the number of tabs representing it's depth
<END>

Example of the Format (If the person was a CS university graduate who liked music):
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
            Fur Elise.mp3
        Pop
            Bad Guy.mp3
            Shape of You.mp3
    High School
        Pictures
            Bali.jpg
            Grade 12 Band Trip
                Denver
                    0102.jpg
                    0103.jpg
                    0104.jpg
                    0105.jpg
                Toronto
                    0201.jpg
                    0202.jpg 
                    0203.jpg
        Biology
            Biology Notes.docx
            Biology Notes 2.docx
        Chemistry
            Final Report.docx
        Math
            Calculus
                Calculus Notes.docx
                Final Exam.docx
            Suderland Grade 12 Math Textbook.pdf
    Software Exe
        pycharm.exe
        vscode.exe 
        Adobe
            Adobe Reader.exe
            Adobe Illustrator Crack.exe
        
<END>
"""

# Enhanced prompt template (From perplexity, bad because ads the file percentage after each folder)
# TREE_PROMPT_TEMPLATE = """Create a file system structure starting from ~ for:
# - Name: {name}
# - Gender: {sex}
# - Born: {date_of_birth}
# - Occupation: {profession} at {company}

# Guidelines:
# 1. Use 1-5 main directories (e.g., Work, Personal)
# 2. Max depth: 4 levels
# 3. Mix files (60%) and directories (40%)
# 4. Use realistic extensions (.docx, .jpg, .py)
# 5. Avoid special characters in filenames

# Format:
# <START>
# ~
#     Directory1
#         Subdir1
#             file1.ext
#     Directory2
#         file2.ext
# <END>
# """

def generate_sample():
    # Generate persona
    profile = fake.profile()
    # persona_response = ollama.chat(
    #     model=MODEL,
    #     messages=[{"role": "user", "content": persona_prompt]
    # )
    # persona = persona_response['message']['content'].strip()
    
    # Generate file tree
    tree_prompt=TREE_PROMPT_TEMPLATE.format(name=profile['name'], 
                                            sex=profile['sex'], 
                                            date_of_birth=str(profile['birthdate']), 
                                            profession=profile['job'], 
                                            company=profile['company'])
    tree_response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": tree_prompt}]
    )
    tree = tree_response['message']['content'].strip()
    
    return {"prompt": tree_prompt, "tree": tree}

if __name__ == "__main__":
    # Verify Ollama service is running first (`ollama serve` in another terminal)
    dataset = []
    fake=Faker()
    for i in range(NUM_SAMPLES):
        try:
            sample = generate_sample()
            dataset.append(sample)
            print(f"Generated sample {i+1}/{NUM_SAMPLES}")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(f"Error generating sample {i+1}: {str(e)}")
    
    with open(os.path.join(RAW_DATA_PATH,FILE_NAME), 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nDataset saved to {FILE_NAME} ({len(dataset)} samples)")

