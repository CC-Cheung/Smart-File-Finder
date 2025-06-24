import argparse
import os
import ollama  # Make sure ollama is installed and running

MODEL_NAME = 'llama3.2'  # Change to your fine-tuned model if needed

def list_tree(root, max_depth=2, prefix='~'):
    """Builds a flat list of files/folders up to max_depth."""
    tree = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            continue
        rel_dir = os.path.relpath(dirpath, os.path.expanduser('~'))
        rel_dir = prefix if rel_dir == '.' else f"{prefix}/{rel_dir}"
        tree.append(rel_dir)
        for d in dirnames:
            tree.append(f"{rel_dir}/{d}")
        for f in filenames:
            tree.append(f"{rel_dir}/{f}")
    return sorted(set(tree))

def prompt_model(description, tree):
    """Send prompt to Ollama and get response."""
    prompt = (
        f"File/Folder description: {description}\n"
        "Pick from one of the following:\n"
        + "\n".join(tree)
    )
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": (
                "You are a file retrieval assistant. Given the following file/folder description and a partial folder tree, "
                "if the desired file/folder is visible, output it, else, predict the next folder to explore.\n"
            )},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content'].strip()

def main():
    parser = argparse.ArgumentParser(description="Smart File Finder with Ollama model")
    parser.add_argument("description", help="File/folder description to search for")
    parser.add_argument(
        "--start",
        default=os.getcwd(),
        help="Start folder for the search (default: current working directory)"
    )
    args = parser.parse_args()

    description = args.description
    cwd = os.path.abspath(os.path.expanduser(args.start))
    max_depth = 1

    while True:
        tree = list_tree(cwd, max_depth=max_depth)
        suggestion = prompt_model(description, tree)
        print(f"\nModel suggestion: {suggestion}")

        # If the suggestion is a file/folder in the current tree, print and exit
        if suggestion in tree:
            print(f"\nFound: {suggestion}")
            break
        # If it's a folder, cd into it and continue
        elif any(suggestion == path for path in tree if os.path.isdir(os.path.join(cwd, os.path.relpath(path, '~')))):
            print(f"Descending into: {suggestion}")
            # Convert suggestion back to real path
            next_dir = os.path.join(cwd, os.path.relpath(suggestion, '~'))
            cwd = os.path.abspath(next_dir)
            max_depth = 1  # Reset depth for next iteration
        else:
            print("Model did not return a valid path. Exiting.")
            break

if __name__ == "__main__":
    main()
